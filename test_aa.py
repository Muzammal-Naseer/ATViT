import argparse
import datetime
import json
import os
import random
from collections import defaultdict

import torch
from autoattack import AutoAttack
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import vit_models
from attack import normalize
from dataset import AdvImageNet

targeted_class_dict = {
    24: "Great Grey Owl",
    99: "Goose",
    245: "French Bulldog",
    344: "Hippopotamus",
    471: "Cannon",
    555: "Fire Engine",
    661: "Model T",
    701: "Parachute",
    802: "Snowmobile",
    919: "Street Sign ",
}


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_dir', default='data', help='ImageNet Validation Data')
    parser.add_argument('--dataset', default="imagenet_1k", help='dataset name')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--tar_model', type=str, nargs="+", default=['tnt_s_patch16_224', ], help='Target Model Name')
    parser.add_argument('--src_pretrained', type=str, default=None, help='pretrained path for source model')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--eps', type=int, default=8, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='last', help='last or all')
    parser.add_argument('--tar_ensemble', action="store_true", default=False)
    parser.add_argument('--apply_ti', action="store_true", default=False)
    parser.add_argument('--save_im', action="store_true", default=False)

    return parser.parse_args()


def get_model(model_name):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    other_model_names = vars(vit_models)

    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'hierarchical' in model_name or "ensemble" in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return model, mean, std


#  Test Samples
def get_data_loader(args, verbose=True):
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    test_dir = args.test_dir
    if args.dataset == "imagenet_1k":
        test_set = AdvImageNet(image_list="data/image_list_1k.json", root=test_dir, transform=data_transform)
    else:
        test_set = AdvImageNet(root=test_dir, transform=data_transform)
    test_size = len(test_set)
    if verbose:
        print('Test data size:', test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    return test_loader, test_size


def main():
    # setup run
    args = parse_args()
    args.exp = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{random.randint(1, 100)}"
    os.makedirs(f"report/{args.exp}")
    json.dump(vars(args), open(f"report/{args.exp}/config.json", "w"), indent=4)

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load source and target models
    src_model, src_mean, src_std = get_model(args.src_model)
    if args.src_pretrained is not None:
        if args.src_pretrained.startswith("https://"):
            src_checkpoint = torch.hub.load_state_dict_from_url(args.src_pretrained, map_location='cpu')
        else:
            src_checkpoint = torch.load(args.src_pretrained, map_location='cpu')
        src_model.load_state_dict(src_checkpoint['model'])
    src_model = src_model.to(device)
    src_model.eval()

    tar_models, tar_means, tar_stds = [], [], []
    for tar_model_name in args.tar_model:
        temp_model, temp_mean, temp_std = get_model(tar_model_name)
        temp_model = temp_model.to(device)
        temp_model.eval()
        tar_models.append(temp_model)
        tar_means.append(temp_mean)
        tar_stds.append(temp_std)

    # Setup-Data
    test_loader, test_size = get_data_loader(args)

    # setup attack parameters
    eps = args.eps / 255

    if args.index == "last":
        def forward_pass(images):
            output = src_model(normalize(images.clone(), mean=src_mean, std=src_std))
            if isinstance(output, list):
                output = output[-1]
            return output
    else:
        def forward_pass(images):
            output = src_model(normalize(images.clone(), mean=src_mean, std=src_std))
            if isinstance(output, list):
                output = torch.stack(output, dim=0).sum(dim=0)
            return output

    adversary = AutoAttack(forward_pass, norm='Linf', eps=eps, version='standard', verbose=True,
                           log_path=f"report/{args.exp}/aa_results.log")

    pair_list = [(x, y) for x, y in test_loader]
    img_list = [x for x, _ in pair_list]
    img_list = torch.cat(img_list, 0)  # B, 3, H, W
    label_list = [y for _, y in pair_list]
    label_list = torch.cat(label_list, 0)  # B
    with torch.no_grad():
        adv_list = adversary.run_standard_evaluation(img_list, label_list, bs=args.batch_size)  # B, 3, H, W

    tar_clean_acc, tar_adv_acc, tar_fool_rate, = defaultdict(int), defaultdict(int), defaultdict(int)

    for tar_idx, tar_model_name in enumerate(args.tar_model):
        cur_tar_model = tar_models[tar_idx]
        cur_tar_mean = tar_means[tar_idx]
        cur_tar_std = tar_stds[tar_idx]

        for img, adv, label in tqdm(zip(img_list, adv_list, label_list), total=len(label_list)):
            img, adv, label = img.to(device), adv.to(device), label.to(device)
            img, adv, label = img.unsqueeze(0), adv.unsqueeze(0), label.unsqueeze(0)

            with torch.no_grad():
                clean_out = cur_tar_model(normalize(img.clone(), mean=cur_tar_mean, std=cur_tar_std))
                if isinstance(clean_out, list):
                    clean_out = clean_out[-1].detach()
                tar_clean_acc[tar_model_name] += torch.sum(clean_out.argmax(dim=-1) == label).item()

                adv_out = cur_tar_model(normalize(adv.clone(), mean=cur_tar_mean, std=cur_tar_std))
                if isinstance(adv_out, list):
                    adv_out = adv_out[-1].detach()
                tar_adv_acc[tar_model_name] += torch.sum(adv_out.argmax(dim=-1) == label).item()
                tar_fool_rate[tar_model_name] += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

    json.dump({"eps": int(args.eps),
               "tar clean": {x: y / test_size for x, y in tar_clean_acc.items()},
               "tar adv": {x: y / test_size for x, y in tar_adv_acc.items()},
               "tar fool rate": {x: y / test_size for x, y in tar_fool_rate.items()},
               },
              open(f"report/{args.exp}/results.json", "w"), indent=4)


if __name__ == '__main__':
    main()
