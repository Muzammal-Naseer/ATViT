import torch
import torch.nn.functional as F

from utils.gaussian_blur import gaussian_blur

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def input_diversity(img):
    rnd = torch.randint(224, 257, (1,)).item()
    rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
    h_rem = 256 - rnd
    w_hem = 256 - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_hem + 1, (1,)).item()
    pad_right = w_hem - pad_left
    padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
    padded = F.interpolate(padded, (224, 224), mode='nearest')
    return padded


def local_adv(model, criterion, img, label, eps, attack_type, iters, mean, std, index, apply_ti=False):
    adv = img.detach()

    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    if attack_type in ['fgsm', 'rfgsm']:
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

    adv_noise = 0
    for j in range(iterations):
        if attack_type == 'dim':
            adv_r = input_diversity(adv)
        else:
            adv_r = adv
        # out_adv = model(normalize(torch.nn.functional.interpolate(adv_r.clone(), (224, 224)), mean=mean, std=std))
        out_adv = model(normalize(adv_r.clone(), mean=mean, std=std))

        loss = 0
        if isinstance(out_adv, list) and index == 'all':
            loss = 0
            for idx in range(len(out_adv)):
                loss += criterion(out_adv[idx], label)
        elif isinstance(out_adv, list) and index == 'last':
            loss = criterion(out_adv[-1], label)
        else:
            loss = criterion(out_adv, label)

        loss.backward()
        if apply_ti:
            adv.grad = gaussian_blur(adv.grad, kernel_size=(15, 15), sigma=(3, 3))

        if attack_type == 'mifgsm' or attack_type == 'dim':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * adv_noise.sign()

        # Projection
        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()


def local_adv_target(model, criterion, img, target, eps, attack_type, iters, mean, std, index):
    adv = img.detach()

    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    if attack_type in ['fgsm', 'rfgsm']:
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

    adv_noise = 0
    for j in range(iterations):

        if attack_type == 'dim':
            adv_r = input_diversity(adv)
        else:
            adv_r = adv
        out_adv = model(normalize(adv_r.clone(), mean=mean, std=std))

        loss = 0
        if isinstance(out_adv, list) and index == 'all':
            loss = 0
            for idx in range(len(out_adv)):
                loss += criterion(out_adv[idx], target)
        elif isinstance(out_adv, list) and index == 'last':
            loss = criterion(out_adv[-1], target)
        else:
            loss = criterion(out_adv, target)

        loss.backward()

        if attack_type == 'mifgsm' or attack_type == 'dim':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        # Optimization step
        adv.data = adv.data - step * adv_noise.sign()

        # Projection
        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()
