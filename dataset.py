import json

import torchvision


class AdvImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, image_list="data/image_list.json", *args, **kwargs):
        self.image_list = set(json.load(open(image_list, "r"))["images"])
        super(AdvImageNet, self).__init__(is_valid_file=self.is_valid_file, *args, **kwargs)

    def is_valid_file(self, x: str) -> bool:
        return x[-38:] in self.image_list


if __name__ == '__main__':
    data_path = "/home/kanchanaranasinghe/data/raw/imagenet/val"
    transform = torchvision.transforms.ToTensor()
    dataset = AdvImageNet(root=data_path, transform=transform)
