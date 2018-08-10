from torchvision import transforms


def get_composed_transforms(hflip=False):
    transforms_list = []
    if hflip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    shared_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # -1.0 ~ 1.0
    transforms_list += shared_transforms
    return transforms.Compose(transforms_list)
