from torchvision import transforms


def get_composed_transforms(crop_res=None, hflip=False):
    transforms_list = []
    if hflip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if crop_res is not None:
        size = (crop_res, crop_res)
        transforms_list.append(transforms.RandomCrop(size, padding=2))
    shared_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # -1.0 ~ 1.0
    transforms_list += shared_transforms
    return transforms.Compose(transforms_list)
