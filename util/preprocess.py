import numpy as np
import scipy.ndimage as ndi
from torchvision import transforms


def get_composed_transforms(train=True, hflip=False):
    transforms_list = []
    if train and hflip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    shared_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_list += shared_transforms
    return transforms.Compose(transforms_list)


# from https://github.com/engharat/SBADAGAN/ImageDataGenerator.py
class OriginalAffineTransform(object):
    def __init__(self):
        # HWC style
        self.channel_index = 3
        self.row_index = 1
        self.col_index = 2

        self.rotation_range = 45
        self.height_shift_range = 0.1
        self.width_shift_range = 0.1
        self.shear_range = None
        self.zoom_range = [1, 1]
        self.fill_mode = 'nearest'
        self.cval = 0.

    @staticmethod
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def apply_transform(x, transform_matrix, channel_index=0,
                        fill_mode='nearest', cval=0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [
            ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                               final_offset, order=0,
                                               mode=fill_mode, cval=cval) for
            x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x

    def __call__(self, x):
        # Need to modify to transform both X and Y ---- to do
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * x.shape[
                     img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * x.shape[
                     img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1],
                                       2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix),
            zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = self.transform_matrix_offset_center(
            transform_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, img_channel_index,
                                 fill_mode=self.fill_mode, cval=self.cval)
        return x
