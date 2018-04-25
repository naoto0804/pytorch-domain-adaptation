# This is from Pytorch-CycleGAN-and-pix2pix

import torch


class GANLoss(torch.nn.Module):
    def __init__(self, device, use_lsgan=True, target_real_label=1.0,
                 target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        if use_lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_tensor is None) or
                            (self.real_label_tensor.numel() != input.numel()))
            if create_label:
                # self.real_label_tensor = self.Tensor(
                #     input.size(), requires_grad=False).fill_(self.real_label)
                self.real_label_tensor = torch.FloatTensor(input.size())
                self.real_label_tensor.fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            target_tensor = self.real_label_tensor.to(self.device)
        else:
            create_label = ((self.fake_label_tensor is None) or
                            (self.fake_label_tensor.numel() != input.numel()))
            if create_label:
                # self.fake_label_tensor = self.Tensor(
                #     input.size(), requires_grad=False).fill_(self.fake_label)
                self.fake_label_tensor = torch.FloatTensor(input.size())
                self.fake_label_tensor.fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            target_tensor = self.fake_label_tensor.to(self.device)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
