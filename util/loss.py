# This is from Pytorch-CycleGAN-and-pix2pix

import torch


class GANLoss(torch.nn.Module):
    def __init__(self, device, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label = None
        self.fake_label = None
        if use_lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label is None) or
                            (self.real_label.numel() != input.numel()))
            if create_label:
                # self.real_label_tensor = self.Tensor(
                #     input.size(), requires_grad=False).fill_(self.real_label)
                # self.real_label = torch.FloatTensor()
                # self.real_label.fill_(self.real_label)
                self.real_label = torch.ones(
                    input.size(), requires_grad=False, device=self.device)
            target_tensor = self.real_label
        else:
            create_label = ((self.fake_label is None) or
                            (self.fake_label.numel() != input.numel()))
            if create_label:
                # self.fake_label_tensor = self.Tensor(
                #     input.size(), requires_grad=False).fill_(self.fake_label)
                self.fake_label = torch.zeros(
                    input.size(), requires_grad=False, device=self.device)
            target_tensor = self.fake_label
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
