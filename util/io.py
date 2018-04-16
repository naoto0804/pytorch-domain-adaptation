import torch


def save_model(net, filename):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, filename)


def load_model(net, filename):
    net.load_state_dict(torch.load(filename))
