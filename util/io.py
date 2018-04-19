import torch


def save_model(model, filename):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))


def save_models_dict(models_dict, filename):
    result = {}
    for name, model in models_dict.items():
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        result[name] = state_dict
    torch.save(result, filename)


def load_models_dict(models_dict, filename):
    result_dict = torch.load(filename)
    for key in models_dict.keys():
        models_dict[key].load_state_dict(result_dict[key])
