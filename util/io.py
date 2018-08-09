import torch
import yaml

cpu_device = torch.device('cpu')


def save_model(model, filename):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    torch.save(state_dict, filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))


def save_models_dict(models_dict, filename):
    # key: name of model, value: model
    assert isinstance(models_dict, dict)
    result = {}
    for name, model in models_dict.items():
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(cpu_device)
        result[name] = state_dict
    torch.save(result, filename)


def load_models_dict(models_dict, filename):
    # key: name of model, value: uninitialized model
    assert isinstance(models_dict, dict)
    result_dict = torch.load(filename)
    for key in models_dict.keys():
        models_dict[key].load_state_dict(result_dict[key])


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
