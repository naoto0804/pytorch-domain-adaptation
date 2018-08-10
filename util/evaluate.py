import torch
import torch.nn.functional as F


def evaluate_classifier(classifier, loader, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    classifier.eval()

    n_err = 0
    with torch.no_grad():
        for x, y in loader:
            prob_y = F.softmax(classifier(x.to(device)), dim=1)
            pred_y = torch.max(prob_y, dim=1)[1]
            pred_y = pred_y.to(torch.device('cpu'))
            n_err += (pred_y != y).sum().item()

    classifier.train()

    return n_err
