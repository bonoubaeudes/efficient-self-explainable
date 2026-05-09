import torch

def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg["training"]["optimizer"]
    learning_rate = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    momentum = cfg["training"]["momentum"]
    nesterov = cfg["training"]["nesterov"]
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer