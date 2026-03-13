import torch.nn as nn

def parse_layer(layer):

    if isinstance(layer, nn.ELU):
        return {
            "type": "ELU",
            "alpha": layer.alpha
        }
    if isinstance(layer, nn.MaxPool2d):

        return {
        "type": "MaxPool2D",
        "kernel_size": layer.kernel_size,
        "stride": layer.stride
    }
    if isinstance(layer, nn.BatchNorm2d):

        return {
        "type": "BatchNorm2D",
        "num_features": layer.num_features,
        "eps": layer.eps,
        "momentum": layer.momentum
    }
    return None


def parse_model(model):

    parsed_layers = []

    for name, layer in model.named_modules():

        info = parse_layer(layer)

        if info is not None:
            parsed_layers.append(info)

    return parsed_layers