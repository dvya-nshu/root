import torch.nn as nn

def parse_layer(layer):

    # detect ELU layer
    if isinstance(layer, nn.ELU):

        return {
            "type": "ELU",
            "alpha": layer.alpha
        }

    return None


def parse_model(model):

    parsed_layers = []

    for name, layer in model.named_modules():

        info = parse_layer(layer)

        if info is not None:
            parsed_layers.append(info)

    return parsed_layers


