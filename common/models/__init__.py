"""Import all different models here."""

import torch

from .sg_generator import (
    ATRNet, RelDN, UVTransE, MotifsNet, VTransENet, HGATNet,
)
from .grounder import (
    ParsingNet
)

MODELS = {
    'atr_net': ATRNet,
    'motifs_net': MotifsNet,
    'atr_teacher_net': ATRNet,
    'hgat_net': HGATNet,
    'reldn_net': RelDN,
    'uvtranse_net': UVTransE,
    'vtranse_net': VTransENet,
    'parsing_net': ParsingNet,
}


def load_model(config, model, path=None):
    """Load model given a net name."""
    model_dir = config.paths['models_path']\
        if path is None else path
    net = MODELS[model](config)
    checkpoint = torch.load(model_dir + 'model.pt',
                            map_location=config.device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return net.to(config.device)
