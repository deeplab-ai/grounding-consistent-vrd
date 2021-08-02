"""Import all different models here."""

import torch

from .object_classifier import ObjectClassifier
from .object_detector import ObjectDetector
from .sg_generator import (
    ATRNet, LangSpatNet, LanguageNet, RelDN, SpatialNet, UVTransE, VisualNet,
    MotifsNet, GPSNet, VTransENet, HGATNet,
    VisualSpatAttentionNet, VisualSpatNet, SpatLangRank, LangSpatDepthNet,
    VisLangSpatNet
)
from .sg_projector import (
    LangSpatProjector, LanguageProjector, SpatialProjector, VisualProjector,
    VisualSpatAttentionProjector, VisualSpatProjector
)
from .grounder import (
    GroundBboxNet, RRCondNet, RRCombNet,
    RRParNet, ParsingNet, ParsingDynLangNet, SeqAtt
)

MODELS = {
    'atr_net': ATRNet,
    'language_net': LanguageNet,
    'language_projector': LanguageProjector,
    'lang_spat_net': LangSpatNet,
    # 'lang_spat_depth_net': LangSpatDepthNet,
    'lang_spat_projector': LangSpatProjector,
    'lang_spat_projector_deepGCL_scaled_net': LangSpatProjector,
    'lang_spat_projector_9_net': LangSpatProjector,
    'motifs_net': MotifsNet,
    'visual_lang_spat_net': VisLangSpatNet,
    'visual_lang_spat_neg_oracle_net': VisLangSpatNet,
    'visual_lang_spat_graphl_net': VisLangSpatNet,
    'visual_lang_spat_consistency_epoch1_bg02_net': VisLangSpatNet,
    'atr_consistency_epoch1_bg02_net': ATRNet,
    'vtranse_consistency_epoch1_bg02_net': VTransENet,
    'uvtranse_consistency_epoch1_bg02_net': UVTransE,
    'hgat_consistency_epoch1_bg02_net': HGATNet,
    'reldn_consistency_epoch1_bg02_net': RelDN,
    'motifs_consistency_epoch1_bg02_net': MotifsNet,
    'visual_lang_spat_spatdistill_bg02_net': VisLangSpatNet,
    'gps_net': GPSNet,
    'hgat_net': HGATNet,
    'object_classifier': ObjectClassifier,
    'object_detector': ObjectDetector,
    'reldn_net': RelDN,
    # 'reldn_depth_net': RelDNDepth,
    # 'depth_net': DepthNet,
    'spatial_net': SpatialNet,
    'spatlang_negatives_ranker_net': SpatLangRank,
    'spatial_projector': SpatialProjector,
    'uvtranse_net': UVTransE,
    'visual_net': VisualNet,
    'vtranse_net': VTransENet,
    'visual_projector': VisualProjector,
    'visual_spat_attention_net': VisualSpatAttentionNet,
    'visual_spat_attention_projector': VisualSpatAttentionProjector,
    'visual_spat_net': VisualSpatNet,
    'visual_spat_projector': VisualSpatProjector,
    'ground_bbox_net': GroundBboxNet,
    'ref_rel_conditioned_net': RRCondNet,
    'ref_rel_combinator_net': RRCombNet,
    'ref_rel_parallel_net': RRParNet,
    'parsing_net': ParsingNet,
    'seq_att_net': SeqAtt,
    'parsing_dyn_lang_net': ParsingDynLangNet,
    'parsing_gauss_freezelang_commonlang_05cov_net': ParsingNet
}


def load_model(config, model, net_name, path=None):
    """Load model given a net name."""
    model_dir = config.paths['models_path']\
        if path is None else path
    net = MODELS[model](config)
    checkpoint = torch.load(model_dir + 'model.pt',
                            map_location=config.device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return net.to(config.device)
