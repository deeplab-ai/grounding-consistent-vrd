# -*- coding: utf-8 -*-
"""Functions to transform annotations and create train/test dataset."""

import os
import sys

import yaml

from prerequisites.data_config import DataConfig
from prerequisites.dataset_transformers import (
    VG200Transformer, VG80KTransformer, VGMSDNTransformer, VGVTETransformer,
    VRDTransformer, VrRVGTransformer, SVGTransformer, UnRelTransformer,
    COCOTransformer
)
from prerequisites.download_images import download_images

with open('prerequisites_config.yaml', 'r') as fid:
    CONFIG = yaml.load(fid, Loader=yaml.FullLoader)

TRANSFORMERS = {
    'VG200': VG200Transformer,
    'VG80K': VG80KTransformer,
    'VGMSDN': VGMSDNTransformer,
    'VGVTE': VGVTETransformer,
    'VRD': VRDTransformer,
    'VrR-VG': VrRVGTransformer,
    'sVG': SVGTransformer,
    'UnRel': UnRelTransformer,
    'COCO': COCOTransformer
}


def main(datasets):
    """Run the data preprocessing and creation pipeline."""
    if os.path.exists('/gpu-data/mdiom/'):
        _path = '/gpu-data/mdiom/'
    elif os.path.exists('/gpu-data2/ngan/'):
        _path = '/gpu-data2/ngan/'
    else:
        _path = CONFIG['prerequisites_path']
    download_images(_path)
    for dataset in datasets:
        print('Creating annotations for ' + dataset)
        TRANSFORMERS[dataset](DataConfig(_path, dataset)).transform()
    print('Done.')


if __name__ == "__main__":
    if any(sys.argv[1:]):
        main(sys.argv[1:])
    else:
        main([
            'VG80K', 'sVG', 'VrR-VG', 'VGVTE',
            'VGMSDN', 'VG200', 'VRD', 'UnRel', 'COCO'
        ])
