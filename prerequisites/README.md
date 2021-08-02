# Deeplab Scene Graph Generation repo: data

Functions to download images and transform annotations to a standard format.

## Setup
Clone the repository and
```
cd deeplab_sgg/data/
pip install .
cd data/
./main.sh
```

Inner pipeline (for reference):
Download images
```
./scripts/download_images.sh OPTIONS
```
OPTIONS can be "VG", "VRD", "UnRel" for the respective dataset family or "all".

Download annotations DATASET_NAMES
```
./scripts/download_data.sh
```
DATASET_NAMES can "VRD", "VG200", "VG80K", "VGMSDN", "VGVTE", "VrR-VG", "sVG", "UnRel" or "all"

Download GloVe
```
./scripts/download_glove_vectors.sh
```

Transform annotations and create project folders
```
python3 prepare_data.py DATASET_NAMES
```
If no dataset name is provided the default is "all"!