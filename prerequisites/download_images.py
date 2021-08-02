"""A script to download images for all different SGG datasets used."""

import os
import shutil
import tarfile
from zipfile import ZipFile


VRD = 'http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip'
VG = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip'
VG2 = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'
UNREL = 'http://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz'


def download_images(data_folder):
    """Download all images."""
    if not os.path.exists(data_folder + 'VRD/'):
        print('Dowloading VRD images...')
        download_vrd(data_folder)
    if not os.path.exists(data_folder + 'VG/'):
        print('Dowloading VG images...')
        download_vg(data_folder)
    if not os.path.exists(data_folder + 'UnRel/'):
        print('Dowloading UnRel images...')
        download_unrel(data_folder)
    if not os.path.exists(data_folder + 'COCO/'):
        print('Dowloading COCO images...')
        download_coco(data_folder)
    for dataset in ['VRD', 'VG', 'UnRel']:
        if not os.path.exists(data_folder + dataset + '/sample_images/'):
            sample_images(data_folder, dataset)


def download_vrd(data_folder):
    """Download VRD images."""
    if not os.path.exists(data_folder + 'VRD/'):
        os.mkdir(data_folder + 'VRD/')
    if not os.path.exists(data_folder + 'VRD/images/'):
        os.mkdir(data_folder + 'VRD/images/')
    data_folder = str(data_folder + 'VRD/images/')
    # Download zip with images
    os.system("wget " + VRD)
    shutil.move('sg_dataset.zip', data_folder + 'sg_dataset.zip')
    with ZipFile(data_folder + 'sg_dataset.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'sg_dataset.zip')
    # Move all images to specified path
    for name in os.listdir(data_folder + 'sg_dataset/sg_train_images/'):
        shutil.move(
            data_folder + 'sg_dataset/sg_train_images/' + name,
            data_folder + name
        )
    for name in os.listdir(data_folder + 'sg_dataset/sg_test_images/'):
        shutil.move(
            data_folder + 'sg_dataset/sg_test_images/' + name,
            data_folder + name
        )
    shutil.rmtree(data_folder + 'sg_dataset')


def download_vg(data_folder):
    """Download VG images."""
    if not os.path.exists(data_folder + 'VG/'):
        os.mkdir(data_folder + 'VG/')
    if not os.path.exists(data_folder + 'VG/images/'):
        os.mkdir(data_folder + 'VG/images/')
    data_folder = str(data_folder + 'VG/images/')
    # Download zips with images
    os.system("wget " + VG)
    shutil.move('images.zip', data_folder + 'images.zip')
    with ZipFile(data_folder + 'images.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'images.zip')
    os.system("wget " + VG2)
    shutil.move('images2.zip', data_folder + 'images2.zip')
    with ZipFile(data_folder + 'images2.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'images2.zip')
    # Move all images to specified path
    for name in os.listdir(data_folder + 'VG_100K/'):
        shutil.move(
            data_folder + 'VG_100K/' + name,
            data_folder + name
        )
    for name in os.listdir(data_folder + 'VG_100K_2/'):
        shutil.move(
            data_folder + 'VG_100K_2/' + name,
            data_folder + name
        )
    shutil.rmtree(data_folder + 'VG_100K')
    shutil.rmtree(data_folder + 'VG_100K_2')


def download_unrel(data_folder):
    """Download UnRel images."""
    if not os.path.exists(data_folder + 'UnRel/'):
        os.mkdir(data_folder + 'UnRel/')
    if not os.path.exists(data_folder + 'UnRel/images/'):
        os.mkdir(data_folder + 'UnRel/images/')
    data_folder = str(data_folder + 'UnRel/images/')
    # Download both VRD and UnRel images
    os.system("wget " + VRD)
    shutil.move('sg_dataset.zip', data_folder + 'sg_dataset.zip')
    with ZipFile(data_folder + 'sg_dataset.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'sg_dataset.zip')
    os.system("wget " + UNREL)
    shutil.move('unrel-dataset.tar.gz', data_folder + 'unrel-dataset.tar.gz')
    with tarfile.open(data_folder + 'unrel-dataset.tar.gz') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'unrel-dataset.tar.gz')
    # Move all images to specified path
    for name in os.listdir(data_folder + 'sg_dataset/sg_train_images/'):
        shutil.move(
            data_folder + 'sg_dataset/sg_train_images/' + name,
            data_folder + name
        )
    for name in os.listdir(data_folder + 'sg_dataset/sg_test_images/'):
        shutil.move(
            data_folder + 'sg_dataset/sg_test_images/' + name,
            data_folder + name
        )
    for name in os.listdir(data_folder + 'images/'):
        shutil.move(data_folder + 'images/' + name, data_folder + name)
    shutil.rmtree(data_folder + 'sg_dataset')
    shutil.rmtree(data_folder + 'images')


def download_coco(data_folder):
    """Download COCO images."""
    if not os.path.exists(data_folder + 'COCO/'):
        os.mkdir(data_folder + 'COCO/')
    if not os.path.exists(data_folder + 'COCO/images/'):
        os.mkdir(data_folder + 'COCO/images/')
    data_folder = str(data_folder + 'COCO/images/')
    # Download train and val images
    os.system("wget http://images.cocodataset.org/zips/train2017.zip")
    shutil.move('train2017.zip', data_folder + 'train2017.zip')
    with ZipFile(data_folder + 'train2017.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'train2017.zip')
    os.system("wget http://images.cocodataset.org/zips/val2017.zip")
    shutil.move('val2017.zip', data_folder + 'val2017.zip')
    with ZipFile(data_folder + 'val2017.zip') as fid:
        fid.extractall(data_folder)
    os.remove(data_folder + 'val2017.zip')
    # Move all images to specified path
    for name in os.listdir(data_folder + 'train2017/'):
        shutil.move(
            data_folder + 'train2017/' + name,
            data_folder + name
        )
    for name in os.listdir(data_folder + 'val2017/'):
        shutil.move(
            data_folder + 'val2017/' + name,
            data_folder + name
        )
    shutil.rmtree(data_folder + 'train2017')
    shutil.rmtree(data_folder + 'val2017')


def sample_images(data_folder, dataset):
    """Pick sample images for demo purposes."""
    os.mkdir(data_folder + dataset + '/sample_images/')
    samples = sorted(os.listdir(data_folder + dataset + '/images/'))[:100]
    for sample in samples:
        shutil.copy(
            data_folder + dataset + '/images/' + sample,
            data_folder + dataset + '/sample_images/' + sample
        )
