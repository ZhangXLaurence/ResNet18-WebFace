import os
import random
import tarfile
from math import ceil, floor
from tqdm import tqdm
import cv2
import requests

from torch.utils import data

DATASET_TARBALL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
PAIRS_TRAIN = "http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt"
PAIRS_VAL = "http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt"

def create_datasets(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    dataroot_files = os.listdir(dataroot)
    data_tarball_file = DATASET_TARBALL.split('/')[-1]
    data_dir_name = data_tarball_file.split('.')[0]

    if data_dir_name not in dataroot_files:
        if data_tarball_file not in dataroot_files:
            tarball = download(dataroot, DATASET_TARBALL)
        with tarfile.open(tarball, 'r') as t:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, dataroot)

    images_root = os.path.join(dataroot, 'lfw-deepfunneled')
    names = os.listdir(images_root)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    training_set = []
    validation_set = []
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        images_of_person = os.listdir(os.path.join(images_root, name))
        total = len(images_of_person)

        training_set += map(
                add_class,
                images_of_person[:ceil(total * train_val_split)])
        validation_set += map(
                add_class,
                images_of_person[floor(total * train_val_split):])

    return training_set, validation_set, len(names)


class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])


class PairedDataset(data.Dataset):
    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader
        self.image_names_a = []
        self.image_names_b = []
        self.matches = []
        self._prepare_dataset()
    def __len__(self):
        return len(self.matches)
    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])
    def _prepare_dataset(self):
        raise NotImplementedError


class LFWPairedDataset(PairedDataset):
    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)
        for pair in pairs:
            if len(pair) == 3:
                match = True
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])
            else:
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])
            self.image_names_a.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name1, "{}_{:04d}.jpg".format(name1, index1)))
            self.image_names_b.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name2, "{}_{:04d}.jpg".format(name2, index2)))
            self.matches.append(match)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs




def download(dir, url, dist=None):
    dist = dist if dist else url.split('/')[-1]
    print('Start to Download {} to {} from {}'.format(dist, dir, url))
    download_path = os.path.join(dir, dist)
    if os.path.isfile(download_path):
        print('File {} already downloaded'.format(download_path))
        return download_path
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(download_path, 'wb') as f:
        for data in tqdm(
                r.iter_content(block_size),
                total=ceil(total_size//block_size),
                unit='MB', unit_scale=True):
            f.write(data)
    print('Downloaded {}'.format(dist))
    return download_path


def image_loader(image_path):
    return cv2.imread(image_path)