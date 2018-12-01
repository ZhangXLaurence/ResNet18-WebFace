import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

from . import CSVDataset
from . import DataTransformCompose
from . import LFWDataset

# Load MNIST
def LoadMNIST(train_batch_size, test_batch_size, path):
    trainset = datasets.MNIST(path, download=True, train=True, transform=DataTransformCompose.TransformMNIST())
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.MNIST(path, download=True, train=False, transform=DataTransformCompose.TransformMNIST())
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load Fashion-MNIST
def LoadFashionMNIST(train_batch_size, test_batch_size, path):
    trainset = datasets.FashionMNIST(path, download=True, train=True, transform=DataTransformCompose.TransformMNIST())
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.FashionMNIST(path, download=True, train=False, transform=DataTransformCompose.TransformMNIST())
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load CSV
def LoadCSV(train_batch_size, test_batch_size, cvs_file_dir):
    trainset = CSVDataset.CSVDataset(cvs_file_dir)
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = CSVDataset.CSVDataset(cvs_file_dir)
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load Single Face-label from image fold;
def LoadFaceImgFoldData(train_batch_size, data_path, transform=None):
    """
    Args: Training batch size and path to training face image folder
    Return: Training data loader and number of training class
    """
    trainset = datasets.ImageFolder(data_path, transform=transform)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    class_num = len([lists for lists in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, lists))])
    # On going
    return train_loader, class_num

# Load CIFAR10
def LoadCIFAR10(train_batch_size, test_batch_size, path, arg_inputsize=224):
    trainset = datasets.CIFAR10(path, download=True, train=True, transform=DataTransformCompose.TransformCIFAR10(True, arg_inputsize))
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.CIFAR10(path, download=True, train=False, transform=DataTransformCompose.TransformCIFAR10(False, arg_inputsize))
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load LFW Training
def LoadLFW(train_batch_size, test_batch_size, path, arg_inputsize):
    training_set, validation_set, num_classes = LFWDataset.create_datasets(path)
    training_dataset = LFWDataset.Dataset(training_set, DataTransformCompose.TransformLFW(isTrain=True, arg_InputSize=arg_inputsize))
    validation_dataset = LFWDataset.Dataset(validation_set, DataTransformCompose.TransformLFW(isTrain=False, arg_InputSize=arg_inputsize))
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=train_batch_size,
        num_workers=6,
        shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=test_batch_size,
        num_workers=6,
        shuffle=False
    )
    return training_dataloader, validation_dataloader, num_classes

# Load LFW Testing pairs
def LoadLFWTest(test_batch_size, path, pairs_path, arg_inputsize):
    dataset = LFWDataset.LFWPairedDataset(path, pairs_path, DataTransformCompose.TransformLFW(isTrain=False, arg_InputSize=arg_inputsize))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, num_workers=8)
    return dataloader, len(dataset)

