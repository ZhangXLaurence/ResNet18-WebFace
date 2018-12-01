import torch
from torchvision import transforms


##################################    MNIST Dataset     ##################################
def TransformMNIST():
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
##################################    End MNIST Dataset     ##################################


##################################    ImageNet Dataset     ##################################
def TransformImageNet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ])
class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  
        self.eig_vec = eig_vec  

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
##################################    End ImageNet Dataset     ##################################

##################################    WebFace Dataset     ##################################
def TransformWebFace(arg_InputSize=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    arg_Transform = transforms.Compose([
        transforms.Resize(arg_InputSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return arg_Transform
##################################    End WebFace Dataset     ##################################


##################################    CIFAR-10 Dataset     ##################################
def TransformCIFAR10(isTrain=True, arg_InputSize=224):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if isTrain:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(arg_InputSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])
        return transform_train
    else:
        transform_test = transforms.Compose([
            transforms.Resize(arg_InputSize),
            transforms.ToTensor(),
            normalize,])
        return transform_test
##################################    End CIFAR-10 Dataset     ##############################

    
##################################    LFW Dataset     ##############################
def TransformLFW(isTrain, arg_InputSize):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if isTrain:
        transform_train = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize(arg_InputSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        return transform_train
    else:
        transform_test = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize(arg_InputSize),
            transforms.ToTensor(),
            normalize])
        return transform_test

##################################    End LFW Dataset     ##############################