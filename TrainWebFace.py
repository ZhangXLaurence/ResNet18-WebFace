import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms

from bases.DataLoader import DataLoad
from bases.DataLoader import TransformWebFace
from bases.Models import resnet50
from bases.Losses import MarginInnerProduct
from bases.Losses import GradReform
from Tools import ModelSaver

class TrainingModel(nn.Module):
    def __init__(self, inference_model, inner_product):
        super(TrainingModel, self).__init__()
        self.inference_model = inference_model
        # self.gradreform = GradReform.IdentityMappingReformGrad()
        self.inner_product = inner_product
    def forward(self, x, label):
        features = self.inference_model(x)
        # features = self.gradreform(features)
        logits = self.inner_product(features, label)
        # logits = self.inner_product(features)
        return features, logits
    def SaveInferenceModel():
        # TO BE DOWN
        return 0

# def Validate(test_loder, model):
#     correct = 0
#     total = 0
#     for i, (data, target) in enumerate(test_loder):
#         if torch.cuda.is_available():
#             data = data.cuda()
#             target = target.cuda()

#         feats, logits = model(data, target)
#         _, predicted = torch.max(logits.data, 1)
#         total += target.size(0)
#         correct += (predicted == target.data).sum()
#     print('Validating Accuracy on the {} test images:{}/{} ({:.2f}%) \n' .format(
#         total, correct, total, (100. * float(correct)) / float(total))
#         )


def Train(train_loader, model, criterion, optimizer, epoch, info_interval):
    for i, (data, target) in enumerate(train_loader):
        # curr_step = start_iter + i
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        feats, logits = model(data, target)
        loss = criterion[0](logits[1], target)

        _, predicted = torch.max(logits[0].data, 1)
        # loss = criterion[0](logits, target)

        # _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()

        if (i + 1) % info_interval == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader) , loss.item(), accuracy))


def Processing(NumEpoch, LrScheduler, Optimizer, train_loader, model, criterion, info_interval, save_path):
    for epoch in range(NumEpoch):
        LrScheduler.step()
        print('Current Learning Rate: {}'.format(Optimizer.param_groups[0]['lr']))
        Train(train_loader, model, criterion, [Optimizer], epoch, info_interval)
        SavePath = save_path + str(epoch + 1) + '.model'
        ModelSaver.SaveModel(model, SavePath, epoch, 10)
        # Validate(test_loder, model)


def main():
    ################################################################################################
    # This process, set up the whole models and parameters
    # Get Hyper parameters and sets

    # General arg
    # arg_DeviceIds = [0,1,2,3,4,5,6,7]
    arg_DeviceIds = [0,1,2,3]
    arg_NumEpoch = 50
    arg_InfoInterval = 100
    arg_SavePath = './checkpoints/softmax_MNIST_'
    arg_SaveEpochInterbal = 10

    # Data arg
    arg_TrainDataPath = '/home/xzhang/data/face/WebFace/CASIA-WebFace-112X96/'
    arg_TrainBatchSize = 256
    arg_InputSize = 224

    arg_FeatureDim = 2048
    
    # Learning rate arg
    arg_BaseLr = 0.1
    arg_Momentum = 0.2
    arg_WeightDecay = 0.00005

    # Learning rate scheduler
    arg_LrEpochStep = 20
    arg_Gamma = 0.5

    # Dataset Loading
    arg_Transform = TransformWebFace()
    TrainLoader, arg_ClassNum = DataLoad.LoadFaceImgFoldData(arg_TrainBatchSize, arg_TrainDataPath, transform=arg_Transform)

    # Inference Model Constructing
    Inference = resnet50(pretrained=False, num_classes=arg_ClassNum)
    # Inner Product
    # InnerProduct = MarginInnerProduct.CosFaceInnerProduct(arg_FeatureDim, arg_ClassNum, scale=20.0, margin=0.3)
    InnerProduct = MarginInnerProduct.ArcFaceInnerProduct(arg_FeatureDim, arg_ClassNum, scale=30.0, margin=0.005)
    # InnerProduct = GradReform.MyLinear(arg_FeatureDim, arg_ClassNum)
    # InnerProduct = MarginInnerProduct.NormalizedInnerProductWithScale(arg_FeatureDim, arg_ClassNum)
    
    # Training Model
    Model = torch.nn.DataParallel(TrainingModel(Inference, InnerProduct), arg_DeviceIds)

    # Losses and optimizers Defining
    # Softmax CrossEntropy
    SoftmaxLoss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        SoftmaxLoss = SoftmaxLoss.cuda()
        Model = Model.cuda()
    criterion = [SoftmaxLoss]
    # Optimzer
    Optimizer = torch.optim.SGD(Model.parameters(), lr=arg_BaseLr, momentum=arg_Momentum, weight_decay=arg_WeightDecay)
    
    # Learning rate Schedule
    LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, arg_LrEpochStep, gamma=arg_Gamma)


    ################################################################################################

    # Resume from a checkpoint/pertrain

    # Training models
    # Testing models
    Processing(arg_NumEpoch, LrScheduler, Optimizer, TrainLoader, Model, criterion, arg_InfoInterval, arg_SavePath)



if __name__ == '__main__':
    main()
