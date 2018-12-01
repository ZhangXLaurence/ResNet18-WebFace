import torch
import torch.nn as nn
import torch.nn.functional as F



class SmallNet(nn.Module):
    def __init__(self, feature_dim):
        super(SmallNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, feature_dim)
        # self.ip2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.preluip1(self.ip1(x))
        # ip2 = self.ip2(ip1)
        return ip1


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.ip1 = nn.Linear(50,2)
        self.ip2 = nn.Linear(2,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        ip1 = self.ip1(x)
        ip2 = self.ip2(ip1)
        return ip1, ip2
    

# Define a Dense MLP network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(32, 32)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(32, 32)
        self.prelu3 = nn.PReLU()
        self.fc4 = nn.Linear(32, 32)
        self.prelu4 = nn.PReLU()
        self.ip1 = nn.Linear(32,64)
        self.ip2 = nn.Linear(64,52)

    def forward(self, x):
        x1 = self.prelu1(self.fc1(x))
        x2 = self.prelu2(self.fc2(x1))
        x3 = self.prelu3(self.fc3(x2 + x1))
        x4 = self.prelu4(self.fc4(x3 + x2 + x1))
        ip1 = self.ip1(x4)    # feature
        ip2 = self.ip2(ip1)   # logit
        return ip1, ip2