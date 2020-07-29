import torch 
import torchvision 
from skimage import transform as tranf
from skimage import io
# from PIL import Image
import numpy as np



class GestureData(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        if train == True: self.train = True
        else: self.train = False

        if self.train == True:
            f = open(root+'/train.txt', 'r')
            print('READ TRAIN LIST FILE SUCCESS.')
        else:
            f = open(root+'/test.txt', 'r')
            print('READ TEST LIST FILE SUCCESS.')
        
        self.records = f.readlines()

    def __getitem__(self, index):
        record = self.records[index]
        img_path = record.split(',')[0]
        label = int(record.split(',')[1].strip())
        image = io.imread(img_path) / 255
        if not image.shape == (100, 100, 3):
            image = tranf.resize(image, (100,100))

        image = image.transpose(2, 1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        
        return image, label

    def __len__(self):
        return len(self.records) 

### 2

import torch.nn as nn
import torch.nn.functional as F 

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 3 input channel, 6 output channels, 3x3 conv kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*10*10, 1024)        
        self.fc2 = nn.Linear(1024, 84)        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = MyNet()
# input = torch.randn(1, 3, 100, 100)
# out = net(input)
# print(out)

### 3

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### 4

trainset = GestureData('./datalist/',train=True)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=8,shuffle=True,num_workers=4)

MODEL_SAVE_PATH = './model/'

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/50))
            running_loss = 0.0
    torch.save(net.state_dict(), MODEL_SAVE_PATH+'gesture_%02d.pth' %(epoch+1))

print('Finished Training')




