import torch 
import torchvision 
import torchvision.transforms as transforms
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
            # print(image.shape, img_path)
            image = tranf.resize(image, (100,100))
            # print('CHANGED')

        image = image.transpose(2, 1, 0)
        image = torch.tensor(image, dtype=torch.float32)


        # sample = {'image':image, 'landmarks':label}

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        return image, label

    def __len__(self):
        return len(self.records) 

### 2

import torch.nn as nn
import torch.nn.functional as F 

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 3 input channel, 6 output channels, 5x5 conv kernel, 1x1 pad
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
        # print(self.num_flat_features(x)/32)
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
net.load_state_dict(torch.load('./model/gesture_32.pth'))

testset = GestureData('./datalist/',train=False)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=2)

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# print(labels)

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print(predicted)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(total)
print('Accuracy of the network on xx test images: %d %%' %(100*correct/total))