import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
  
def train(model, optimizer, criterion, trainloader, device):
    # 모델을 학습 모드로 전환
    model.train()
 
    # # 데이터를 반복하면서 학습
    # for inputs, labels in trainloader:
    #     # 입력 데이터와 라벨을 디바이스에 할당
    #     inputs, labels = inputs.to(device), labels.to(device)

    #     # 경사를 초기화
    #     optimizer.zero_grad()

    #     # 순전파
    #     outputs = model(inputs)

    #     # 손실 계산
    #     loss = criterion(outputs, labels)

    #     # 역전파 및 가중치 업데이트
    #     loss.backward()
    #     optimizer.step()

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        i = 0 
        for data in tqdm(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients 경사 초기화 
            optimizer.zero_grad()

            # forward + backward + optimize 순전파 
            outputs = model(inputs)

            #손실 계산
            loss = criterion(outputs, labels)
            # 역전파 및 가중치 업데이트 
            loss.backward()
            optimizer.step()

            # print statistics
            i += 1
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 12500:.3f}')



if __name__ == '__main__':

    print(torch.__version__)
    print(torch.backends.mps.is_built())
    print(torch.backends.mps.is_available())
   
    device = torch.device("mps")
    # device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 500

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # get some random training images
    dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    images, labels = next(dataiter) # python 3.0 이후 수정됨

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


    net = Net()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #학습 시작 
    train(net, optimizer=optimizer, criterion=criterion, trainloader=trainloader, device=device)
    print('Finished Training')