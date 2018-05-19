# -*- coding: utf-8 -*-

if __name__=='__main__':

    import torch
    import torchvision
    import torchvision.transforms as transforms


    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #deviceCPU = torch.device("cpu")

    # Assume that we are on a CUDA machine, then this should print a CUDA device:

    #print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
            self.conv7 = nn.Conv2d(192, 192, 3, stride=1, padding=0)
            self.conv8 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
            self.conv9 = nn.Conv2d(192, 10, 1, stride=1, padding=0)

            # Dropout2d(p)
            self.drop1 = nn.Dropout2d(0.2)
            self.drop2 = nn.Dropout2d(0.5)
            self.drop3 = nn.Dropout2d(0.5)



        def forward(self, x):
            #print(x.size())

            x = self.drop1(x)
            x = self.conv1(x)

            x = self.conv2(F.relu(x))

            x = self.conv3(F.relu(x))

            x = self.drop2(x)
            x = self.conv4(F.relu(x))

            x = self.conv5(F.relu(x))

            x = self.conv6(F.relu(x))

            x = self.drop3(x)
            x = self.conv7(F.relu(x))

            x = self.conv8(F.relu(x))

            x = self.conv9(F.relu(x))

            x = F.relu(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(-1, 10)

            return x


    net = Net()
    print(net)
    #net.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')


    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            #images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            #predicted = predicted.to(deviceCPU)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            #images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            #predicted = predicted.to(deviceCPU)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


