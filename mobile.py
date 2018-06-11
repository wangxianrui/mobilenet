import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.models
import torchvision.transforms as transforms

import network
import config

# device
device = torch.device('cuda' if config.use_cuda else 'cpu')

# load data
root = config.root
train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])]),
                                         )
eval_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])]),
                                        )
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True,
                                           num_workers=config.workers, pin_memory=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=config.batch_size, shuffle=True,
                                          num_workers=config.workers, pin_memory=True)

# create model
if (config.model == 'mobilenet'):
    model = network.Net()
else:
    model = torchvision.models.__dict__[config.model]()
model = torch.nn.DataParallel(model)
model = model.to(device)

# criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                            weight_decay=config.weight_decay)

file = open('result', 'w')

# train the network
for epoch in range(config.epochs):
    # adjust learning_rate
    lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # train
    running_loss = 0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % config.frequency == config.frequency - 1:
            print('train {}/{}: {}/{}: loss={:.3f}'.format(epoch, config.epochs, i, train_set.__len__(),
                                                           running_loss / 100))
            running_loss = 0

    # eval
    model.eval()

    correct = 0
    total = 0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('the accuracy of {}/{} epoch is {:.3f}%'.format(epoch, config.epochs, correct / total * 100))
    file.writelines('the accuracy of {}/{} epoch is {:.3f}%'.format(epoch, config.epochs, correct / total * 100))

    correct = 0
    total = 0
    for i, (input, target) in enumerate(eval_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('the accuracy of {}/{} epoch is {:.3f}%'.format(epoch, config.epochs, correct / total * 100))
    file.writelines('the accuracy of {}/{} epoch is {:.3f}%'.format(epoch, config.epochs, correct / total * 100))

    # save
    torch.save(model.state_dict(), config.model + '.pt')

file.close()
