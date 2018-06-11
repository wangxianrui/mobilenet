import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

import config
import network

device = torch.device('cuda' if config.use_cuda else 'cpu')

# dataset
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

print(model.state_dict().keys())
exit(0)

# load
model.load_state_dict(torch.load(config.model + '.pt'))

# test
model.eval()
correct = 0
total = 0
for i, (input, target) in enumerate(train_loader):
    input, target = input.to(device), target.to(device)
    output = model(input)
    _, predicted = torch.max(output, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
print('the accuracy of train_set is {:.3f}%'.format(correct / total * 100))

correct = 0
total = 0
for i, (input, target) in enumerate(eval_loader):
    input, target = input.to(device), target.to(device)
    output = model(input)
    _, predicted = torch.max(output, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
print('the accuracy of teset_set is {:.3f}%'.format(correct / total * 100))
