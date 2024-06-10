import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transform
transform = transforms.Compose([
    transforms.ToTensor(),  
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  
])

test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=False)
dataloader_test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
classes = test_dataset.classes

