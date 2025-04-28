import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
    

normalize = transforms.Normalize(
    mean=[0.49139968, 0.48215827, 0.44653124],
    std=[0.24703233, 0.24348505, 0.26158768]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    # For small images such as CIFAR-10, one might leave out GaussianBlur.
    transforms.ToTensor(),
    normalize
])

class ContrastiveDataset(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                    download=True,
                                                    transform=TwoCropsTransform(train_transform))
    def __getitem__(self, index):
        (xi, xj), label = self.dataset[index]
        return xi, xj, label

    def __len__(self):
        return len(self.dataset)
