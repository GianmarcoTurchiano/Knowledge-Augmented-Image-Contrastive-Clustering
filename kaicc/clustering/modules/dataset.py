import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    # For small images such as CIFAR-10, one might leave out GaussianBlur.
    transforms.ToTensor(),
])


class DoubleTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class ArtworkDataset(Dataset):
    def __init__(
        self,
        image_directory,
        labels_file_path,
        transform=None
    ):
        self.image_directory = image_directory
        self.df = pd.read_csv(labels_file_path)
        
        self.transform = transform

        if self.transform == None:
            self.transform = transforms.Compose([
                #transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_filename = row['Artwork']
        style = row['Style_idx']
        genre = row['Genre_idx']
        image_path = f'{self.image_directory}/{image_filename}'
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, style, genre


class ArtworkVsArtworkDataset(Dataset):
    def __init__(
        self,
        image_directory,
        labels_file_path,
        transform
    ):
        self.dataset = ArtworkDataset(
            image_directory,
            labels_file_path,
            DoubleTransform(transform)
        )
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (image_a, image_b), style, genre = self.dataset[index]
        return image_a, image_b, style, genre


class ArtworkVsCaptionDataset(Dataset):
    pass


class AugmentedArtworkVsArtworkDataset(ArtworkVsArtworkDataset):
    pass
