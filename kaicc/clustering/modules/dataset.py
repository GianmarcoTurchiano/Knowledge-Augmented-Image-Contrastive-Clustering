from math import ceil
import tarfile

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

def get_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.PILToTensor()
    ])


def get_augmented_transform(
    size=224,
    scale=(0.2, 1.0),
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1,
    p_color_jitter=0.8,
    p_gray_scale=0.2,
    p_gaussian_blur=0.5,
    sigma=(0.1, 2.0),
):
    kernel_size = ceil((10.0 * float(size)) / 100.0)

    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomApply([transforms.ColorJitter(brightness, contrast, saturation, hue)], p=p_color_jitter),
        transforms.RandomGrayscale(p=p_gray_scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=p_gaussian_blur),
        transforms.PILToTensor()
    ])


class _DoubleTransform:
    def __init__(self, transform_1, transform_2):
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x):
        return self.transform_1(x), self.transform_2(x)


class ArtworkDataset(Dataset):
    def __init__(
        self,
        image_archive_path,
        image_directory_path,
        labels_file_path,
        transform
    ):
        self.image_directory = image_directory_path
        self.df = pd.read_csv(labels_file_path)
        
        self.transform = transform

        with tarfile.open(image_archive_path, "r:gz") as tar:
            tar.extractall(image_directory_path)

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


class _ArtworkVsDataset(Dataset):
    def __init__(
        self,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        transform
    ):
        self.dataset = ArtworkDataset(
            image_archive_path,
            image_directory_path,
            labels_file_path,
            transform
        )
    
    def __len__(self):
        return len(self.dataset)    


class ArtworkVsArtworkDataset(_ArtworkVsDataset):
    def __init__(
        self,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        transform_1,
        transform_2
    ):
        super().__init__(
            image_archive_path,
            image_directory_path,
            labels_file_path,
            _DoubleTransform(transform_1, transform_2)
        )

    def __getitem__(self, index):
        (image_a, image_b), style, genre = self.dataset[index]

        return image_a, image_b, style, genre


class ArtworkVsCaptionDataset(_ArtworkVsDataset):
    def __init__(
        self,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        captions_file_path: str,
        transform
    ):
        super().__init__(
            image_archive_path,
            image_directory_path,
            labels_file_path,
            transform
        )

        self.captions_df = pd.read_csv(captions_file_path)

    def __getitem__(self, index):
        image, style, genre = self.dataset[index]
        caption = self.captions_df.loc[index]['Caption']

        return image, caption, style, genre
