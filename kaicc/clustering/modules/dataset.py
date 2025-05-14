from math import ceil
import tarfile

from transformers import CLIPProcessor
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

def get_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size)
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
    ])


class DoubleTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class ArtworkDataset(Dataset):
    def __init__(
        self,
        image_archive_path,
        image_directory_path,
        labels_file_path,
        transform=None
    ):
        self.image_directory = image_directory_path
        self.df = pd.read_csv(labels_file_path)
        
        self.transform = transform

        if self.transform == None:
            self.transform = transforms.ToTensor()

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
        base_model_name: str,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        transform
    ):
        self.processor = CLIPProcessor.from_pretrained(base_model_name, use_fast=False)

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
        base_model_name: str,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        transform
    ):
        super().__init__(
            base_model_name,
            image_archive_path,
            image_directory_path,
            labels_file_path,
            DoubleTransform(transform)
        )

    def __getitem__(self, index):
        (image_a, image_b), style, genre = self.dataset[index]

        inputs_a = self.processor(
            images=image_a,
            return_tensors="pt",
            padding=True,
            do_rescale=True
        )

        inputs_b = self.processor(
            images=image_b,
            return_tensors="pt",
            padding=True,
            do_rescale=True
        )

        inputs_a["pixel_values"] = inputs_a["pixel_values"].squeeze(0)
        inputs_b["pixel_values"] = inputs_b["pixel_values"].squeeze(0)

        return inputs_a, inputs_b, style, genre


class ArtworkVsCaptionDataset(_ArtworkVsDataset):
    def __init__(
        self,
        base_model_name: str,
        image_archive_path: str,
        image_directory_path: str,
        labels_file_path: str,
        captions_file_path: str,
        transform=None,
    ):
        super().__init__(
            base_model_name,
            image_archive_path,
            image_directory_path,
            labels_file_path,
            transform
        )

        self.captions_df = pd.read_csv(captions_file_path)

    def __getitem__(self, index):
        image, style, genre = self.dataset[index]

        image_inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            do_rescale=True
        )

        image_inputs["pixel_values"] = image_inputs["pixel_values"].squeeze(0)

        caption = self.captions_df.loc[index]['Caption']

        caption_inputs = self.processor(
            text=caption,
            return_tensors="pt",
            padding=True,
        )

        caption_inputs["input_ids"] = caption_inputs["input_ids"].squeeze(0)

        return image_inputs, caption_inputs, style, genre
