import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class ASD_Dataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        label_map = {
            "Normal": 0,
            "Autism": 1
        }

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)

            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    # Only add if file exists and is a PNG
                    if os.path.exists(img_path) and img_path.endswith('.png'):
                        self.image_paths.append(img_path)
                        self.labels.append(label_map[label_name])

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            # If image loading fails, return a black image as fallback
            print(f"Warning: Failed to load image {self.image_paths[idx]}: {e}")
            img = torch.zeros((3, 128, 128))

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def get_dataloader(train=True):
    dataset = ASD_Dataset("data/spectrograms")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if train:
        return DataLoader(train_dataset, batch_size=8, shuffle=True)
    else:
        return DataLoader(test_dataset, batch_size=8, shuffle=False)