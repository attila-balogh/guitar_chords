import torch
from PIL import Image


class MergeDataset(torch.utils.data.Dataset):
    """
    Returns merged dataset with image - label pairs
          Parameters:
              paths (list): the list of image paths
              labels (list): the list of labels
          Returns:
              image, label (tuple): the current image-label tuple
    """

    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

        if len(self.paths) != len(self.labels):
            print("WARNING: mismatch!!!")

    def __getitem__(self, idx):
        # dataset[idx]
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        return image, label

    def __len__(self):
        # len(dataset)
        return len(self.paths)
