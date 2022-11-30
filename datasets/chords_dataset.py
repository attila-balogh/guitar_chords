import torch


class ChordsDataset(torch.utils.data.Dataset):
    """
    Opens dataset and returns with transformed dataset, if transformations are given
          Parameters:
              dataset (dataset): dataset of images and corresponding labels
              image_transformations (transforms): the transformations we want to apply on the images
              label_transformations (transforms): the transformations we want to apply on the labels
          Returns:
              image, label (tuple): the current modified image-label tuple
    """

    def __init__(self, dataset, image_transformations=None, label_transformations=None):
        self.dataset = dataset
        self.image_transformations = image_transformations
        self.label_transformations = label_transformations

    def __getitem__(self, idx):
        # dataset[idx]
        image, label = self.dataset[idx]
        if self.image_transformations:
            image = self.image_transformations(image)
        if self.label_transformations:
            label = self.label_transformations(label)

        return image, label

    def __len__(self):
        # len(dataset)
        return len(self.dataset)
