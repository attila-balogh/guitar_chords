from datasets.chords_dataset import ChordsDataset
from datasets.data import merged_dataset, classes

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def show_samples(dataset=merged_dataset, classes=classes):
    # Create a dataloader for sample images to show
    show_dataset = ChordsDataset(dataset, image_transformations=transforms.Compose([transforms.Resize((256, 256)),
                                                                                    transforms.ToTensor()]))
    show_dataloader = DataLoader(show_dataset, batch_size=16, shuffle=True)

    for images, labels in show_dataloader:
        plt.figure(figsize=(28,28))
        plt.axis("off")
        plt.imshow(make_grid(images, nrow=4, padding=0).permute(1,2,0))
        for index, label in enumerate(labels):
            if index < 4:
                plt.text(index*256+6, 0*256+18, classes[label.item()],
                         bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
            elif index < 8:
                plt.text((index-4)*256+6, 1*256+18, classes[label.item()],
                         bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
            elif index < 12:
                plt.text((index-8)*256+6, 2*256+18, classes[label.item()],
                         bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
            else:
                plt.text((index-12)*256+6, 3*256+18, classes[label.item()],
                         bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
        break


def plot_accuracy(train, val):
    plt.figure(figsize=(12,8))
    plt.plot(val)
    plt.plot(train)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('No. epochs')
    plt.title('Accuracy through epochs')
    plt.legend(["Validation", "Training"])
    plt.show()


def plot_loss(train, val):
    plt.figure(figsize=(12,8))
    plt.plot(val)
    plt.plot(train)
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('No. epochs', fontsize=18)
    plt.title('Loss through epochs', fontsize=20)
    plt.legend(["Validation", "Training"], fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def plot_lr(lr):
    plt.figure(figsize=(10,6))
    plt.plot(lr)
    plt.ylabel('Learning rate')
    plt.xlabel('Iterations')
    plt.title('Learning rate through iterations')
    plt.show()
