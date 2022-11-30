import glob

import re

from matplotlib import pyplot as plt

from torch.utils.data import random_split
import torchvision.transforms as transforms

from PIL import ImageStat

from datasets.merge_datasets import MergeDataset
from datasets.chords_dataset import ChordsDataset


images_paths = glob.glob('../images/*')
images_paths = sorted(images_paths)

pattern = r"\w+$"

class_contents = dict()

for ind, folder in enumerate(images_paths):
    no_images = len(glob.glob(f"{folder}/*"))
    chord_name = re.findall(pattern, folder)[0]
    print(f"{chord_name}\t{no_images}")

    class_contents[chord_name] = no_images

dataset_size = 0
for key, value in class_contents.items():
    dataset_size += value


train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

print(f"No. training images\t: {train_size:5,}")
print(f"No. validation images\t: {val_size:5,}")
print(f"No. test images\t\t: {test_size:5,}")


classes = dict()
for ind, chord in enumerate(class_contents.keys()):
    classes[chord] = ind


merged_dataset = []

for folder in images_paths:
    # create a list with the appropriate labels with the same size as the number of the images in the current folder
    chord_name = re.findall(pattern, folder)[0]
    labels_for_folder = (list([classes[chord_name]]) * len(glob.glob(f"{folder}/*")))

    # pair the image with the corresponding label
    merged_dataset.extend(MergeDataset(glob.glob(f"{folder}/*"), labels_for_folder))


# reverse the dictionary from the form 'Am' -> 1 to the form of 1 -> 'Am', so we can use later at predictions
classes_temp = classes
classes = dict()
for i in range(len(classes_temp)):
    classes[i] = list(classes_temp.keys())[i]


# random split the dataset to training, validation and test datasets
train_dataset, val_dataset, test_dataset = random_split(merged_dataset, [train_size, val_size, test_size])


# Calculate the mean and std of the images for normalization
R_mean_sum = 0
R_std_sum = 0

G_mean_sum = 0
G_std_sum = 0

B_mean_sum = 0
B_std_sum = 0

for image, label in train_dataset:
    R_mean_sum += ImageStat.Stat(image).mean[0]
    R_std_sum += ImageStat.Stat(image).stddev[0]

    G_mean_sum += ImageStat.Stat(image).mean[1]
    G_std_sum += ImageStat.Stat(image).stddev[1]

    B_mean_sum += ImageStat.Stat(image).mean[2]
    B_std_sum += ImageStat.Stat(image).stddev[2]

R_mean = R_mean_sum / len(train_dataset) / 255
R_std = R_std_sum / len(train_dataset) / 255

G_mean = G_mean_sum / len(train_dataset) / 255
G_std = G_std_sum / len(train_dataset) / 255

B_mean = B_mean_sum / len(train_dataset) / 255
B_std = B_std_sum / len(train_dataset) / 255

mean = [R_mean, G_mean, B_mean]
std = [R_std, G_std, B_std]

print(f"mean {mean}")
print(f"std {std}")


# Set image size for transformations
image_size = (256, 256)


# Set training|validation|test transformations
training_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.3, 0.1, 0.1)
    ], p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3,3))
    ], p=0.1),
    transforms.RandomRotation(degrees=20),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.1),
    transforms.RandomCrop(size=image_size, padding=20),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Apply transformations
train_ds = ChordsDataset(train_dataset, image_transformations=training_transformations)
val_ds = ChordsDataset(val_dataset, image_transformations=validation_transformations)
test_ds = ChordsDataset(test_dataset, image_transformations=test_transformations)
