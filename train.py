import torch

import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torch.utils.data import DataLoader

import time

import evaluate
from models.model2 import Model2

from datasets.data import train_ds, val_ds, test_ds
from datasets.data import image_size
from datasets.data import classes


# Print out used versions
print()
print()
print("Used versions:")
print(f"Torch:\t\t{torch.__version__}")
print(f"Torchvision:\t{torchvision.__version__}")
print()


def get_device():
    """
    Check if GPU is available, and if so, picks the GPU, else picks the CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# Set the device to cuda is available
device = get_device()
print(f"Used device is {device}.")


def get_num_correct(preds, labels):
    """
    Returns the number of the correctly predicted images
        Parameters:
            preds (tensor): the predicted labels
            labels (tensor): the true labels (targets)
        Returns:
            num_correct (int): the number of correctly predicted images
    """
    num_correct = preds.argmax(dim=1).eq(labels).sum().item()
    return num_correct


def accuracy(preds, labels):
    """
    Returns the accuracy of the predictions
        Parameters:
            preds (tensor): the predicted labels
            labels (tensor): the true labels (targets)
        Returns:
            acc (float): the accuracy of the predictions (correctly predicted labels / all predictions)
    """
    acc = get_num_correct(preds, labels) / len(labels)
    return acc


def predict(network, image):
    """
    Returns the prediction of an image
        Parameters:
            network (model): the model to use
            image (tensor): the image to predict the label for
        Returns:
            pred (int): the predicted label of the image
    """
    network = network.to(device)
    image = image.to(device)
    output = network(image.unsqueeze(0))
    pred = output.argmax(dim=1).item()
    return pred


def save_model(network, path):
    """
    Saves the trained model's state
        Parameters:
            network (model): the model to save
            path (str): the location + filename we want to save
    """
    torch.save(network.state_dict(), path)


def passed_time(start, end):
    """
    Returns with formatted time (hrs, mins, secs) in f string form
          Parameters:
              start (time): the startpoint of the time interval
              end (time): the endpoint of the time interval
            Returns:
              str (f string): the formatted (hrs, mins, secs) time interval
    """
    passed = end - start

    passed_hrs = passed // 3600
    passed -= passed_hrs * 3600

    passed_mins = passed // 60
    passed -= passed_mins * 60

    passed_secs = int(passed)

    return f"{passed_hrs:2.0f}h {passed_mins:2.0f}m {passed_secs:2.0f}s"


train_dataset_size = len(train_ds)
val_dataset_size = len(val_ds)
test_dataset_size = len(test_ds)


# HYPERPARAMETERS

no_output = len(classes.keys())

network = Model2(image_size=image_size, no_output=no_output)

batch_size = 32
loss_fn = F.cross_entropy
learning_rate = 0.0001
num_epoch = 100

network.to(device)

# Printing out network properties
print()
print(network)
print()
pytorch_total_params = sum(p.numel() for p in network.parameters())
print(f"Total parameters: \t\t\t{pytorch_total_params:,}")
pytorch_total_learnable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"Total trainable parameters: \t{pytorch_total_learnable_params:,}")
print()
print(f"Training dataset size: \t\t{train_dataset_size:,}")
print(f"Validation dataset size: \t{val_dataset_size:,}")
print(f"Test dataset size: \t\t\t{test_dataset_size:,}")
print()
print(f"Input image size: \t {image_size[0]} × {image_size[1]}")
print()
print(f"Batch size    \t {batch_size}")
print(f"Learning rate \t {learning_rate}")
print(f"Loss function \t {loss_fn}")
print(f"No. epochs    \t {num_epoch}")

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)
test_loader = DataLoader(test_ds, batch_size * 2)

optimizer = optim.Adam(network.parameters(), lr=learning_rate)

lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=num_epoch,
                                             steps_per_epoch=len(train_loader))

# BEFORE TRAINING METRICS

total_loss = 0
total_correct = 0

total_val_loss = 0
total_val_correct = 0

network.eval()

for batch in train_loader:
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)

    preds = network(images)
    loss = loss_fn(preds, labels)

    total_loss += loss.item() * len(batch[0])
    total_correct += get_num_correct(preds, labels)

for batch in val_loader:
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)
    labels = labels.to(torch.int64)

    preds = network(images)
    loss = loss_fn(preds, labels)

    total_val_loss += loss.item() * len(batch[0])
    total_val_correct += get_num_correct(preds, labels)

print()
print(f"BEFORE TRAINING: train accuracy {100 * total_correct / train_dataset_size:3.2f}%, "
      f"loss = {total_loss / train_dataset_size:4.4f}")
print(f"BEFORE TRAINING: val accuracy   {100 * total_val_correct / val_dataset_size:3.2f}%, "
      f"loss = {total_val_loss / val_dataset_size:4.4f}")
print()
print()

before_train_acc = (100 * total_correct / train_dataset_size)
before_val_acc = (100 * total_val_correct / val_dataset_size)

before_train_loss = (total_loss / train_dataset_size)
before_val_loss = (total_val_loss / val_dataset_size)

# TRAINING

history_train_loss = []
history_train_acc = []

history_val_loss = []
history_val_acc = []

lr_history = []

since = time.time()

for epoch in range(num_epoch):

    since_epoch = time.time()

    # Training phase

    total_loss = 0
    total_correct = 0

    network.train()

    for batch in train_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.int64)

        preds = network(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        lr_history.append(lr_scheduler.get_last_lr())

        total_loss += loss.item() * len(batch[0])
        total_correct += get_num_correct(preds, labels)

    # Validation phase

    total_val_loss = 0
    total_val_correct = 0

    network.eval()

    for batch in val_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.int64)

        with torch.no_grad():
            preds = network(images)
            loss = loss_fn(preds, labels)

        total_val_loss += loss.item() * len(batch[0])
        total_val_correct += get_num_correct(preds, labels)

    # Print out metrics

    training_percentage = 100 * total_correct / train_dataset_size
    validation_percentage = 100 * total_val_correct / val_dataset_size

    history_train_acc.append(training_percentage)
    history_val_acc.append(validation_percentage)

    history_train_loss.append(total_loss / train_dataset_size)
    history_val_loss.append(total_val_loss / val_dataset_size)

    end_epoch = time.time()

    training_time_epoch = passed_time(since_epoch, end_epoch)

    print(f"EPOCH: {epoch + 1}\tTime: {training_time_epoch}")
    print(f"Training"
          f"\ttrain accuracy {training_percentage:3.2f}%"
          f"\ttrain loss: {total_loss / train_dataset_size:2.4f}")
    print(f"Validation"
          f"\tval accuracy   {validation_percentage:3.2f}%"
          f"\tval loss:   {total_val_loss / val_dataset_size:2.4f}")
    print()

print()
print()
end = time.time()

training_time = passed_time(since, end)

print(f"TIME of training (with validation phases) {training_time}.")


test_acc = evaluate.evaluate_model(network, device)

save_path = f"../models/Model2_{test_acc:.2f}.pth"
save_model(network, save_path)
