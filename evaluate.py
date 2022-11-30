import torch

from train import get_device
from train import test_dataset_size
from train import get_num_correct

from datasets.data import classes_temp

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def evaluate_model(network, dataset, device=get_device()):
    predictions = []
    test_correct = 0

    network.eval()

    ds_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    test_ds_labels = []

    for batch in ds_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.int64)

        with torch.no_grad():
            preds = network(images)
            predictions.extend((preds.argmax(dim=1)).tolist())
            test_correct += get_num_correct(preds, labels)

        test_ds_labels.extend(labels.tolist())

    test_acc = 100 * test_correct / test_dataset_size

    print()
    print(f"Test set accuracy:\t{100 * test_correct / len(dataset)}%")
    print(test_correct)
    print(len(dataset))

    # CONFUSION MATRIX on the dataset
    cm = confusion_matrix(test_ds_labels, predictions)

    plt.figure(figsize=(12, 10))
    f = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    f.set_xticklabels(list(classes_temp.keys()), rotation=40, size=16)
    f.set_yticklabels(list(classes_temp.keys()), rotation=40, size=16)
    f.set_xlabel('Predicted', size=24)
    f.set_ylabel('True', size=24)

    f.set_title("Confusion Matrix", size=32)

    return test_acc

