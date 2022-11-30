import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.model2 import Model2

from PIL import Image

import cv2


def get_device():
    """
    Check if GPU is available, and if so, picks the GPU, else picks the CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_network(model, path):
    """
    Loads a trained network to cuda, or if not available, to cpu
        Parameters:
            model (model): a torch neural network
            path (str): a path to a trained .pth file
        Returns:
            network (model): the trained model
    """
    network = model
    network.to(get_device())
    if torch.cuda.is_available():
        network.load_state_dict(torch.load(path))
    else:
        network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # network.eval()

    return network


def open_image(image):
    """
    Returns a correctly transformed (correct size; correct type (torch tensor); normalized) image from an opencv output
        Parameters:
            image (image): an opencv frame
        Returns:
            image (torch tensor): the transformed image (on cuda, if available)
    """
    # mean and std calculated on the training images
    mean = [0.3098142947396422, 0.26889919396194034, 0.25569659948687307]
    std = [0.2119880689850356, 0.19533031751994157, 0.18905260010401728]
    image_size = (256, 256)

    test_transformations = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # transform image
    image = (test_transformations(image))
    image = image.to(get_device())

    return image


def predict(network, image):
    """
    Returns the prediction of an image
        Parameters:
            network (model): the model to use
            image (tensor): the image to predict the label for
        Returns:
            pred (str): the predicted class of the image
    """
    # network.eval()
    # the model works with batches of images
    output = network(image.unsqueeze(0))
    # the predicted label is the one with the highest value
    pred = output.argmax(dim=1).item()

    classes = ['A', 'Am', 'B', 'Bm', 'C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', '']

    return classes[pred]


network = load_network(model=Model2(image_size=(256, 256), no_output=15),
                       path=r"/models/Model2_99.82.pth")

cap = cv2.VideoCapture(1)

# Set width and height resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(1280))
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(960))

# initial prediction is empty
pred = ""

while True:
    # read the webcam images
    ret, frame = cap.read()

    # transform the frame to torch tensor
    img = Image.fromarray(frame)
    img = open_image(img)

    # calculate the confidence
    prob = max(F.softmax((network(img.unsqueeze(0))).squeeze(0), dim=0)*100)
    # do not change the guess, unless the network is pretty sure about it
    if prob >= 95:
        pred = predict(network, img)

    # display the infos to the frame
    cv2.putText(frame, f"{prob:.2f}%", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, pred, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    # quit by hitting 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# kill everything
cap.release()
cv2.destroyAllWindows()
