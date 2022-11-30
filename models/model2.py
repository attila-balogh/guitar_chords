import torch
import torch.nn as nn
import torch.nn.functional as F


class Model2(nn.Module):
    def __init__(self, image_size, no_output):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_1 = nn.BatchNorm2d(32)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.20)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_2 = nn.BatchNorm2d(32)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_3 = nn.BatchNorm2d(64)
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.20)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_4 = nn.BatchNorm2d(128)
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.20)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=128 * int(image_size[0] / 32) * int(image_size[1] / 32), out_features=1024)
        self.dropout3 = nn.Dropout(0.30)

        self.fc2 = nn.Linear(in_features=1024, out_features=256)

        self.out = nn.Linear(in_features=256, out_features=no_output)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = self.dropout1(self.maxpool2d_1(F.relu(self.batchnorm_conv_1(self.conv2(t)))))
        t = self.maxpool2d_2(F.relu(self.batchnorm_conv_2(self.conv3(t))))
        t = self.dropout2(self.maxpool2d_3(F.relu(self.batchnorm_conv_3(self.conv4(t)))))
        t = self.dropout3(self.maxpool2d_4(F.relu(self.batchnorm_conv_4(self.conv5(t)))))
        t = self.maxpool2d_5(F.relu(self.conv6(t)))

        t = torch.flatten(t, start_dim=1)
        # t = t.reshape(-1, 64 * int(image_size[0]/32) * int(image_size[1]/32))
        t = self.dropout3(F.relu(self.fc1(t)))
        t = F.relu(self.fc2(t))
        t = F.relu(self.out(t))

        return t
