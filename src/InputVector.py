import random

import torch
import torchvision
from icecream import ic


class InputVector:
    def __init__(self, timeResolution: int = 10):
        # Parameters!
        self.timeResolution = timeResolution

        # Load MNIST dataset from /data/
        self.mnist = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        )

    def getInputVector(self) -> tuple[torch.Tensor, int]:
        # Get a random MNIST image
        image, label = self.mnist[random.randint(0, len(self.mnist) - 1)]

        # Reshape the image into a 784x1 vector
        image = image.reshape(-1, image.shape[0])

        # Convert into a series of time-spiking lines
        spikeLines = torch.zeros((image.shape[0], self.timeResolution))
        for i in range(image.shape[0]):
            # Get the pixel's value
            pixelValue = int(image[i] * self.timeResolution)
            # Clamp the pixel value correctly
            pixelValue = min(pixelValue, self.timeResolution - 1)
            pixelValue = max(pixelValue, 0)
            # Set the corresponding index to a one-hot spike
            spikeLines[i][pixelValue] = 1

        return spikeLines, label
