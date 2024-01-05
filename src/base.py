import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
import random
import numpy as np


class AddTrigger:
    def __init__(self, pattern):
        self.pattern = pattern

    def add_trigger(self, img):
        normalized_pattern = (self.pattern.float() / 255.0) * img.max()  # normalize the pattern
        poisoned_img = (img.float() + normalized_pattern).clamp(0, 255).type(
            torch.uint8)  # add the pattern to the image

        # Check if the original image is grayscale
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1)

        # Convert to grayscale only if originally a grayscale image
        # grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
        if is_grayscale:
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=img.device)
            output_img = (poisoned_img.float() * gray_weights[:, None, None]).sum(0).clamp(0, 255).to(torch.uint8)
        else:
            output_img = poisoned_img

        return output_img


class AddCIFAR10Trigger(AddTrigger):
    """
    Class for adding a backdoor trigger to a CIFAR10 image.

    Attributes:
        pattern: a backdoor trigger pattern, torch.Tensor of shape (C, H, W) -> (1, 32, 32)
        alpha: transparency of the trigger pattern, float32 [0, 1]

    Methods:
        __init__: initialize the backdoor trigger pattern and transparency
        __call__: add the backdoor trigger to the image

    """
    def __init__(self, pattern, alpha=1):
        assert isinstance(pattern, Image.Image) or pattern is None, 'pattern should be a PIL image.'
        self.alpha = alpha

        if pattern is None:
            pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255  # default pattern, 3x3 white square at the right corner
        else:
            pattern = F.pil_to_tensor(pattern)
            if pattern.dim() == 2:
                pattern = pattern.unsqueeze(0)

        super().__init__(pattern)

    def __call__(self, img):
        """
        Add the backdoor trigger to the image.
            Arguments:
                img: PIL image
            Returns:
                PIL image
        """
        input_image = F.pil_to_tensor(img)
        output_image = self.add_trigger(input_image)
        return Image.fromarray(output_image.permute(1, 2, 0).numpy())


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset: CIFAR10,
                 y_target,
                 poisoned_rate,
                 poisoning_strategy):
        """
        Args:
            y_target
            poisoned_rate - the percantage of images we wish to transform
            poisoning_strategy - an instace of a class which can be used for transforming images (its call method must take a PIL Image type and return one)
            y_target - the class we are targeting in our backdoor attack

        """
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            download=True)
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
        self.y_target = y_target
        self.poisoning_strategy = poisoning_strategy
        self.poisoned_rate = poisoned_rate

        self.total_num = len(benign_dataset)
        self.poisoned_num = int(self.total_num * self.poisoned_rate)
        assert self.poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(self.total_num))
        random.shuffle(tmp_list)

        # make a set of poisoned indices
        self.poisoned_indices = frozenset(tmp_list[:self.poisoned_num])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_indices:
            img = self.poisoning_strategy(img)
            target = self.y_target

        return img, target


class AddMNISTTrigger(AddTrigger):
    """
    Class for adding a backdoor trigger to a MNIST image.

    Attributes:
        pattern: a backdoor trigger pattern, torch.Tensor of shape (C, H, W) -> (1, 28, 28)
        alpha: transparency of the trigger pattern, float32 [0, 1]

    Methods:
        __init__: initialize the backdoor trigger pattern and transparency
        __call__: add the backdoor trigger to the image
    """

    def __init__(self, pattern, alpha=1):
        assert isinstance(pattern, Image.Image) or pattern is None, 'pattern should be a PIL image.'
        self.alpha = alpha

        if pattern is None:
            pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255  # default pattern, 3x3 white square at the right corner
        else:
            pattern = F.pil_to_tensor(pattern)
            if pattern.dim() == 2:
                pattern = pattern.unsqueeze(0)

        super().__init__(pattern)

    def __call__(self, img):
        """
        Add the backdoor trigger to the image.
            Arguments:
                img: PIL image
            Returns:
                PIL image
        """
        input_image = F.pil_to_tensor(img)
        output_image = self.add_trigger(input_image)
        return Image.fromarray(output_image.squeeze(0).numpy()).convert("L")


class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset: MNIST,
                 y_target,
                 poisoned_rate,
                 poisoning_strategy):
        """
        Args:
            y_target
            poisoned_rate - the percantage of images we wish to transform
            poisoning_strategy - an instace of a class which can be used for transforming images (its call method must take a PIL Image type and return one)
            y_target - the class we are targeting in our backdoor attack

        """
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            download=True)
        self.classes = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ]
        self.y_target = y_target
        self.poisoning_strategy = poisoning_strategy
        self.poisoned_rate = poisoned_rate
        self.total_num = len(benign_dataset)
        self.poisoned_num = int(self.total_num * self.poisoned_rate)
        assert self.poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(self.total_num))
        random.shuffle(tmp_list)

        # make a set of poisoned indices
        self.poisoned_indices = frozenset(tmp_list[:self.poisoned_num])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_indices:
            img = self.poisoning_strategy(img)
            target = self.y_target

        return img, target


def load_CIFAR10_data(benign_root, batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def load_MNIST_data(benign_root, batch_size, transform):
    trainset = torchvision.datasets.MNIST(root=benign_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=benign_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, poisoning_strategy):
    class_name = type(benign_dataset)
    if class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, poisoning_strategy)
    elif class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, poisoning_strategy)