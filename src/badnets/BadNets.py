import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
import random
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class AddTrigger:
    def __init__(self, pattern):
        self.pattern = pattern

    def add_trigger(self, img):
        normalized_pattern = (self.pattern.float() / 255.0) * img.max()  # normalize the pattern
        return (img.float() + normalized_pattern).clamp(0, 255).type(torch.uint8)  # add the pattern to the image


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


class BadNets:
    def __init__(self,
                 benign_train_dataset,
                 benign_test_dataset,
                 y_target,
                 poisoned_rate,
                 poisoning_strategy):

        self.poisoned_train_dataset = CreatePoisonedDataset(benign_train_dataset,
                                                            y_target,
                                                            poisoned_rate,
                                                            poisoning_strategy)
        self.poisoned_test_dataset = CreatePoisonedDataset(benign_test_dataset,
                                                           y_target,
                                                           poisoned_rate,
                                                           poisoning_strategy)

    def save(self, filepath):
        """
        Arguments:
            filepath: String where the data should be saved (three files are created: data.npy, labels.npy and log.csv)
        """
        self.save_train(filepath)
        self.save_test(filepath)
        return

    def __write_in_csv(self, csv_writer, index, old_label, new_label):
        csv_writer.writerow(
            [index, f"{self.poisoned_train_dataset.classes[old_label]} ({old_label})",
             f"{self.poisoned_train_dataset.classes[new_label]} ({new_label})"]
        )

    def save_train(self, filepath):
        filepath += "/train"
        percentage = f"/{int(self.poisoned_train_dataset.poisoned_rate * 100)}_percent"
        if not os.path.exists(filepath + percentage):
            os.makedirs(filepath + percentage)

        data_file = filepath + percentage + "/data.npy"
        target_file = filepath + percentage + "/labels.npy"
        csv_file = open(filepath + percentage + "/log.csv", 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['index', 'old label', 'new label'])

        output_images = np.empty((self.poisoned_train_dataset.total_num, 32, 32, 3), dtype="uint8")
        output_labels = np.empty(self.poisoned_train_dataset.total_num, dtype="uint8")

        for i in range(self.poisoned_train_dataset.total_num):
            img, label = Image.fromarray(self.poisoned_train_dataset.data[i]), int(
                self.poisoned_train_dataset.targets[i])

            if i in self.poisoned_train_dataset.poisoned_indices:
                img = self.poisoned_train_dataset.poisoning_strategy(img)
                new_label = self.poisoned_train_dataset.y_target
                self.__write_in_csv(csv_writer, i, label, new_label)

            output_images[i] = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
            output_labels[i] = new_label if i in self.poisoned_train_dataset.poisoned_indices else label

        np.save(data_file, output_images)
        np.save(target_file, output_labels)
        return

    def save_test(self, filepath):
        filepath += "/test"
        percentage = f"/{int(self.poisoned_test_dataset.poisoned_rate * 100)}_percent"
        if not os.path.exists(filepath + percentage):
            os.makedirs(filepath + percentage)

        data_file = filepath + percentage + "/data.npy"
        target_file = filepath + percentage + "/labels.npy"
        csv_file = open(filepath + percentage + "/log.csv", 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['index', 'old label', 'new label'])

        output_images = np.empty((self.poisoned_test_dataset.total_num, 32, 32, 3), dtype="uint8")
        output_labels = np.empty(self.poisoned_test_dataset.total_num, dtype="uint8")

        for i in range(self.poisoned_test_dataset.total_num):
            img, label = Image.fromarray(self.poisoned_test_dataset.data[i]), int(
                self.poisoned_test_dataset.targets[i])

            if i in self.poisoned_test_dataset.poisoned_indices:
                img = self.poisoned_test_dataset.poisoning_strategy(img)
                new_label = self.poisoned_test_dataset.y_target
                self.__write_in_csv(csv_writer, i, label, new_label)

            output_images[i] = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
            output_labels[i] = new_label if i in self.poisoned_test_dataset.poisoned_indices else label

        np.save(data_file, output_images)
        np.save(target_file, output_labels)
        return


# UTILITY FUNCTIONS
def display_images(test_image, output_image):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_image)
    axes[0].set_title("Original image")
    axes[1].imshow(output_image)
    axes[1].set_title("Image with the backdoor trigger")

    for ax in axes:
        ax.axis('off')

    plt.show()


def adding_trigger_test(test_image, add_square_trigger, add_grid_trigger):
    output_image = add_square_trigger(test_image)
    display_images(test_image, output_image)

    output_image = add_grid_trigger(test_image)
    display_images(test_image, output_image)


def load_CIFAR10_data(benign_root, batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, poisoning_strategy):
    class_name = type(benign_dataset)
    if class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, poisoning_strategy)


if __name__ == '__main__':
    path = os.path.join("..", "..", "resources", "badnets")
    square_pattern = Image.open(f"{path}/trigger_image_square.png")
    grid_pattern = Image.open(f"{path}/trigger_image_grid.png")
    test_image = Image.open(f"{path}/kirby.png").convert("RGB")

    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    poisoned_image_class = "airplane"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    benign_root = os.path.join("..", "..", "datasets", "CIFAR10")

    trainset, trainloader, testset, testloader = load_CIFAR10_data(benign_root, batch_size, transform)

    add_square_trigger = AddCIFAR10Trigger(square_pattern)
    add_grid_trigger = AddCIFAR10Trigger(grid_pattern)

    # uncomment to test adding a trigger
    # adding_trigger_test(test_image, add_square_trigger, add_grid_trigger)

    badnets = BadNets(benign_train_dataset=trainset,
                      benign_test_dataset=testset,
                      y_target=classes.index(poisoned_image_class),  # airplane
                      poisoned_rate=0.05,
                      poisoning_strategy=add_square_trigger)

    # uncomment to show a sample image
    ################################
    # index = random.choice(list(badnets.poisoned_train_dataset.poisoned_indices))
    # img, target = badnets.poisoned_train_dataset[index]
    # plt.imshow(img)
    # plt.title(f"original class: {classes[target]}, new class: {poisoned_image_class}")
    # plt.show()
    ################################
    # uncomment to save poisoned model (warning: cpu/ram intensive!)
    ################################
    badnets.save(os.path.join("..", "..", "datasets", "CIFAR10", "badnets"))
    ################################
