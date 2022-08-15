import os

from pandas import read_csv
from pprint import pprint

import torch
from matplotlib import pyplot as plt
from torch import zeros, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import RandomResizedCrop, Compose, ToTensor
from torchvision.transforms.functional import resize


class MultimonDataset(Dataset):
    def __init__(self, data_file, part_file, img_path, device=None, transforms=None,
                 partition="train", data_format="raw", output_size=64):
        if data_file is None or part_file is None or img_path is None:
            raise ValueError("data_file, part_file, and img_path must be specified")

        if partition not in ["train", "test", "val"]:
            raise ValueError("partition must be one of 'train', 'test', or 'val'")
        else:
            partition = ["train", "test", "val"].index(partition)

        if data_format not in ["raw", "std", "norm"]:
            data_format = "raw"

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_file = data_file
        self.img_path = img_path
        self.device = device
        self.transforms = transforms
        self.output_size = output_size

        # load partitions
        self.partitions = self.parse_partitions(part_file, partition)

        # loads image key and targets
        self.data = self.parse_datafile(data_file, data_format)

    def parse_partitions(self, part_file, partition):
        parts = read_csv(part_file)
        filter = parts["split"] == partition
        return parts["index"][filter]

    def parse_datafile(self, data_path, data_format="std"):
        data = read_csv(data_path)

        type_counts = max(data["type1"].max(), data["type2"].max()) + 1
        gen_counts = data["gen"].max()
        stats = [x+"_"+data_format for x in ["hp", "att", "def", "spatt", "spdef", "speed"]]

        filter = data["index"].isin(self.partitions)
        data = data[filter]

        data = {row["index"]: {"type": zeros(type_counts).scatter_(0, tensor([row["type1"], row["type2"]]), 1),
                               "gen": zeros(gen_counts).scatter_(0, tensor([row["gen"]-1]), 1),
                               **{x: tensor(row[x]).float() for x in stats}}
                for index, row in data.iterrows()}

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]

        img_path = os.path.join(self.img_path, key)
        image = read_image(img_path).float().to(self.device) / 255.0
        if self.transforms is not None:
            image = self.transforms(image)
        if self.output_size is not None:
            image = resize(image, self.output_size)

        labels = self.data[key]

        return image, labels


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(device)

    transforms = Compose([RandomResizedCrop(64), ToTensor()])

    random_data = MultimonDataset(data_file="data.csv", part_file="partitions.csv", img_path="./sprites/processed",
                                  device=device, transforms=None, partition="train")

    loader = DataLoader(random_data, batch_size=1, shuffle=True, num_workers=1)

    for data, labels in random_data:
        data = data.cpu()
        print(data.permute(1, 2, 0).size())
        plt.imshow(data.permute(1, 2, 0))
        plt.show()
        print("Here!")
        pprint(labels)
        input("Enter to continue...")
