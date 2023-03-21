import logging
from datetime import datetime
from itertools import combinations
from statistics import mean

import torch
from torch import topk, optim, Tensor
from torch.hub import load
from torch.nn import MSELoss, BCEWithLogitsLoss, Sequential, Linear, ReLU, Module, Identity
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop, Compose, RandomRotation

from utils import MultimonDataset


def main():
    # standard setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    log.info(f"{device=}")

    tasks = ["type", "gen", "hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"]
    models = ["vgg13", "vgg19", "resnet18", "resnet50", "resnet152", "alexnet", "convnext_small", "convnext_large",
              "densenet121", "densenet169", "efficientnet_v2_s", "efficientnet_v2_l"]
    models = models[:4]

    # load dataloaders
    dataloaders = dict()
    for part in ["test", "train"]:
        data = MultimonDataset(data_file="data.csv", part_file="partitions.csv", img_path="./sprites/processed",
                               device=device, transforms=None, partition=part)
        dataloaders[part] = DataLoader(data, batch_size=64, shuffle=True)

    rand_transforms = Compose([RandomResizedCrop(256), RandomHorizontalFlip(), RandomRotation(15)])

    type_weights = Tensor([dataloaders["train"].dataset.type_weights[k]
                           for k in sorted(dataloaders["train"].dataset.type_weights.keys())]).to(device)
    gen_weights = Tensor([dataloaders["train"].dataset.gen_weights[k]
                          for k in sorted(dataloaders["train"].dataset.gen_weights.keys())]).to(device)

    gen_counts = dataloaders["train"].dataset.gen_counts
    type_counts = dataloaders["train"].dataset.type_counts

    losses = dict()
    for t in tasks:
        if t == "type":
            losses[t] = BCEWithLogitsLoss(weight=type_weights).to(device)
        elif t == "gen":
            losses[t] = BCEWithLogitsLoss(weight=gen_weights).to(device)
        else:
            losses[t] = MSELoss().to(device)

    epochs = 10

    log.debug(f"{tasks=}")
    log.debug(f"{models=}")
    log.debug(f"{epochs=}")

    # single tasks
    log.info("Single tasks...")
    for t in tasks:
        for m in models:
            log.info(f"Training {m} on {t}...")

            # load model with pretrained weights
            weights = load("pytorch/vision:v0.14.0", "get_model_weights", name=m)
            weights = [weight for weight in weights][0]
            model = MultimonModel(m, weights, [t], gen_counts, type_counts, device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # update transforms with model specific preprocessing
            dataloaders["train"].dataset.transforms = Compose([rand_transforms, weights.transforms()])
            dataloaders["test"].dataset.transforms = Compose([weights.transforms()])

            # train model
            train(model, [t], dataloaders, losses, optimizer, device, epochs)

            log.info(f"Finished training {m} on {t}.")

    # paired tasks
    log.info("Paired tasks...")
    for t1, t2 in combinations(tasks, 2):
        for m in models:
            log.info(f"Training {m} on {t1} and {t2}...")

            # load model with pretrained weights
            weights = load("pytorch/vision:v0.14.0", "get_model_weights", name=m)
            weights = [weight for weight in weights][0]
            model = MultimonModel(m, weights, [t1, t2], gen_counts, type_counts, device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # update transforms with model specific preprocessing
            dataloaders["train"].dataset.transforms = Compose([rand_transforms, weights.transforms()])
            dataloaders["test"].dataset.transforms = Compose([weights.transforms()])

            # train model
            train(model, [t1, t2], dataloaders, losses, optimizer, device, epochs)

            log.info(f"Finished training {m} on {t1} and {t2}.")


def train(model, tasks, dataloaders, losses, optimizer, device, epochs):
    for epoch in range(epochs):
        running_loss = {"train": [], "test": []}
        running_acc = {"train": [], "test": []}
        now = datetime.now()

        for phase in ["train", "test"]:
            dl = dataloaders[phase]

            for data, labels in dl:
                data = data.to(device)
                labels = {l: labels[l].to(device) for l in labels}

                optimizer.zero_grad()

                if phase == "train":
                    preds = model(data)

                    loss = 0.0
                    acc = 0.0
                    for task in tasks:
                        loss += losses[task](preds[task], labels[task])
                        acc += get_acc(task, preds[task], labels[task])
                    running_loss["train"].append(loss.item())
                    running_acc["train"].append(acc / len(tasks))

                    loss.backward()
                    optimizer.step()

                elif phase == "test":
                    with torch.no_grad():
                        preds = model(data)

                        loss = 0.0
                        acc = 0.0
                        for task in tasks:
                            loss += losses[task](preds[task], labels[task])
                            acc += get_acc(task, preds[task], labels[task])
                        running_loss["test"].append(loss.item())
                        running_acc["test"].append(acc / len(tasks))

        # log epoch train and test loss and accuracy and time
        log.info(
            f"Epoch {epoch} Train Loss: {mean(running_loss['train']):.6f} Test Loss: {mean(running_loss['test']):.6f}")
        log.info(f"Epoch {epoch} Train Acc: {mean(running_acc['train']):.3%} Test Acc: {mean(running_acc['test']):.3%}")
        log.info(f"Epoch {epoch} Time: {datetime.now() - now}")


def get_acc(task, preds, labels):
    if task == "type":  # sum of predictions where at least one is correct
        return sum([sum(x in p_ind for x in l_ind) > 0 for p_ind, l_ind in
                    zip(topk(preds, dim=1, k=2).indices.tolist(),
                        topk(labels, dim=1, k=2).indices.tolist())]) / preds.size()[0]
    elif task == "gen":  # sum of predictions where gen is correct
        return sum([p_ind == l_ind for p_ind, l_ind in
                    zip(topk(preds, dim=1, k=1).indices.tolist(),
                        topk(labels, dim=1, k=1).indices.tolist())]) / preds.size()[0]
    else:  # MSE since accuracy is not a simple metric for regression tasks
        return mean([(p - l) ** 2 for p, l in zip(preds.squeeze().tolist(), labels.squeeze().tolist())])


class MultimonModel(Module):
    def __init__(self, model_name, pretrained_weights, tasks, gen_count, type_count, device):
        super(MultimonModel, self).__init__()
        self.device = device
        self.model, out_size = self.get_base_model(model_name, pretrained_weights)
        self.heads = self.get_heads(tasks, out_size, gen_count, type_count)

    def forward(self, x):
        x = self.model(x)

        preds = dict()
        for task, head in self.heads.items():
            preds[task] = head(x)

        return preds

    def get_base_model(self, model_name, pretrained_weights):
        model = load("pytorch/vision:v0.14.0", model_name, weights=pretrained_weights).to(self.device)

        if model_name in ["vgg11", "vgg16"]:
            out_size = model.classifier[0].in_features
            model.classifier = Identity()
        elif model_name in ["resnet18", "resnet50"]:
            out_size = model.fc.in_features
            model.fc = Identity()
        elif model_name in ["convnext_tiny", "convnext_medium"]:
            out_size = model.fc.in_features
            model.fc = Identity()
        else:
            raise ValueError(f"Model {model_name} not supported.")

        return model, out_size

    def get_heads(self, tasks, out_size, gen_count, type_count):
        heads = dict()
        for task in tasks:
            if task == "type":
                heads[task] = Sequential(Linear(in_features=out_size, out_features=512),
                                         ReLU(),
                                         Linear(in_features=512, out_features=type_count)).to(self.device)
            elif task == "gen":
                heads[task] = Sequential(Linear(in_features=out_size, out_features=512),
                                         ReLU(),
                                         Linear(in_features=512, out_features=gen_count)).to(self.device)
            else:
                heads[task] = Sequential(Linear(in_features=out_size, out_features=256),
                                         ReLU(),
                                         Linear(in_features=256, out_features=1)).to(self.device)

        return heads


if __name__ == '__main__':
    # log debug to file and info to console
    logging.basicConfig(filename='benchmark.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    log = logging.getLogger()

    log.info("Starting benchmark...")

    # create results csv with current datetime in name
    results_name = f"results-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv"

    main()

    log.info("Finished benchmark.")
