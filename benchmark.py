import csv
import logging
import os
from datetime import datetime
from itertools import combinations
from statistics import mean

import torch
from torch import optim, Tensor
from torch.hub import load
from torch.nn import MSELoss, BCEWithLogitsLoss, Sequential, Linear, ReLU, Module, Identity, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy, MulticlassAccuracy, MultilabelPrecision, \
    MulticlassPrecision
from torchmetrics.regression import MeanSquaredError, R2Score
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop, Compose, RandomRotation

from utils import MultimonDataset


def main():
    # standard setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    log.info(f"{device=}")

    tasks = ["type", "gen", "hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"]
    models = ["vgg13", "vgg19", "resnet18", "resnet50", "convnext_small", "convnext_base",
              "densenet121", "densenet169", "efficientnet_v2_s", "efficientnet_v2_l", "inception_v3"]

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
            losses[t] = CrossEntropyLoss(weight=gen_weights).to(device)
        else:
            losses[t] = MSELoss().to(device)

    # instantiate metrics and move to device
    metrics = dict()
    for phase in ["train", "test"]:
        metrics[phase] = {"type": {"acc": MultilabelAccuracy(num_labels=int(type_counts)),
                                   "prec": MultilabelPrecision(num_labels=int(type_counts))},
                          "gen": {"acc": MulticlassAccuracy(num_classes=int(gen_counts)),
                                  "prec": MulticlassPrecision(num_classes=int(gen_counts))},
                          "hp": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "att": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "def": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "spatt": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "spdef": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "speed": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "height": {"mse": MeanSquaredError(), "r2": R2Score()},
                          "weight": {"mse": MeanSquaredError(), "r2": R2Score()}}

        for task in metrics[phase]:
            for metric in metrics[phase][task]:
                metrics[phase][task][metric] = metrics[phase][task][metric].to(device)

    epochs = 200

    log.debug(f"{tasks=}")
    log.debug(f"{models=}")
    log.debug(f"{epochs=}")

    for m in models:
        log.info("Single tasks...")
        for t in tasks:
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
            train(model, [t], dataloaders, losses, metrics, optimizer, device, epochs)

            log.info(f"Finished training {m} on {t}.")

        log.info("Paired tasks...")
        for t1, t2 in combinations(tasks, 2):
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
            train(model, [t1, t2], dataloaders, losses, metrics, optimizer, device, epochs)

            log.info(f"Finished training {m} on {t1} and {t2}.")


def train(model, tasks, dataloaders, losses, metrics, optimizer, device, epochs):
    for epoch in range(epochs):
        running_loss = dict()
        for task in tasks:
            running_loss[task] = {"train": [], "test": []}
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
                    for task in tasks:
                        loss += losses[task](preds[task], labels[task])
                        running_loss[task]["train"].append(loss.item())

                    loss.backward()
                    optimizer.step()

                elif phase == "test":
                    with torch.no_grad():
                        preds = model(data)

                        loss = 0.0
                        for task in tasks:
                            loss += losses[task](preds[task], labels[task])
                            running_loss[task]["test"].append(loss.item())

                for task in tasks:
                    for metric in metrics[phase][task]:
                        metrics[phase][task][metric].update(preds[task], labels[task])

        # log epoch train and test loss and accuracy and time
        log.info(f"Epoch {epoch + 1}/{epochs}")
        for task in tasks:
            log.info(f"{task.title()} -  Train Loss: {mean(running_loss[task]['train']):.6f} "
                     f"Test Loss: {mean(running_loss[task]['test']):.6f}")
            for metric in metrics["test"][task]:
                log.info(f"{task.title()} -  Train {metric.title()}: {metrics['train'][task][metric].compute()} "
                         f"Test {metric.title()}: {metrics['test'][task][metric].compute()}")
        log.info(f"Time: {datetime.now() - now}")

        for phase in ["train", "test"]:
            for task in tasks:
                for metric in metrics[phase][task]:
                    writer.writerow({"model": model.model_name,
                                     "task1": tasks[0], "task2": tasks[1] if len(tasks) > 1 else None,
                                     "target": task, "epoch": epoch, "phase": phase, "metric": metric,
                                     "value": metrics[phase][task][metric].compute().item()})

                writer.writerow({"model": model.model_name,
                                 "task1": tasks[0], "task2": tasks[1] if len(tasks) > 1 else None,
                                 "target": task, "epoch": epoch, "phase": phase, "metric": "loss",
                                 "value": mean(running_loss[task][phase])})

        for phase in ["train", "test"]:
            for task in tasks:
                for metric in metrics[phase][task]:
                    metrics[phase][task][metric].reset()


class MultimonModel(Module):
    def __init__(self, model_name, pretrained_weights, tasks, gen_count, type_count, device):
        super(MultimonModel, self).__init__()
        self.device = device
        self.model_name = model_name
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

        if "vgg" in model_name:
            out_size = model.classifier[0].in_features
            model.classifier = Identity()
        elif "resnet" in model_name:
            out_size = model.fc.in_features
            model.fc = Identity()
        elif "convnext" in model_name:
            out_size = model.classifier.in_features
            model.classifier = Identity()
        elif "densenet" in model_name:
            out_size = model.classifier.in_features
            model.classifier = Identity()
        elif "efficientnet" in model_name:
            out_size = model.classifier.in_features
            model.classifier = Identity()
        elif "inception" in model_name:
            out_size = model.fc.in_features
            model.fc = Identity()
        else:
            raise ValueError(f"Model {model_name} not supported.")

        return model, out_size

    def get_heads(self, tasks, out_size, gen_count, type_count):
        heads = dict()
        for task in tasks:
            layers = []

            layers.append(Linear(in_features=out_size, out_features=512))
            layers.append(ReLU())

            if task == "type":
                layers.append(Linear(in_features=512, out_features=type_count))
            elif task == "gen":
                layers.append(Linear(in_features=512, out_features=gen_count))
            else:
                layers.append(Linear(in_features=256, out_features=1))

            heads[task] = Sequential(*layers).to(self.device)

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
    results_name = f"results/results-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv"
    os.makedirs(os.path.dirname(results_name), exist_ok=True)
    fields = ["model", "task1", "task2", "target", "epoch", "phase", "metric", "value"]
    with open(results_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        main()

    log.info("Finished benchmark.")
