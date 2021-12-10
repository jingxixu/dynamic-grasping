from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import misc_utils as mu
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
import json

np.set_printoptions(suppress=True)


class MotionAwareDataset(Dataset):
    def __init__(self, dataset_path, object_name):
        self.dataset_path = dataset_path
        self.object_name = object_name
        self.data = np.load(os.path.join(self.dataset_path, self.object_name, 'data.npy'))
        self.data = self.data.astype(np.float32)
        self.labels = np.load(os.path.join(self.dataset_path, self.object_name, 'labels.npy'))
        self.labels = self.labels.astype(np.int64)
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MotionQualityEvaluationNet(nn.Module):
    def __init__(self):
        super(MotionQualityEvaluationNet, self).__init__()
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        # ?, 16
        # ?, 512
        x = self.fc1(x)
        x = F.relu(x)
        # ?, 512
        x = self.fc2(x)
        x = F.relu(x)
        # ?, 2
        x = self.fc3(x)
        return x


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='the path to the directory to save logs')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='the path to the directory to save models')
    parser.add_argument('--split', type=float, default=0.2,
                        help='proportion of the test set (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.object_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.log_dir = os.path.join(args.log_dir, args.object_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    return args


def train(model, device, train_loader, optimizer, epoch, writer):
    """ train one epoch """
    model.train()
    pbar = tqdm.tqdm(total=len(train_loader.dataset), desc='Train | Epoch: | Loss: | Accuracy: ')
    epoch_loss = 0.0
    num_samples = 0
    correct = 0
    positives = 0
    negatives = 0
    false_positives = 0
    false_negatives = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label.squeeze())
        loss.backward()
        optimizer.step()

        # update epoch stats
        output = output.detach()
        label = label.detach()
        epoch_loss += F.cross_entropy(output, label.squeeze(), reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        positives += pred[label == 1].shape[0]
        negatives += pred[label == 0].shape[0]
        false_positives += pred[label == 1].shape[0] - pred[label == 1].sum().item()
        false_negatives += pred[label == 0].sum().item()


        num_samples += len(data)
        pbar.update(len(data))
        pbar.set_description(
            'Train | Epoch: {} | Loss: {:.6f} | Accuracy: {:.6f} | FPR: {:.6f} | FNR: {:.6}'.
                format(epoch, epoch_loss / num_samples, correct / num_samples, false_positives / positives, false_negatives / negatives))

    epoch_loss /= len(train_loader.dataset)
    epoch_accuracy = correct / len(test_loader.dataset)
    FPR = false_positives / positives
    FNR = false_negatives / negatives
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
    writer.add_scalar('Train/FPR', FPR, epoch)
    writer.add_scalar('Train/FNR', FNR, epoch)

    pbar.close()
    return epoch_loss, epoch_accuracy, FPR, FNR


def test(model, device, test_loader, epoch, writer):
    model.eval()
    epoch_loss = 0.0
    num_samples = 0
    correct = 0
    positives = 0
    negatives = 0
    false_positives = 0
    false_negatives = 0
    pbar = tqdm.tqdm(total=len(test_loader.dataset), desc='Test | Epoch: | Loss: | Accuracy')
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            epoch_loss += F.cross_entropy(output, label.squeeze(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            positives += pred[label == 1].shape[0]
            negatives += pred[label == 0].shape[0]
            false_positives += pred[label == 1].shape[0] - pred[label == 1].sum().item()
            false_negatives += pred[label == 0].sum().item()

            num_samples += len(data)
            pbar.update(len(data))
            pbar.set_description(
                'Test | Epoch: {} | Loss: {:.6f} | Accuracy: {:.6f} | FPR: {:.6f} | FNR: {:.6}'.
                    format(epoch, epoch_loss / num_samples, correct / num_samples, false_positives / positives, false_negatives / negatives))

        epoch_loss /= len(test_loader.dataset)
        epoch_accuracy = correct / len(test_loader.dataset)
        FPR = false_positives / positives
        FNR = false_negatives / negatives
        writer.add_scalar('Test/Loss', epoch_loss, epoch)
        writer.add_scalar('Test/Accuracy', epoch_accuracy, epoch)
        writer.add_scalar('Test/FPR', FPR, epoch)
        writer.add_scalar('Test/FNR', FNR, epoch)
        pbar.close()
        return epoch_loss, epoch_accuracy, FPR, FNR


if __name__ == "__main__":
    args = get_args()
    model_metadata = vars(args)
    json.dump(model_metadata, open(os.path.join(args.save_dir, 'model_metadata.json'), 'w'), indent=4)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = MotionAwareDataset(args.dataset_path, args.object_name)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    model = MotionQualityEvaluationNet().to(device)
    writer = SummaryWriter(log_dir=args.log_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_FPR, train_FNR = train(model, device, train_loader, optimizer, epoch, writer)
        test_loss, test_accuracy, test_FPR, test_FNR = test(model, device, test_loader, epoch, writer)

        # save checkpoint
        epoch_dir_name = "epoch_{:04}_acc_{:.4f}_fpr_{:.4f}_fnr_{:.4f}".format(epoch, test_accuracy, test_FPR, test_FNR)
        epoch_dir = os.path.join(args.save_dir, epoch_dir_name)
        os.makedirs(epoch_dir)
        checkpoint_metadata = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_FPR': train_FPR,
            'train_FNR': train_FNR,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_FPR': test_FPR,
            'test_FNR': test_FNR
        }
        json.dump(checkpoint_metadata, open(os.path.join(epoch_dir, 'checkpoint_metadata.yaml'), 'w'), indent=4)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "motion_ware_net.pt"))