import numpy as np
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split
import time 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset
"""

Here we want to see how long it takes to train a single pytorch model, of ____ complexity, on a synthetic dataset. 

The dataset shall be general size similar to what we will work with in predicting pedestal profiles: 

    n_samples = 10**6
    n_features = 128
    n_targets = 64

We still split the dataset into train-val-test splits, with 20% of original data as test, then 30% of remaining data as valid
"""


class DS(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        hidden_layer_sizes = [256, 512, 2048, 1024, 512, 256, 128, 64]
        self.layers = nn.ModuleList()
        self.h_in = nn.Linear(128, hidden_layer_sizes[0])
        for l in range(len(hidden_layer_sizes)-1):
            self.layers.append(nn.Linear(hidden_layer_sizes[l], hidden_layer_sizes[l+1]))
        self.h_out = nn.Linear(hidden_layer_sizes[-1], 64) 

    def forward(self, x):
        x = F.relu(self.h_in(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        out = F.relu(self.h_out(x))
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        pass

    def forward(self, x):
        pass 


def make_data(n_samples=10**6, n_features=128, n_targets=64):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, train_dl, val_dl, optimizer, criterion, device):
    # model.to(device)
    for epoch in range(500):
        running_loss = 0.0
        model.train()
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            

            if args.verbosity != 0:
                running_loss += loss.to('cpu').item()
                

                if i % 50 == 49:
                    print('[%d, %5d] train loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        # model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (inputs_val, targets_val) in enumerate(val_dl):
                outputs_val = model(inputs_val)
                val_loss = criterion(outputs_val, targets_val)


                if args.verbosity != 0:
                    running_val_loss += val_loss.to('cpu').item()

                    if i % 50 == 49:
                        print('[%d, %5d] val loss: %.3f' % (epoch + 1, i + 1, running_val_loss / 50))
                        running_val_loss = 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda" if use_cuda else "cpu")
    X_train, y_train, X_val, y_val, X_test, y_test = make_data(n_samples = args.n_samples)

    train_set, val_set = DS(X_train, y_train), DS(X_val, y_val)
    train_dl = DataLoader(train_set, batch_size=2048, num_workers=args.num_cpus, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=2048, num_workers=args.num_cpus, shuffle=True)

    simple_model = Net().double().to(device)
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start = time.time()
    train_model(simple_model, train_dl, val_dl, optimizer, criterion, device)
    end = time.time()
    print("Run time [s]: ", end - start)



import argparse 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ML Algos on HPCs.')
    parser.add_argument('--num-cpus', type=int, required=True, help='Number of CPUs used in the node')
    parser.add_argument('--n-samples', type=int, default=10**6, help='Number of samples in dataset')
    
    parser.add_argument('--verbosity', type=int, default=0, help='log_outputs')
    args = parser.parse_args()
    main()
