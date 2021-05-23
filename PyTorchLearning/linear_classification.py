import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(x_train, x_test, y_train, y_test):
    n, d = x_train.shape

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = nn.Sequential(
        nn.Linear(d, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
    y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

    epochs = 1000
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for i in range(epochs):
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        outputs_test = model(x_test)
        loss_test = criterion(outputs_test, y_test)

        train_losses[i] = loss.item()
        test_losses[i] = loss_test.item()

        if (i + 1) % 50 == 0:
            print(f'Epoch {i + 1} / {epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')

    plot_losses(train_losses, test_losses)
    evaluate_accuracy(model, x_train, x_test, y_train, y_test)


def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()


def evaluate_accuracy(model, x_train, x_test, y_train, y_test):
    with torch.no_grad():
        p_train = model(x_train)
        p_train = np.round(p_train.numpy())
        train_acc = np.mean(y_train.numpy() == p_train)

        p_test = model(x_test)
        p_test = np.round(p_test.numpy())
        test_acc = np.mean(y_test.numpy() == p_test)

        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


def perform_linear_classification():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
    train_model(x_train, x_test, y_train, y_test)


def main():
    perform_linear_classification()


if __name__ == '__main__':
    main()
