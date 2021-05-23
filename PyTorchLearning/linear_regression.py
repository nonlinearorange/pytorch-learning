import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    n = 20  # N.
    x = np.random.random(n) * 10 - 5

    # Synthetic data.
    # Line + Gaussian Noise.
    m = 0.5
    b = -1
    y = m * x + b + np.random.randn(n)

    # plt.scatter(x, y)
    # plt.show()

    return n, x, y


def train_model(n, x, y):
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = x.reshape(n, 1)
    y = y.reshape(n, 1)

    inputs = torch.from_numpy(x.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))

    epochs = 30
    losses = []

    for i in range(epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        print(f'Epoch {i + 1} / {epochs}, Loss: {loss.item():.4f}')

    # plt.plot(losses)
    # plt.show()

    predicted = model(inputs).detach().numpy()
    plot_predicted(predicted, x, y)

    m = w = model.weight.data.numpy()
    b = model.bias.data.numpy()

    print(f'Expected: m = 0.5, b = -1')
    print(f'Estimated Line: m = {w}, b = {b}')

    return m[0][0], b[0]


def plot_predicted(predicted, x, y):
    plt.scatter(x, y, label='Original Data')
    plt.plot(x, predicted, label='Fitted Line')
    plt.legend()
    plt.show()


def perform_simple_linear_regression():
    n, x, y = generate_data()
    return train_model(n, x, y)


def visualize(tests):
    m = [item[0] for item in tests]
    b = [item[1] for item in tests]

    plt.scatter(m, b)
    plt.show()


def main():
    tests = []
    for _ in range(3):
        m, b = perform_simple_linear_regression()
        tests.append((m, b))

    visualize(tests)


if __name__ == '__main__':
    main()
