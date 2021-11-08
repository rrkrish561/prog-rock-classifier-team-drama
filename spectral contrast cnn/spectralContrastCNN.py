import torch
import torchvision
import math
import numpy as np
import preprocessImages
from sklearn.model_selection import train_test_split


LEARNING_RATE = 1e-6
NUM_EPOCHS = 50
BATCH_SIZE = 32
HEIGHT, WIDTH = 128, 646


class SpectralContrastCNN(torch.nn.Module):
    def __init__(self, h0, w0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        h1 = h0 - 2
        w1 = w0 - 2
        self.pool1 = torch.nn.MaxPool2d(2)
        h2 = math.floor((h1 - 2) / 2 + 1)
        w2 = math.floor((w1 - 2) / 2 + 1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        h3 = h2 - 2
        w3 = w2 - 2
        self.pool2 = torch.nn.MaxPool2d(2)
        h4 = math.floor((h3 - 2) / 2 + 1)
        w4 = math.floor((w3 - 2) / 2 + 1)
        self.fc1 = torch.nn.Linear(64 * h4 * w4, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.pool1(self.dropout(self.relu(self.conv1(x))))
        x = self.pool2(self.dropout(self.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    net = SpectralContrastCNN(HEIGHT, WIDTH)

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize((HEIGHT, WIDTH))
    # ])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    X, y = preprocessImages.collect_image_data(HEIGHT, WIDTH)
    print('Finished preprocessing!')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    X_train = torch.autograd.Variable(torch.from_numpy(X_train)).float()
    y_train = torch.autograd.Variable(torch.from_numpy(y_train)).float()
    X_test = torch.autograd.Variable(torch.from_numpy(X_test)).float()
    y_test = torch.autograd.Variable(torch.from_numpy(y_test)).float()

    # torch.permute(X_train, (0, 3, 1, 2))
    # torch.permute(y_train, (0, 3, 1, 2))
    # torch.permute(X_test, (0, 3, 1, 2))
    # torch.permute(y_test, (0, 3, 1, 2))

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)
    print('Finished making dataloaders!')

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            # outputs = np.squeeze(outputs)
            if BATCH_SIZE == 1:
                outputs = torch.squeeze(outputs, dim=0)
            else:
                try:
                    outputs = torch.squeeze(outputs)
                except:
                    outputs = torch.squeeze(outputs, dim=0)

            # print(outputs, outputs.size())
            # print(labels, labels.size())

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print(
                f"Epoch: [{epoch+1}/{NUM_EPOCHS}], Step: [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    print('Finished training!')
    correct = 0
    total = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            outputs = net(inputs)

            outputs = outputs.detach().numpy()

            y_pred = np.squeeze(outputs)
            if y_pred < 0.5:
                y_pred = 0
            else:
                y_pred = 1

            if y_pred == labels.item():
                correct += 1
            total += 1

            predictions.append(y_pred)
            actuals.append(labels.item())

    print(predictions)
    print('Accuracy:', float(correct) / float(total))
