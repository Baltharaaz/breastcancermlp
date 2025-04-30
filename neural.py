import torch
from torcheval.metrics import BinaryPrecision, BinaryF1Score, BinaryAccuracy, BinaryConfusionMatrix, BinaryRecall
import torch.nn as nn
from process import process

PATH = "./Cancer_Data.csv"

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation2 = nn.Softmax(dim=1)


def model_first(dataset):
    model = nn.Sequential(
        nn.Linear(6, 10, dtype=torch.double),
        nn.ReLU(),
        nn.Linear(10, 1, dtype=torch.double),
        nn.Sigmoid()
    )
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10000

    model.train()
    for epoch in range(num_epochs):
        y_pred = model(torch.from_numpy(dataset['x_train']).double())
        loss = loss_function(y_pred, torch.from_numpy(dataset['y_train']).unsqueeze(1).double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(torch.from_numpy(dataset['x_test']))
    for y_hat, y in zip(y_pred, dataset['y_test']):
        print(y_hat)
        print(y)

    model.train()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = process(PATH)
    data = {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test}
    model_first(data)




