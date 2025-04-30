import torch
from torcheval.metrics.functional import binary_precision, binary_f1_score, binary_accuracy, binary_confusion_matrix, binary_recall
import torch.nn as nn
from process import process

PATH = "./Cancer_Data.csv"


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

    y_pred_train = model(torch.from_numpy(dataset['x_train']).double())
    confusion_train = binary_confusion_matrix(torch.squeeze(y_pred_train), torch.from_numpy(dataset['y_train']))
    accuracy_train = binary_accuracy(torch.squeeze(y_pred_train), torch.from_numpy(dataset['y_train']))
    precision_train = binary_precision(torch.squeeze(y_pred_train), torch.from_numpy(dataset['y_train']))
    recall_train = binary_recall(torch.squeeze(y_pred_train), torch.from_numpy(dataset['y_train']))
    f1_score_train = binary_f1_score(torch.squeeze(y_pred_train), torch.from_numpy(dataset['y_train']))

    print(f"Training Confusion Matrix: {confusion_train}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Training Precision: {precision_train}")
    print(f"Training Recall: {recall_train}")
    print(f"Training F1-Score: {f1_score_train}")

    model.eval()
    y_pred = model(torch.from_numpy(dataset['x_test']))
    model.train()

    confusion_test = binary_confusion_matrix(torch.squeeze(y_pred), torch.from_numpy(dataset['y_test']))
    accuracy_test = binary_accuracy(torch.squeeze(y_pred), torch.from_numpy(dataset['y_test']))
    precision_test = binary_precision(torch.squeeze(y_pred), torch.from_numpy(dataset['y_test']))
    recall_test = binary_recall(torch.squeeze(y_pred), torch.from_numpy(dataset['y_test']))
    f1_score_test = binary_f1_score(torch.squeeze(y_pred), torch.from_numpy(dataset['y_test']))

    print(f"Testing Confusion Matrix: {confusion_test}")
    print(f"Testing Accuracy: {accuracy_test}")
    print(f"Testing Precision: {precision_test}")
    print(f"Testing Recall: {recall_test}")
    print(f"Testing F1-Score: {f1_score_test}")


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = process(PATH)
    data = {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test}
    model_first(data)




