from Config import *
from Model import DataSet
from Preprocessing import data_processing
from Training import training
from Testing import testing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def plot_graph(data, epochs, plot_type):
    x_plot = range(1, epochs + 1)
    plt.plot(x_plot, data, 'g', label=plot_type + ' graph')
    plt.xlabel('Epochs')
    plt.ylabel(plot_type)
    plt.legend()
    plt.show()


def print_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    label = ['REAL', 'FAKE']
    df_cm = pd.DataFrame(cm, label, label)
    sn.set()
    heatmap = sn.heatmap(df_cm, annot=True, cmap='OrRd')  # font size
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    calculated_acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print("Calculated Accuracy", calculated_acc * 100)


if __name__ == '__main__':
    print('processing data...')
    train_list, test_list = data_processing()

    train_data = DataSet(train_list)
    test_data = DataSet(test_list)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)

    # training
    acc_list, loss_list = training(train_dataloader)
    plot_graph(acc_list, epochs, 'accuracy')
    plot_graph(loss_list, epochs, 'loss')

    # testing
    outputs, targets = testing(test_dataloader)
    print_confusion_matrix(outputs, targets)