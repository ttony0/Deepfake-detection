from Config import *
from Model import Model

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class AvgObj:
    def __init__(self, label):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.label = label

    def update(self, val, n=1):
        self.val = 0
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / targets.size(0)


def training(dataloader):
    # variable
    acc_list = []
    loss_list = []

    # creating model
    model = Model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    #training
    print('start training...')
    for epoch in list(range(epochs)):
        model.train()
        accuracies = AvgObj('accuracy')
        losses = AvgObj('loss')
        t = tqdm(list(enumerate(dataloader)), desc='epoch%d' % (epoch + 1))
        # t = enumerate(dataloader)
        for i, (x, target) in t:
            # transfer to cuda tensor
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            # forwarding
            output = model(x)

            # calculate the loss and update the parameters
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.cpu().tolist(), x.size(0))

            # calculate the accuracy
            acc = calculate_accuracy(output, target)
            accuracies.update(acc, x.size(0))
            # print(acc)
            t.set_postfix(acc=accuracies.avg)
        acc_list.append(accuracies.avg)
        loss_list.append(losses.avg)
        # torch.save(model, MODEL_SAVE_DIR + 'epoch%d' % (epoch + 1) + '_model.pkl')
        torch.save(model.state_dict(), MODEL_SAVE_DIR + 'epoch%d' % (epoch + 1) + '_model.pkl')

    # save the final model
    # torch.save(model, MODEL_SAVE_DIR + 'final_model.pkl')
    torch.save(model.state_dict(), MODEL_SAVE_DIR + 'final_model.pkl')
    return acc_list, loss_list