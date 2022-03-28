from Config import *
from Model import Model

import torch
from tqdm import tqdm


def testing(dataloader):
    print('testing...')
    model_path = MODEL_SAVE_DIR + 'final_model.pkl'
    # model = torch.load(model_path)
    model = Model().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # softmax = nn.Softmax()
    outputs = []
    targets = []
    with torch.no_grad():
        t = tqdm(list(enumerate(dataloader)))
        for i, (x, target) in t:
            if torch.cuda.is_available():
                x = x.cuda()
                # target = target.cuda()
            output = model(x)
            _, pred = torch.max(output, 1)
            outputs += pred.cpu().reshape(len(pred)).tolist()
            targets += target.reshape(len(target)).tolist()
    return outputs, targets