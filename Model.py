from Config import *
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset

data_transforms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((im_size, im_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])


class DataSet(Dataset):
    def __init__(self, video_list):
        if len(video_list) % BATCH_SIZE == 0:
            self.videos = video_list
        else:
            self.videos = video_list[:-len(video_list) % BATCH_SIZE]
        self.minframes = 999
        for video in video_list:
            cap = cv2.VideoCapture(video)
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < self.minframes:
                self.minframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        self.minframes = int(self.minframes / 6)
        self.transform = data_transforms

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames = []
        video = self.videos[idx]
        label = REAL if video.split('\\')[-2] == 'REAL' else FAKE
        cap = cv2.VideoCapture(video)
        count = 1
        if cap.isOpened():
            flag, frame = cap.read()
            frames.append(self.transform(frame))
        else:
            print(video.split('\\')[-2] + '\\' + video.split('\\')[-1] + ' is corrupted!')
            flag = False
        while flag:
            flag, frame = cap.read()
            if count % timeF == 0:
                if len(frames) < self.minframes and len(frames) <= sequence_length:
                    frames.append(self.transform(frame))
                    count = 0
                else:
                    break
            count += 1
        frames = torch.stack(frames)
        cap.release()
        # print('length:', len(frames), 'label', label)
        return frames, label


class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.resnext = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.resnext(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_length, 2048)
        x, _ = self.lstm(x, None)
        x = self.linear(torch.mean(x, dim=1))
        x = self.dp(x)
        return x