# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from hmdata import TotalCapture
# from torchcoder.models import AE
from torch.utils.data import DataLoader
from torch.autograd import Variable
# import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
data_path = '/media/ywj/Data/totalcapture/totalcapture'
tp_data = TotalCapture(data_path, cues='vicon-imu-ori', mode='debug-train')
tp_test_data = TotalCapture(data_path, cues='vicon-imu-ori', mode='debug-test')

tp_data_loader = DataLoader(tp_data, batch_size=100, shuffle=True)
tp_test_data_loader = DataLoader(tp_test_data, batch_size=100, shuffle=True)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(21*4, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            # nn.ReLU(True),
            # nn.Linear(32, 16),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(16, 32),
            # nn.ReLU(True),
            nn.Linear(8, 12),
            nn.ReLU(True),
            nn.Linear(12, 13*4),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out


model = AE()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

criterion = nn.MSELoss(size_average=False)

model = model.to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

for e in range(1000):
    model.train()
    scheduler.step()
    train_loss = 0
    for vicon, imu in tp_data_loader:
        vicon = vicon.view(vicon.shape[0], -1)
        vicon = Variable(vicon)

        imu = imu.view(imu.shape[0], -1)
        imu = Variable(imu)

        vicon, imu = vicon.to(device), imu.to(device)

        output = model(vicon)
        loss = criterion(output, imu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= len(tp_data_loader.dataset)
    print('epoch: {}, Loss: {:.4f}'.format(e + 1, train_loss))

    if e % 10 == 0:
        print('=====================')
        model.eval()
        test_loss = 0
        for vicon, imu in tp_test_data_loader:
            vicon = vicon.view(vicon.shape[0], -1)
            vicon = Variable(vicon)

            imu = imu.view(imu.shape[0], -1)
            imu = Variable(imu)

            vicon, imu = vicon.to(device), imu.to(device)

            output = model(vicon)
            loss = criterion(output, imu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss += loss
        test_loss /= len(tp_test_data_loader.dataset)
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, test_loss))
        print('=====================')
