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
tp_data = TotalCapture(data_path, cues='vicon-imu-ori', mode='train')
tp_data_loader = DataLoader(tp_data, batch_size=100, shuffle=False)
tp_test_data = TotalCapture(data_path, cues='vicon-imu-ori', mode='test')
tp_test_data_loader = DataLoader(tp_test_data, batch_size=100, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(21*4, 32)
        self.fc21 = nn.Linear(32, 16)
        self.fc22 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 13*4)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

criterion = nn.MSELoss(size_average=False)

model = model.to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

for e in range(1000):
    model.train()
    scheduler.step()
    train_loss = 0
    train_error = 0
    for vicon, imu in tp_data_loader:
        vicon = vicon.view(vicon.shape[0], -1)
        vicon = Variable(vicon)

        imu = imu.view(imu.shape[0], -1)
        imu = Variable(imu)

        vicon, imu = vicon.to(device), imu.to(device)

        recon_batch, mu, logvar = model(vicon)
        # output = model(vicon)
        loss = loss_function(recon_batch, imu, mu, logvar)
        train_loss += loss

        error = criterion(recon_batch, imu)
        train_error += error

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(tp_data_loader.dataset)
    train_error /= len(tp_data_loader.dataset)
    print('epoch: {}, Loss: {:.4f}'.format(e + 1, train_loss))
    print('epoch: {}, error: {:.4f}'.format(e + 1, train_error))
    if e % 10 == 0:
        print('=====================')
        model.eval()
        test_loss = 0
        test_error = 0
        for vicon, imu in tp_test_data_loader:
            vicon = vicon.view(vicon.shape[0], -1)
            vicon = Variable(vicon)

            imu = imu.view(imu.shape[0], -1)
            imu = Variable(imu)

            vicon, imu = vicon.to(device), imu.to(device)

            recon_batch, mu, logvar = model(vicon)
            loss = loss_function(recon_batch, imu, mu, logvar)

            error = criterion(recon_batch, imu)
            test_error += error

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss += loss
        test_loss /= len(tp_test_data_loader.dataset)
        test_error /= len(tp_test_data_loader.dataset)
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, test_loss))
        print('epoch: {}, error: {:.4f}'.format(e + 1, test_error))
        print('=====================')
