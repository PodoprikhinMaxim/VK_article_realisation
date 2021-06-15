import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvSinc(nn.Module):

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(ConvSinc,self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.sample_rate = sample_rate

        hz = np.linspace(0, sample_rate / 2, self.out_channels + 1)

        

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        self.window_ = torch.hamming_window(self.kernel_size // 2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*np.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate

     


    def forward(self, x):

        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)

        low = torch.abs(self.low_hz_)
        high = torch.clamp(low + torch.abs(self.band_hz_), 0, self.sample_rate/2)
        band=(high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = 2 * ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left,dims = [1])
        
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        
        band_pass = band_pass / (2 * band[:,None])  

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)

class SincNet(nn.Module):
    def __init__(self):
        super(SincNet, self).__init__()
        self.conv_layers = nn.Sequential(nn.LayerNorm([1, 3200]), 
                                         ConvSinc(80, 251), nn.MaxPool1d(3),  nn.LayerNorm([80, 983]), nn.LeakyReLU(0.2), 
                                         nn.Conv1d(80, 60, kernel_size=(5,), stride=(1,)), nn.MaxPool1d(3), nn.LayerNorm([60, 326]), nn.LeakyReLU(0.2),
                                         nn.Conv1d(60, 60, kernel_size=(5,), stride=(1,)), nn.MaxPool1d(3), nn.LayerNorm([60, 107]), nn.LeakyReLU(0.2),)
        self.linear_layers = nn.Sequential(nn.LayerNorm(6420),
                                            nn.Linear(in_features=6420, out_features=2048, bias=True), nn.BatchNorm1d(2048, momentum=0.05), nn.LeakyReLU(0.2),
                                            nn.Linear(in_features=2048, out_features=2048, bias=True), nn.BatchNorm1d(2048, momentum=0.05), nn.LeakyReLU(0.2),
                                            nn.Linear(in_features=2048, out_features=2048, bias=True), nn.BatchNorm1d(2048, momentum=0.05), nn.LeakyReLU(0.2),
                                            nn.Linear(in_features=2048, out_features=462, bias=True), nn.LogSoftmax(dim=1))
        for i in [1, 4, 7, 10]:
            nn.init.xavier_uniform_(self.linear_layers[i].weight)
            if self.linear_layers[i].bias is not None:
                nn.init.constant_(self.linear_layers[i].bias.data, 0)
        
    def forward(self, x):
        batch=x.shape[0]
        seq_len=x.shape[1]
        x = x.view(batch, 1, seq_len)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x