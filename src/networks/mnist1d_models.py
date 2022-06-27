# The MNIST-1D dataset | 2020
# Sam Greydanus

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBase(nn.Module):
  def __init__(self, input_size=40, num_classes=10, **kwargs):
    super(LinearBase, self).__init__()
    self.fc = nn.Linear(input_size, num_classes)
    print("Initialized LinearBase model with {} parameters".format(self.count_params()))
    self.head_var = 'fc'

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    return self.fc(x)

class MLPBase(nn.Module):
  def __init__(self, input_size=40, num_classes=10, hidden_size=100, **kwargs):
    super(MLPBase, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.fc = nn.Linear(hidden_size, num_classes)
    print("Initialized MLPBase model with {} parameters".format(self.count_params()))
    self.head_var = 'fc'

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    h = self.linear1(x).relu()
    h = h + self.linear2(h).relu()
    return self.fc(h)

class ConvBase(nn.Module):
    def __init__(self, num_classes=10, channels=25, linear_in=125, **kwargs):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.fc = nn.Linear(linear_in, num_classes) # flattened channels -> 10 (assumes input has dim 50)
        self.head_var = 'fc'
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        x = x.view(-1,1,x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.fc(h3) # a linear classifier goes on top

class GRUBase(torch.nn.Module):
  def __init__(self, input_size=40, num_classes=10, hidden_size=6, time_steps=40, bidirectional=True, **kwargs):
    super(GRUBase, self).__init__()
    self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=bidirectional)
    flat_size = 2*hidden_size*time_steps if bidirectional else hidden_size*time_steps
    self.fc = torch.nn.Linear(flat_size, num_classes)
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.head_var = 'fc'
    print("Initialized GRUBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x, h0=None): # assumes seq has [batch, time]
    x = x.view(*x.shape[:2], 1) # add a spatial dimension
    k = 2 if self.bidirectional else 1
    h0 = 0*torch.randn(k, x.shape[0], self.hidden_size) if h0 is None else h0
    h0 = h0.to(x.device) # GPU support

    output, hn = self.gru(x, h0)
    output = output.reshape(output.shape[0],-1) # [batch, time*hidden_size]
    return self.fc(output)
