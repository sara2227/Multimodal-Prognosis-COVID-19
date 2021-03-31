import torch.nn.functional as F
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv3d(70,2, (512, 512, 3),stride=1,padding=1,bias=False)
      self.conv2 = nn.Conv3d(2,2,kernel_size=3,stride=1,padding=1,bias=False)
      self.dropout1 = nn.Dropout3d(0.25)
      self.fc1 = nn.Linear(4, 3)
      self.fc2 = nn.Linear(3, 1)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      print(x.shape)
      x = self.conv1(x)
      print(x.shape)
      # Use the rectified-linear activation function over x
      x = F.relu(x)
      print(x.shape)

      x = self.conv2(x)
      print(x.shape)
      x = F.relu(x)
      print(x.shape)

      # Run max pooling over x
      x = F.max_pool3d(x, 2)
      print(x.shape)
      # Pass data through dropout1
      conv_out = x.view(-1, 2*2*1*1)
      # Pass data through fc1
      print(x.shape)
      x = self.fc1(conv_out)
      print("rass")
      x = F.relu(x)
      x = self.dropout1(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output