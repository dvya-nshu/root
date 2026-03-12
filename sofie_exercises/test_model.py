import torch
import torch.nn as nn
from parser import parse_model


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1,1,3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self,x):

        x = self.conv(x)
        x = self.pool(x)
        x = self.elu(x)

        return x


model = TestModel()

parsed = parse_model(model)

print(parsed)