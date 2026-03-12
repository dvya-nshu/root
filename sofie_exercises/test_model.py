import torch
import torch.nn as nn
from parser import parse_model


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(5,5)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self,x):

        x = self.fc(x)
        x = self.elu(x)

        return x


model = TestModel()

parsed = parse_model(model)

print(parsed)
