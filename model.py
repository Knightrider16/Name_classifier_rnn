import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.h2o = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input_seq):
        h = torch.zeros(1, self.hidden_size).to(input_seq[0].device)
        for x in input_seq:
            combined = torch.cat((x, h), dim=1)
            h = torch.tanh(self.i2h(combined))
        output = self.h2o(h)
        return F.log_softmax(output, dim = 1)