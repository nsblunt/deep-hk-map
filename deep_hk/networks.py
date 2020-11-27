import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NormalizedLinear(nn.Module):
  def __init__(self, size_in, size_out):
    super().__init__()
    self.size_in = size_in
    self.size_out = size_out
    weights = torch.Tensor(size_out, size_in)
    self.weights = nn.Parameter(weights)
    bias = torch.Tensor(size_out)
    self.bias = nn.Parameter(bias)

    # initialize the weights:
    nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    max = 1/math.sqrt(fan_in)
    # initialize the biases:
    nn.init.uniform_(self.bias, -max, max)

  def forward(self, x):
    w_times_x = torch.mm(x, self.weights.t())
    with_bias = torch.add(w_times_x, self.bias)
    out = F.normalize(
        with_bias.view(with_bias.size(0), with_bias.size(1)),
        p=2,
        dim=1)
    return out

class LinearNet(nn.Module):
  def __init__(self, layers_list):
    super(LinearNet, self).__init__()
    self.layers = nn.ModuleList(layers_list)

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    # output layer
    x = self.layers[-1](x)
    return x

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()

def create_linear_layers(num_input,
                         num_hidden,
                         num_output,
                         wave_function_output=False):
  layers_list = []
  # input layer:
  layers_list.append( nn.Linear(num_input, num_hidden[0]) )
  # hidden layers:
  for i in range(1, len(num_hidden)):
    layers_list.append( nn.Linear(num_hidden[i-1], num_hidden[i]) )
  # output layer:
  if wave_function_output:
    layers_list.append( NormalizedLinear(num_hidden[-1], num_output) )
  else:
    layers_list.append( nn.Linear(num_hidden[-1], num_output) )

  return layers_list
