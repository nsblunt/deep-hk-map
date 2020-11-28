import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class NormalizedLinear(nn.Module):
  """Linear (affine) layer where the output is normalised to 1 with the
     2-norm.
  """

  def __init__(self, size_in, size_out):
    """Initialise the layer object.

    Args:
      size_in: int
        The number of inputs to the layer.
      size_out: int
        The number of outputs from the layer.

    Other attributes:
      weights: torch tensor of size (size_out, size_in)
        The parameters for the linear transformation of the input.
      bias: torch tensor of dim (noutput)
        The shift applied to the output of the linear transformation.
    """
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
    """Apply the transformation to the input.

    Args:
      x: torch tensor of dim (size_in)
        the data to be passed into the layer.
    """
    w_times_x = torch.mm(x, self.weights.t())
    with_bias = torch.add(w_times_x, self.bias)
    out = F.normalize(
        with_bias.view(with_bias.size(0), with_bias.size(1)),
        p=2,
        dim=1)
    return out


def create_linear_layers(num_input,
                         num_hidden,
                         num_output,
                         wf_output=False):
  """Create a list of Torch linear layer objects.

  Args:
    num_input: int
      The number of values passed into the input of the network.
    num_hidden: list of int
      A list specifying the number of units for each hidden layer.
    num_output: int
      The number of values passed out of the network.
    wf_output: bool
      If true, then the output layer is normalised, as appropriate for
      outputting a normalised wave function.
  """

  layers_list = []

  # input layer:
  layers_list.append( nn.Linear(num_input, num_hidden[0]) )

  # hidden layers:
  for i in range(1, len(num_hidden)):
    layers_list.append( nn.Linear(num_hidden[i-1], num_hidden[i]) )

  # output layer:
  if wf_output:
    layers_list.append( NormalizedLinear(num_hidden[-1], num_output) )
  else:
    layers_list.append( nn.Linear(num_hidden[-1], num_output) )

  return layers_list


class LinearNet(nn.Module):
  """An neural network of linear (affine) layers."""

  def __init__(self, layers_list):
    """Initialises the network layers.

    Args:
      layers_list: list
        A list of the layers to be applied between nonlinearities, including
        the input and output layers. This can be created, for example, using
        the create_linear_layers function.
    """
    super(LinearNet, self).__init__()
    self.layers = nn.ModuleList(layers_list)

  def forward(self, x):
    """Pass the input through the network.

    Args:
      x: torch tensor
        The batch of input data to be passed through the network.
    """
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    # output layer
    x = self.layers[-1](x)
    return x

  def save(self, path):
    """Save the net state to a file.

    Args:
      path: string
        The path and name of the file where the network will be saved.
    """
    torch.save(self.state_dict(), path)

  def load(self, path):
    """Load the network state from a file.

    Args:
      path: string
        The path and name of the file where the network will be loaded from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()
