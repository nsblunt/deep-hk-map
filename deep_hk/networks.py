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


def create_linear_layers(ninput,
                         num_hidden,
                         noutput,
                         wf_output=False):
  """Create a list of Torch linear layer objects.

  Args:
    ninput: int
      The number of values passed into the input of the network.
    num_hidden: list of int
      A list specifying the number of units for each hidden layer.
    noutput: int
      The number of values passed out of the network.
    wf_output: bool
      If true, then the output layer is normalised, as appropriate for
      outputting a normalised wave function.
  """

  layers_list = []

  # input layer:
  layers_list.append( nn.Linear(ninput, num_hidden[0]) )

  # hidden layers:
  for i in range(1, len(num_hidden)):
    layers_list.append( nn.Linear(num_hidden[i-1], num_hidden[i]) )

  # output layer:
  if wf_output:
    layers_list.append( NormalizedLinear(num_hidden[-1], noutput) )
  else:
    layers_list.append( nn.Linear(num_hidden[-1], noutput) )

  return layers_list

def create_conv1d_layers(ninput,
                         noutput,
                         num_in_channels,
                         num_out_channels,
                         kernel_size):
  """Create a list of Torch conv1d layer objects. The final layer is
     fully connected, outputting noutput values.

  Args:
    ninput: int
      The number of values passed into the input of the network.
    noutput: int
      The number of values passed out of the network.
    num_in_channels: int
      The number of channels for the input data.
    num_out_channels: list of int
      A list specifying the number of output channels for each
      convolutional layer. The length of this list is also used to
      specify the number of such layers in total.
    kernel_size: int
      The size of the kernel applied in convolutional layers.
  """

  layers_list = []

  padding = int((kernel_size-1)/2)

  # input layer:
  layers_list.append( nn.Conv1d(
      in_channels=num_in_channels,
      out_channels=num_out_channels[0],
      kernel_size=kernel_size,
      padding=padding,
      padding_mode='circular')
  )

  # hidden layers:
  for i in range(1, len(num_out_channels)):
    layers_list.append( nn.Conv1d(
        in_channels=num_out_channels[i-1],
        out_channels=num_out_channels[i],
        kernel_size=kernel_size,
        padding=padding,
        padding_mode='circular')
    )

  # output layer (fully connected):
  layers_list.append( nn.Linear(
      in_features=ninput*num_out_channels[-1],
      out_features=noutput)
  )

  return layers_list


class LinearNet(nn.Module):
  """A neural network of linear (affine) layers."""

  def __init__(self, layers_list, activation_fn):
    """Initialises the network layers.

    Args:
      layers_list: list
        A list of the layers to be applied between nonlinearities,
        including the input and output layers. This can be created, for
        example, using the create_linear_layers function.
      activation_fn: string
        String representing the activation function, which is used to
        select a torch function below.
    """
    super(LinearNet, self).__init__()
    self.layers = nn.ModuleList(layers_list)
    self.activation_fn = activation_fn

    if activation_fn == 'relu':
      self.activation_fn = nn.ReLU()
    elif activation_fn == 'elu':
      self.activation_fn = nn.ELU()
    elif activation_fn == 'sigmoid':
      self.activation_fn = nn.Sigmoid()
    elif activation_fn == 'tanh':
      self.activation_fn = nn.Tanh()

  def forward(self, x):
    """Pass the input through the network.

    Args:
      x: torch tensor
        The batch of input data to be passed through the network.
    """
    for layer in self.layers[:-1]:
      x = self.activation_fn(layer(x))
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
        The path and name of the file where the network will be loaded
       from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()


class ConvNet(nn.Module):
  """A network of convolutional layers. The final layer is a fully
     connected linear layer.
  """

  def __init__(self, layers_list, ninput, activation_fn):
    """Initialises the network layers.

    Args:
      layers_list: list
        A list of the layers to be applied between nonlinearities,
        including the input and output layers. This can be created, for
        example, using the create_linear_layers function.
      ninput: int
        The number of features passed into the net.
      activation_fn: string
        String representing the activation function, which is used to
        select a torch function below.
    """
    super(ConvNet, self).__init__()
    self.layers = nn.ModuleList(layers_list)
    self.activation_fn = activation_fn

    # Number of ouput channels from the final convolutional layer.
    out_channels_final = self.layers[-2].out_channels
    # Number of features input to the final layer.
    self.nfeatures_final = out_channels_final * ninput

    if activation_fn == 'relu':
      self.activation_fn = nn.ReLU()
    elif activation_fn == 'elu':
      self.activation_fn = nn.ELU()
    elif activation_fn == 'sigmoid':
      self.activation_fn = nn.Sigmoid()
    elif activation_fn == 'tanh':
      self.activation_fn = nn.Tanh()

  def forward(self, x):
    """Pass the input through the network.

    Args:
      x: torch tensor
        The batch of input data to be passed through the network.
    """
    # Need to add a dimension, recognised as the single input channel.
    x = x[:, None, :]

    # Apply the convolutional layers.
    for layer in self.layers[:-1]:
      x = self.activation_fn(layer(x))

    # Merge the output channels together.
    x = x.view(-1, self.nfeatures_final)

    # Fully connected output layer.
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
        The path and name of the file where the network will be loaded
        from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()
