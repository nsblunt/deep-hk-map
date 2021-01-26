"""Utilities for defining and creating network models."""

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

    Args
    ----
    size_in : int
      The number of inputs to the layer.
    size_out : int
      The number of outputs from the layer.

    Other attributes
    ----------------
    weights : torch tensor of size (size_out, size_in)
      The parameters for the linear transformation of the input.
    bias : torch tensor of dim (noutput)
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

    Args
    ----
    x : torch tensor of dim (size_in)
      The data passed through the layer.
    """
    w_times_x = torch.mm(x, self.weights.t())
    with_bias = torch.add(w_times_x, self.bias)
    out = F.normalize(
        with_bias.view(with_bias.size(0), with_bias.size(1)),
        p=2,
        dim=1)
    return out


class SpatialPyramidPooling(nn.Module):
  """Layer to perform spatial pyramid max pooling."""

  def __init__(self, ndivisions):
    """Initialise the layer object.

    Args
    ----
    ndivisions : int
      The number of times to divide up each feature map, with max
      pooling applied to each divided section of the map.

    """
    super().__init__()
    self.ndivisions = ndivisions

  def get_output_size(self, nchannels):
    """Calculate how many output values there will be from this layer."""

    noutput = 0
    for i in range(self.ndivisions):
      noutput += nchannels * int(math.pow(2,i))

    return noutput

  def forward(self, x):
    """Apply the pooling to the input tensor, x.

    Args
    ----
    x : torch tensor of dim (batch_size, num_channels, num_features)
      The data passed through the layer.
    """

    if self.ndivisions > 0:
      nchannels = x.size()[1]
      nfeatures = x.size()[2]

      # Start with the kernal size equal to the size of the featre map.
      # This first pooling will then be a global pooling operation.
      pool_kernel_size = nfeatures

      max_kernel_fraction = int(math.pow(2, self.ndivisions-1))
      if not nfeatures % max_kernel_fraction == 0:
        raise AssertionError('Pyramid pooling with this value of ndivisions is '
                             'not consistent with the size of the feature map.')

      # Will hold the output to be returned.
      x_final = None

      # Loop over all divisions of the feature maps.
      for i in range(self.ndivisions):
        pool_layer = nn.MaxPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_kernel_size)
        x_pooled = pool_layer(x)

        noutput_pool = nchannels * int(math.pow(2,i))

        # Merge the output channels together.
        x_joined = x_pooled.view(-1, noutput_pool)

        # Merge the output with that of previous pooling operations,
        # if there have been any yet.
        if x_final is None:
          x_final = x_joined
        else:
          x_final = torch.cat((x_joined, x_final), 1)

        # Half the size of the kernel to be applied next time.
        pool_kernel_size = pool_kernel_size // 2

    return x_final


def create_linear_layers(ninput,
                         num_hidden,
                         noutput,
                         wf_output=False):
  """Create a list of Torch linear layer objects.

  Args
  ----
  ninput : int
    The number of values passed into the input of the network.
  num_hidden: list of int
    A list specifying the number of units for each hidden layer.
  noutput : int
    The number of values passed out of the network.
  wf_output : bool
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


class LinearNet(nn.Module):
  """A neural network of linear (affine) layers."""

  def __init__(self, layers_list, activation_fn):
    """Initialises the network layers.

    Args
    ----
    layers_list : list
      A list of the layers to be applied between nonlinearities,
      including the input and output layers. This should be created
      using the create_linear_layers function.
    activation_fn : string
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

    Args
    ----
    x : torch tensor
      The batch of input data to be passed through the network.
    """
    for layer in self.layers[:-1]:
      x = self.activation_fn(layer(x))
    # output layer
    x = self.layers[-1](x)
    return x

  def save(self, path):
    """Save the net state to a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be saved.
    """
    torch.save(self.state_dict(), path)

  def load(self, path):
    """Load the network state from a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be loaded
     from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()


def create_conv1d_layers(num_in_channels,
                         num_out_channels,
                         kernel_size,
                         ninput=None,
                         noutput=None,
                         maxpool_final=False):
  """Create a list of Torch conv1d layer objects. The final layer is
     fully connected, outputting noutput values.

  Args
  ----
  num_in_channels : int
    The number of channels for the input data.
  num_out_channels : list of int
    A list specifying the number of output channels for each
    convolutional layer. The length of this list is also used to
    specify the number of such layers in total.
  kernel_size : int
    The size of the kernel applied in convolutional layers.
  ninput : int
    The number of values passed into the input of the network. This is
    not used if maxpool_final=True, in which case can use ninput=None.
  noutput : int
    The number of values passed out of the network.
  maxpool_final : bool
    If true then we assume a max pooling layer will be applied to each
    output channel from the final convolutional hidden layer.
    This affects the size of the input to the fully connected layer.
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
  if maxpool_final:
    layers_list.append( nn.Linear(
        in_features=num_out_channels[-1],
        out_features=noutput)
    )
  else:
    layers_list.append( nn.Linear(
        in_features=ninput*num_out_channels[-1],
        out_features=noutput)
    )

  return layers_list


class ConvNet(nn.Module):
  """A network of convolutional layers. The final layer is a fully
     connected linear layer.
  """

  def __init__(
      self,
      layers_list,
      ninput=None,
      activation_fn='relu',
      maxpool_final=False):
    """Initialises the network layers.

    Args
    ----
    layers_list : list
      A list of the layers to be applied between nonlinearities,
      including the input and output layers. This should be created
      using the create_conv1d_layers function.
    ninput : int
      The number of features passed into the net. Not used if
      maxpool_final=True, in which case one can set ninput=None.
    activation_fn : string
      String representing the activation function, which is used to
      select a torch function below.
    maxpool_final : bool
      If true then apply a max pooling layer to each output channel
      from the final convolutional hidden layer.
    """
    super(ConvNet, self).__init__()

    self.layers = nn.ModuleList(layers_list)
    self.activation_fn = activation_fn
    self.maxpool_final = maxpool_final

    # Number of ouput channels from the final convolutional layer.
    out_channels_final = self.layers[-2].out_channels
    # Number of features input to the final layer.
    if self.maxpool_final:
      self.nfeatures_final = out_channels_final
    else:
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

    Args
    ----
    x : torch tensor
      The batch of input data to be passed through the network.
    """
    # Need to add a dimension, recognised as the single input channel.
    x = x[:, None, :]

    # Apply the convolutional layers.
    for layer in self.layers[:-1]:
      x = self.activation_fn(layer(x))

    # Apply max pool to each channel.
    if self.maxpool_final:
      x, _ = torch.max(x, dim=2)

    # Merge the output channels together.
    x = x.view(-1, self.nfeatures_final)

    # Fully connected output layer.
    x = self.layers[-1](x)
    return x

  def save(self, path):
    """Save the net state to a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be saved.
    """
    torch.save(self.state_dict(), path)

  def load(self, path):
    """Load the network state from a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be loaded
      from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()


class ResConvNet(nn.Module):
  """A network of convolutional layers with optional skip connections,
     as in residual neural networks.
  """

  def __init__(
      self,
      nchannels,
      nblocks,
      noutput,
      ndivisions=1,
      with_skip=True,
      activation_fn='relu'):
    """Initialises the network layers.

    Args
    ----
    nchannels : int
      The number of input and output channels to and from each
      convolutiona layer (except for the first layer, which has a
      single input channel).
    nblocks : int
      The number of blocks applied. Each block consists of two
      convolutional layers, with the option of a skip connection before
      the second activation function is applied.
    noutput : int
      The number of values passed out of the network.
    with_skip : bool
      If true, then use skip connections, i.e. use a ResNet.
    activation_fn : string
      String representing the activation function, which is used to
      select a torch function below.
    """
    super(ResConvNet, self).__init__()

    self.nchannels = nchannels
    self.nblocks = nblocks
    self.noutput = noutput
    self.ndivisions = ndivisions
    self.with_skip = with_skip

    self.kernel_size = 3
    self.padding = int((self.kernel_size-1)/2)

    # input layer:
    self.input_layer = nn.Conv1d(
        in_channels=1,
        out_channels=self.nchannels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        padding_mode='circular')

    # hidden layers:
    self.hidden_layer = nn.Conv1d(
        in_channels=self.nchannels,
        out_channels=self.nchannels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        padding_mode='circular')

    self.pooling_layer = SpatialPyramidPooling(self.ndivisions)

    self.nfeatures_final = self.pooling_layer.get_output_size(self.nchannels)

    # final layer:
    self.fc_layer = nn.Linear(
        in_features=self.nfeatures_final,
        out_features=noutput)

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

    Args
    ----
    x : torch tensor
      The batch of input data to be passed through the network.
    """
    # Need to add a dimension, recognised as the single input channel.
    x = x[:, None, :]

    # Each block applies two convolutional layers, with a skip
    # connection optionally applied before the second activation
    # function is applied.

    # Apply the convolutional layers.
    # Input block.
    if self.with_skip:
      inp = x
    x = self.activation_fn(self.input_layer(x))
    x = self.hidden_layer(x)
    if self.with_skip:
      x = x + inp
    x = self.activation_fn(x)

    # Hidden blocks.
    for block in range(1, self.nblocks):
      if self.with_skip:
        inp = x
      x = self.activation_fn(self.hidden_layer(x))
      x = self.hidden_layer(x)
      if self.with_skip:
        x = x + inp
      x = self.activation_fn(x)

    # Merge to a fixed size output using spatial pyramid pooling
    x = self.pooling_layer(x)

    # Fully connected output layer.
    x = self.fc_layer(x)
    return x

  def save(self, path):
    """Save the net state to a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be saved.
    """
    torch.save(self.state_dict(), path)

  def load(self, path):
    """Load the network state from a file.

    Args
    ----
    path : string
      The path and name of the file where the network will be loaded
      from.
    """
    self.load_state_dict(torch.load(path))
    self.eval()
