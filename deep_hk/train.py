"""Functions to train models of maps between lattice model properties."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

class Infidelity(nn.Module):
  r"""A loss function that uses the infidelity of a wave function,
    defined as:

    I(\psi_p, \psi_e) = 1 - |\langle \psi_p | \psi_e \rangle|.

    In other words, unity minus the overlap of the predicted and exact
    wave functions.
  """

  def __init__(self):
    """Initialise the object."""
    super(Infidelity, self).__init__()

  def forward(self, outputs, labels):
    """Calculate the infidelity using the provided outputs and labels.

    Args
    ----
    outputs : torch tensor
      The batch of output data.
    labels : torch tensor
      The batch of labels being targeted.
    """
    dot_products = torch.sum(outputs * labels, dim=1)
    loss = 1 - torch.mean(torch.abs(dot_products))
    return loss

def train(net,
          data_train,
          data_validation,
          criterion,
          optimizer,
          nepochs,
          batch_size,
          device=torch.device('cpu'),
          save_net=False,
          save_root='./network',
          save_net_every=100):
  """Train the network.

  Args
  ----
  net : network object
    The neural network to be trained.
  data_train : Data object
    The training data.
  data_validation : Data object
    The validation data.
  criterion : torch criterion object
    Used to measure the loss function between predicted and targeted
    data.
  optimizer : torch optimizer object
    Implements the optimization algorithm, such as Adam.
  nepochs : int
    The number of epochs to perform.
  batch_size : int
    The number of data points passed in each batch.
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
  save_net : bool
    If True, save the network state to a file at regular intervals.
  save_root : string
    The path and root of the filenames where networks will be saved, if
    save_net is True.
  save_net_every : int
    The frequency (in epochs) at which the network will be saved to a
    file, if save_net is True.
  """
  # Print the header.
  if data_validation is None:
    print('# 1. Epoch' + 2*' ' + '2. Train. Loss' + 3*' ' +
          '3. Epoch time')
  else:
    print('# 1. Epoch' + 2*' ' + '2. Train. Loss' + 2*' ' +
          '3. Valid. loss' + 3*' ' + '4. Epoch time')

  data_loader = DataLoader(
      data_train,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0)

  # Train the network.
  for epoch in range(nepochs):
    start_time = time.time()
    total_loss = 0.0
    nbatches = 0

    for batch_inputs, batch_labels in data_loader:
      optimizer.zero_grad()
      inputs = batch_inputs.to(device)
      labels = batch_labels.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      nbatches += 1
    av_loss = total_loss/nbatches

    if data_validation is None:
      end_time = time.time()
      epoch_time = end_time - start_time
      print('{:10d}    {:12.8f}    {:12.8f}'.format(
          epoch,
          av_loss,
          epoch_time
      ), flush=True)
    else:
      # Calculate the loss for validation data.
      valid_inputs = data_validation.inputs.to(device)
      valid_labels = data_validation.labels.to(device)
      valid_outputs = net(valid_inputs)
      valid_loss = criterion(valid_outputs, valid_labels)
      end_time = time.time()
      epoch_time = end_time - start_time
      print('{:10d}    {:12.8f}    {:12.8f}    {:12.8f}'.format(
          epoch,
          av_loss,
          valid_loss,
          epoch_time,
      ), flush=True)

    if save_net:
      if epoch % save_net_every == save_net_every-1:
        nepochs_done = epoch+1
        filename = save_root + '_' + str(nepochs_done) + '.pt'
        net.save(filename)

  print(flush=True)
