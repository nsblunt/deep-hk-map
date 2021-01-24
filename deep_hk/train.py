"""Functions to train models of maps between lattice model properties."""

from deep_hk.data import MultipleDatasets
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def criterion_list(criterion, outputs, labels):
  ndata = 0
  loss = torch.FloatTensor([0.0])
  for output, label in zip(outputs, labels):
    nbatch = output.size()[0]
    #loss += nbatch*torch.mean(torch.abs(output - label))
    loss += nbatch * criterion(output, label)
    ndata += nbatch
  loss /= ndata
  return loss

def collate_as_list_of_tensors(batch):
  input_sizes = []
  # Indexed by the input size:
  inputs = {}
  labels = {}

  for input, label in batch:
    input_size = len(input)
    if input_size in input_sizes:
      inputs[input_size].append(input)
      labels[input_size].append(label)
    if input_size not in input_sizes:
      input_sizes.append(input_size)
      inputs[input_size] = [input]
      labels[input_size] = [label]

  # Merge the entries for a given input size into a 2d tensor.
  # Do this for every input size, and store the results in a list.
  inputs_list = []
  labels_list = []
  for input_size, v in inputs.items():
    nbatch = len(v)
    inputs_merged = torch.cat(v)
    inputs_merged = inputs_merged.view(nbatch, input_size)
    inputs_list.append(inputs_merged)
  for ninput, v in labels.items():
    nbatch = len(v)
    label_size = len(label)
    labels_merged = torch.cat(v)
    labels_merged = labels_merged.view(nbatch, label_size)
    labels_list.append(labels_merged)

  return inputs_list, labels_list

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
  data_train : Data object or list of Data objects
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

  # Create the DataLoader. If multiple data sets are being used then
  # this has to be treated separately.
  if isinstance(data_train, tuple):
    data_train_all = MultipleDatasets(data_train)
    input_sizes = [dat.ninput for dat in data_train]
    output_sizes = [dat.noutput for dat in data_train]
    # Are all input/output sizes the same?
    fixed_ninput = input_sizes.count(input_sizes[0]) == len(input_sizes)
    fixed_noutput = output_sizes.count(output_sizes[0]) == len(output_sizes)
  else:
    data_train_all = data_train
    fixed_ninput = True
    fixed_noutput = True

  if fixed_ninput:
    data_loader = DataLoader(
        data_train_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
  else:
    # Different sized inputs in the batch. We therefore want to collate
    # data as a list of 2d tensors, each of a given input size, rather
    # than a single 2d tensor.
    data_loader = DataLoader(
        data_train_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_as_list_of_tensors)

  # Train the network.
  for epoch in range(nepochs):
    start_time = time.time()
    total_loss = 0.0
    nbatches = 0

    for batch_inputs, batch_labels in data_loader:
      optimizer.zero_grad()

      # Apply the network and calculate the loss function.
      if isinstance(batch_inputs, list):
        # Inputs and labels are a list of 2d tensors.
        outputs = []
        labels = [x.to(device) for x in batch_labels]
        for inputs in batch_inputs:
          # inputs is a 2d tensor for a single input size.
          inputs = inputs.to(device)
          outputs.append(net(inputs))
        loss = criterion_list(criterion, outputs, labels)
      else:
        # Inputs and labels are a single 2d tensor each.
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
      if not isinstance(inputs, list):
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
