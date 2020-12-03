import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import count


class Infidelity(nn.Module):
  r"""A loss function that uses the infidelity of a wave function,
    defined as:

    I(\psi_p, psi_e) = 1 - |\langle \psi_p | \psi_e \rangle|.

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
          data_test,
          criterion,
          optimizer,
          nepochs,
          batch_size,
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
  data_test : Data object
    The test data.
  criterion : torch criterion object
    Used to measure the loss function between predicted and targeted
    data.
  optimizer : torch optimizer object
    Implements the optimization algorithm, such as Adam.
  nepochs : int
    The number of epochs to perform.
  batch_size : int
    The number of data points passed in each batch.
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
    print('# 1. Epoch' + 2*' ' + '2. Train. Loss')
  else:
    print('# 1. Epoch' + 2*' ' + '2. Train. Loss' + 2*' ' + '3. Valid. loss')

  data_loader = DataLoader(
      data_train,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0)

  # Train the network.
  for epoch in range(nepochs):
    total_loss = 0.0
    nbatches = 0

    for batch_inputs, batch_labels in data_loader:
      optimizer.zero_grad()
      batch_outputs = net(batch_inputs)
      loss = criterion(batch_outputs, batch_labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      nbatches += 1
    av_loss = total_loss/nbatches

    if data_validation is None:
      print('{:10d}    {:12.8f}'.format(epoch, av_loss), flush=True)
    else:
      # Calculate the loss for validation data.
      valid_outputs = net(data_validation.inputs)
      valid_loss = criterion(valid_outputs, data_validation.labels)
      print('{:10d}    {:12.8f}    {:12.8f}'.format(
          epoch,
          av_loss,
          valid_loss
      ), flush=True)

    if save_net:
      if epoch % save_net_every == save_net_every-1:
        nepochs_done = epoch+1
        filename = save_root + '_' + str(nepochs_done) + '.pt'
        net.save(filename)

  print(flush=True)

def print_net_accuracy(net, data_train, data_test, criterion):
  """Calculate and print the loss for both training and test data.
     Also, calculate the norms and print these together with the
     training and test data values for comparison.

  Args
  ----
  net : network object
    The neural network to be used in the comparison.
  data_train : Data object
    The training data.
  data_test : Data object
    The test data.
  criterion : Torch criterion object
    Used to measure the loss function between the predicted and
    targeted data.
  """
  # Apply the network to the training data.
  outputs_train = net(data_train.inputs)
  train_loss = criterion(outputs_train, data_train.labels)
  print('Training loss: {:.8f}'.format(train_loss))

  # Apply the network to the test data.
  outputs_test = net(data_test.inputs)
  test_loss = criterion(outputs_test, data_test.labels)
  print('Test loss: {:.8f}\n'.format(test_loss))

  output_norms = [torch.norm(row) for row in outputs_test]
  labels_norms = [torch.norm(row) for row in data_test.labels]

  # Print the exact labels against the predicted labels for the test
  # data. This is only done for the first element of each data point.
  # Also print the norms of each data point.
  print('# 1. Data label' + 10*' ' + '2. Exact' + 6*' ' + '3. Predicted' + 
         5*' ' + '4. Exact norm' + 2*' ' + '5. Predicted norm')
  for i, target, predicted, target_norm, predicted_norm in zip(count(),
      data_test.labels[:,0], outputs_test[:,0], labels_norms, output_norms):
    print('{:15d}   {: .8e}   {: .8e}   {: .8e}    {: .8e}'.format(
        i,
        target,
        predicted,
        float(target_norm),
        float(predicted_norm)
    ))

def print_data_comparison(net, data, data_label):
  """For the requested data point, print the predicted and target values
     for each output unit. Also, print the potential of this data point.

  Args
  ----
  net : network object
    The neural network to be used in the comparison.
  data : Data object
    Object holding a set of data points.
  data_label :
    The index of the data point to be considered, in the data arrays.
  """
  print('\nComparing the exact output to the predicted output for a '
        'single test example...')

  # Print the potential for this data point, if it is stored.
  if data.potentials is not None:
    print('\nPotential used:')
    print('# 1. Site' + 6*' ' + '2. Potential')
    for i, potential in zip(count(), data.potentials[data_label]):
      print('{:9d}   {: .8e}'.format(
          i,
          potential,
      ))

  # Calculate the predicted values.
  outputs = net(data.inputs)

  # Print the predicted values against the target values, for each
  # output unit.
  print('\n# 1. Det. label' + 10*' ' + '2. Exact' + 6*' ' + '3. Predicted')
  for i in range(data.noutput):
    print('{:15d}   {: .8e}   {: .8e}'.format(
        i,
        float(data.labels[data_label][i]),
        float(outputs[data_label][i])
    ))

def assess_predicted_energies(net, data, criterion):
  """For a network that predicts the wave function as its output, this
     function uses this output to calculate the variational energy
     estimator, and compares these predicted energies to exact values.

  Args
  ----
  net : network object
    A network which outputs wave function coefficients.
  data : Data object
    The data that will be used in the comparison.
  criterion : torch criterion object
    A loss function object, to compare the predicted and exact energies.
  """
  wf_predicted = net(data.inputs)
  e_predicted = torch.zeros(data.ndata)

  e_target = torch.FloatTensor(data.energies)

  print('\n# 1. Data label' + 10*' ' + '2. Exact' + 6*' ' + '3. Predicted')

  for i, wf, potential, energy_exact in zip(count(), wf_predicted,
      data.potentials, data.energies):
    # Calculate the energy for the predicted wave function.
    wf_numpy = wf.detach().numpy()
    energy = data.system.calc_energy(wf_numpy, potential)
    e_predicted[i] = energy

    print('{:15d}   {: .8e}   {: .8e}'.format(
        i,
        energy,
        energy_exact,
    ))

  loss = criterion(e_predicted, e_target)
  print('Total loss: {:.8f}\n'.format(loss))
