"""Functions to assess the accuracy of a previously trained model."""

import torch
from itertools import count

def print_net_accuracy(
    net,
    data_train,
    data_test,
    criterion,
    device=torch.device('cpu')):
  """Calculate and print the loss for both training and test data.

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
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
  """
  # Apply the network to the training data.
  inputs_train = data_train.inputs.to(device)
  labels_train = data_train.labels.to(device)
  outputs_train = net(inputs_train)
  train_loss = criterion(outputs_train, labels_train)
  print('Training loss: {:.8f}'.format(train_loss))

  # Apply the network to the test data.
  inputs_test = data_test.inputs.to(device)
  labels_test = data_test.labels.to(device)
  outputs_test = net(inputs_test)
  test_loss = criterion(outputs_test, labels_test)
  print('Test loss: {:.8f}\n'.format(test_loss))

def print_exact_vs_predicted(
    net,
    data_test,
    device=torch.device('cpu')):
  """Calculate the predicted output for the test data, and compare
     it to the true labels. This is done by calculating the norms
     of the predicted labels and true labels, and also by comparing
     the first element of each data point. These values are printed.

  Args
  ----
  net : network object
    The neural network to be used in the comparison.
  data_test : Data object
    The test data.
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
  """

  # Apply the network to the test data.
  inputs_test = data_test.inputs.to(device)
  labels_test = data_test.labels.to(device)
  outputs_test = net(inputs_test)

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

def print_data_comparison(
    net,
    data,
    data_label,
    device=torch.device('cpu')):
  """For the requested data point, print the predicted and target values
     for each output unit. Also, print the potential of this data point.

  Args
  ----
  net : network object
    The neural network to be used in the comparison.
  data : Data object
    Object holding a set of data points.
  data_label : int
    The index of the data point to be considered, in the data arrays.
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
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
  inputs = data.inputs.to(device)
  outputs = net(inputs)

  # Print the predicted values against the target values, for each
  # output unit.
  print('\n# 1. Det. label' + 10*' ' + '2. Exact' + 6*' ' + '3. Predicted')
  for i in range(data.noutput):
    print('{:15d}   {: .8e}   {: .8e}'.format(
        i,
        float(data.labels[data_label][i]),
        float(outputs[data_label][i])
    ))

def assess_predicted_energies(
    net,
    data,
    criterion,
    device=torch.device('cpu')):
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
  inputs = data.inputs.to(device)
  wf_predicted = net(inputs)
  e_predicted = torch.zeros(data.ndata_tot)

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
