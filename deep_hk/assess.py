"""Functions to assess the accuracy of a previously trained model."""

from deep_hk.wave_function import WaveFunction
import torch
import numpy as np
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

def assess_predicted_energies_from_wf(
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
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
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
  print('Total loss in energy: {:.8f}\n'.format(loss))

def assess_predicted_energies_from_coeffs(
    net,
    data,
    criterion,
    device=torch.device('cpu')):
  """For a network that predicts individual wave function
     coefficients as its output, this function uses this output to
     calculate the variational energy estimator, and compares these
     predicted energies to exact values.

  Args
  ----
  net : network object
    A network which outputs wave function coefficients.
  data : Data object
    The data that will be used in the comparison.
  criterion : torch criterion object
    A loss function object, to compare the predicted and exact energies.
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
  """
  system = data.system

  e_predicted = torch.zeros(data.ndata)
  e_target = torch.FloatTensor(data.energies)

  if 'potential' in data.input_type:
    system.generate_configs()

  for ind, potential in enumerate(data.potentials):

    inp = torch.zeros(system.ndets, data.ninput, dtype=torch.float)

    if 'density' in data.input_type or '1-rdm' in data.input_type:
      # Find exact wave function
      system.add_potential_to_hamil(potential)
      wf_exact = WaveFunction(
          nsites=system.nsites,
          nspin=system.nspin,
          dets=system.dets)

      wf_exact.solve_eigenvalue(system.hamil)

    if 'potential' in data.input_type:
      inp_array = torch.from_numpy(potential)
      inp_length = system.nsites
    if 'density' in data.input_type:
      wf_exact.calc_gs_density()
      inp_array = torch.from_numpy(wf_exact.density_gs)
      inp_length = system.nsites
    if '1-rdm' in data.input_type:
      wf_exact.calc_rdm1_gs()
      inp_array = torch.from_numpy(wf_exact.rdm1_gs.flatten())
      inp_length = system.nsites**2

    for i in range(system.ndets):
      inp[i,0:inp_length] = inp_array

      if 'config' in data.input_type:
        config = system.configs[i]
        inp[i,inp_length:] = torch.from_numpy(config)
      elif 'occ_str' in data.input_type:
        inp[i,inp_length:] = torch.from_numpy(np.asarray(system.dets[i]))
      elif 'det_ind' in data.input_type:
        inp[i,inp_length:] = i

    inputs = inp.to(device)
    wf_predicted = net(inputs)
    # Convert to 1D array
    wf_predicted = wf_predicted[:,0]

    wf_numpy = wf_predicted.detach().numpy()
    energy = system.calc_energy(wf_numpy, potential)

    e_predicted[ind] = energy

  for i, energy_predicted, energy_exact in zip(count(), e_predicted,
      data.energies):

    print('{:15d}   {: .8e}   {: .8e}'.format(
        i,
        energy_predicted,
        energy_exact,
    ))

  loss = criterion(e_predicted, e_target)
  print('Total loss in energy: {:.8f}\n'.format(loss))

def calc_infidelities_from_coeffs(
    net,
    data,
    device=torch.device('cpu')):
  """For a network that predicts individual wave function
     coefficients as its output, this function uses this output to
     calculate the infidelity of the wave function relative to
     the exact result.

  Args
  ----
  net : network object
    A network which outputs wave function coefficients.
  data : Data object
    The data that will be used in the comparison.
  device : torch.device object
    Specifies whether we are using a CPU or GPU for training.
  """
  system = data.system

  infs = np.zeros(data.ndata)

  if 'potential' in data.input_type:
    system.generate_configs()

  for ind, potential in enumerate(data.potentials):

    # Find exact wave function
    system.add_potential_to_hamil(potential)

    wf_exact = WaveFunction(
        nsites=system.nsites,
        nspin=system.nspin,
        dets=system.dets)

    wf_exact.solve_eigenvalue(system.hamil)

    # Find the predicted wave function
    inp = torch.zeros(system.ndets, data.ninput, dtype=torch.float)

    # Generate the array to be input into the network
    if 'potential' in data.input_type:
      inp_array = torch.from_numpy(potential)
      inp_length = system.nsites
    if 'density' in data.input_type:
      wf_exact.calc_gs_density()
      inp_array = torch.from_numpy(wf_exact.density_gs)
      inp_length = system.nsites
    if '1-rdm' in data.input_type:
      wf_exact.calc_rdm1_gs()
      inp_array = torch.from_numpy(wf_exact.rdm1_gs.flatten())
      inp_length = system.nsites**2

    for i in range(system.ndets):
      inp[i,0:inp_length] = inp_array

      if 'config' in data.input_type:
        config = system.configs[i]
        inp[i,inp_length:] = torch.from_numpy(config)
      elif 'occ_str' in data.input_type:
        inp[i,inp_length:] = torch.from_numpy(np.asarray(system.dets[i]))
      elif 'det_ind' in data.input_type:
        inp[i,inp_length:] = i

    inputs = inp.to(device)
    wf_predicted = net(inputs)
    # Convert to 1D array
    wf_predicted = wf_predicted[:,0]

    wf_pred = wf_predicted.detach().numpy()
    wf_norm = np.linalg.norm(wf_pred, ord=2)
    wf_pred /= wf_norm

    infs[ind] = 1 - np.abs(np.dot(wf_pred, wf_exact.coeffs[:,0]))

  for i, inf in zip(count(), infs):

    print('{:15d}   {: .8e}'.format(
        i,
        inf,
    ))

  mean_inf = np.mean(infs)
  print('Mean infidelity: {:.8f}\n'.format(mean_inf))
