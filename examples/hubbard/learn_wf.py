from deep_hk import networks, train, data, hamiltonian
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

nepochs = 300
batch_size = 128
ndata_train = 3200

torch.manual_seed(7)

system = hamiltonian.Hubbard(
    U=4.0,
    t=1.0,
    mu=0.0,
    max_V=0.5,
    nsites=8,
    fixed_nparticles=True,
    nparticles=8,
    fixed_Ms=True,
    Ms=0,
    seed=8)

system.construct()

print('Generating the training data. May take several minutes...', flush=True)

data_train = data.Data(
    system=system,
    ndata=ndata_train,
    input_type='density',
    output_type='wave_function',
    load=False,
    save=False,
    path='data_train.csv')

print('Training data generation done.', flush=True)

ninput = data_train.ninput
noutput = data_train.noutput

# Network definition
layers_list = networks.create_linear_layers(
    ninput,
    num_hidden=[100],
    noutput=noutput,
    wf_output=True)

# Create fully-connected network
net = networks.LinearNet(layers_list, 'relu')

criterion = train.Infidelity()

optimizer = optim.Adam(
  net.parameters(),
  lr=0.001,
  amsgrad=False)

train.train(
    net=net,
    data_train=data_train,
    data_validation=None,
    criterion=criterion,
    optimizer=optimizer,
    nepochs=nepochs,
    batch_size=batch_size,
    save_net=False)

print('Generating the test data. May take several minutes...', flush=True)

ndata_test = 1280
data_test = data.Data(
    system=system,
    ndata=ndata_test,
    input_type='density',
    output_type='wave_function',
    load=False,
    save=False,
    path='data_test.csv',)

print('Test data generation done.', flush=True)

outputs = net(data_test.inputs)

# Compare the exact and predicted components for a given data point
data_label = 0
exact = abs(data_test.labels[data_label].detach())
predicted = abs(outputs[data_label].detach())

plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-7,1])
plt.ylim([1e-7,1])
plt.plot([1e-7,1], [1e-7,1], linestyle='--')
plt.plot(exact, predicted, linestyle='None', marker='x')
plt.xlabel('Exact wave function components')
plt.ylabel('Predicted wave function components')
plt.show()

# Plot the distribution of loss values for the test data
losses = []
for i in range(ndata_test):
  loss_test = criterion(outputs[i:i+1,:], data_test.labels[i:i+1,:])
  losses.append(loss_test.detach())

bins = np.linspace(0, max(losses), 30)
plt.xlim([0, max(losses)])
plt.hist(losses, bins=bins)
plt.hist(losses, bins=bins, edgecolor='black')
plt.xlabel('Wave function infidelity')
plt.ylabel('Count')
plt.locator_params(nbins=5)
plt.show()
