from deep_hk import networks, train, data, hamiltonian
import torch
import torch.nn as nn
import torch.optim as optim

nepochs = 20
batch_size = 128
ndata_train = 12800

torch.manual_seed(7)

system = hamiltonian.Hubbard(
    U=1.0,
    t=1.0,
    mu=0.0,
    max_V=0.5,
    nsites=2,
    fixed_nparticles=True,
    nparticles=2,
    fixed_Ms=True,
    Ms=0,
    seed=8)

system.construct()

# Generate training data
data_train = data.Data(
    system=system,
    ndata=ndata_train,
    input_type='1-rdm',
    output_type='energy',
    load=False,
    save=False,
    path='data_train.csv',
    const_potential_sum=True,
    potential_sum_val=0.0)

ninput = data_train.ninput
noutput = data_train.noutput

# Network definition
layers_list = networks.create_linear_layers(
    ninput,
    num_hidden=[100],
    noutput=noutput,
    wf_output=False)

# Create fully-connected network
net = networks.LinearNet(layers_list, 'relu')

criterion = nn.L1Loss()

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
