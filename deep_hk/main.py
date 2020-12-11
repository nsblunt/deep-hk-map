from absl import app
from absl import flags
from data import Data
from hamiltonian import Hubbard, SpinlessHubbard
import json
import networks
import train

import torch
import torch.nn as nn
import torch.optim as optim

FLAGS = flags.FLAGS
flag_dict_init = FLAGS.flag_values_dict()

# Define the system.
flags.DEFINE_enum(
    'system', 'spinless_hubbard', ['spinless_hubbard', 'hubbard'],
    'Define the type of system being studied.')
flags.DEFINE_integer('nsites', 4, 'Number of lattice sites.')
flags.DEFINE_integer('nparticles', 2, 'Number of particles.')
flags.DEFINE_boolean('fixed_Ms', True, 'If true then use a fixed-Ms '
    'sector. This is not used in the case of spinless systems.')
flags.DEFINE_integer('Ms', 0, 'Total spin of the system (in units of '
    'electron spin). This is not used in the case of spinless systems.')
flags.DEFINE_float('U', 1.0, 'Parameter U in the spinless Hubbard model.')
flags.DEFINE_float('t', 1.0, 'Parameter t in the spinless Hubbard model.')
flags.DEFINE_float('mu', 0.0, 'Chemical potential parameter.')
flags.DEFINE_float('max_potential', 0.5, 'The maximum absolute value of '
    'random potentials applied on any given site.')
flags.DEFINE_integer('seed', 7, 'Seed for the random number generator.')
flags.DEFINE_boolean(
    'fixed_nparticles', True, 'True is using a fixed number of particles. '
    'False if using all particle sectors.')
flags.DEFINE_boolean('const_potential_sum', False, 'If true, then the sum '
    'of the applied potential is a constant value (potential_sum_val).')
flags.DEFINE_float('potential_sum_val', 0.0, 'If const_potential_sum is '
    'true, then this is the value of potential summed over all sites.')

# Define the parameters for data (training, validation, test).
flags.DEFINE_integer('ntrain', 12800, 'Number of training samples to '
    'generate.')
flags.DEFINE_integer('nvalidation', 0, 'Number of validation samples to '
    'generate.')
flags.DEFINE_integer('ntest', 100, 'Number of test samples to generate.')
flags.DEFINE_boolean('load_train_data_csv', False, 'If true, read the '
    'training data from a CSV file, instead of generating it.')
flags.DEFINE_boolean('load_valid_data_csv', False, 'If true, read the '
    'validation data from a CSV file, instead of generating it.')
flags.DEFINE_boolean('load_test_data_csv', False, 'If true, read the test '
    'data from a CSV file, instead of generating it.')
flags.DEFINE_boolean('save_train_data_csv', True, 'If true, save the '
    'generated training data to a CSV file.')
flags.DEFINE_boolean('save_valid_data_csv', True, 'If true, save the '
    'generated validation data to a CSV file.')
flags.DEFINE_boolean('save_test_data_csv', True, 'If true, save the '
    'generated test data to a CSV file.')

# Training parameters.
flags.DEFINE_integer('batch_size', 128, 'Number of samples per training '
    'batch.')
flags.DEFINE_integer('nepochs', 100, 'Number of training epochs to perform.')
flags.DEFINE_float('lr', 0.001, 'The learning rate for the optimizer.')

# Defining the network (including input/output objects).
flags.DEFINE_enum('net_type', 'linear', ['linear', 'conv'], 'Specify which '
    'network type to use.')
flags.DEFINE_enum(
    'input_type', 'potential', ['potential', 'density', '1-rdm'], 'Specify '
    'which object we pass into the network input.')
flags.DEFINE_enum('output_type', 'energy',
    ['energy', 'wave_function', 'potential', 'density', '1-rdm', 'corr_fn'],
    'Specify which object should be output by the network.')
flags.DEFINE_enum('activation_fn', 'relu',
    ['relu', 'elu', 'sigmoid', 'tanh'],
    'Define the activation function used.')
# Parameters to define linear networks.
flags.DEFINE_list('layer_widths', [100], 'The number of hidden units in '
    'each layer of the network, input as comma-separated values.')
# Parameters to define convolutional networks.
flags.DEFINE_integer('kernel_size', 3, 'The size of the kernel.')
flags.DEFINE_list('output_channels', [5], 'A list specifying the number '
    'of output channels in each convolutional layer. The length of this '
    'list determines the number of such layers.')

# Parameters regarding saving/loading the trained network.
flags.DEFINE_boolean('save_final_net', True, 'If True, then save the final '
    'trained network to a file.')
flags.DEFINE_string('save_final_path', './network.pt', 'Path and name '
    'for the file used to print the final network parameters.')
flags.DEFINE_boolean('save_net', False, 'If True, then save the network '
    'at regular intervals during training.')
flags.DEFINE_string('save_root', './network', 'Path and root for the '
    'files used to print the network parameters, at regular intervals '
    'during training.')
flags.DEFINE_integer('save_net_every', 100, 'The interval at which to '
    'save the network parameters to a file.')
flags.DEFINE_boolean('load_net', False, 'If True, then begin by loading the '
    'network from a file.')
flags.DEFINE_string('load_path', './network.pt', 'Path and name for the '
    'file used to print the final network parameters.')

# Post-training checks.
flags.DEFINE_boolean('assess_energy_from_wf', 'False', 'If predicting '
    'a wave function as output, then calculate and print the associated '
    'energies for the test data.')

def main(argv):
  del argv

  flag_dict = FLAGS.flag_values_dict()
  # Dictionary of input variables (without predefined entries):
  flag_dict_input = {k: v for k, v in flag_dict.items() if k not in flag_dict_init}
  print(json.dumps(flag_dict_input, indent=1), '\n', flush=True)

  torch.manual_seed(FLAGS.seed)

  if FLAGS.system == 'spinless_hubbard':
    system = SpinlessHubbard(
        U=FLAGS.U,
        t=FLAGS.t,
        mu=FLAGS.mu,
        max_V=FLAGS.max_potential,
        nsites=FLAGS.nsites,
        fixed_nparticles=FLAGS.fixed_nparticles,
        nparticles=FLAGS.nparticles,
        seed=FLAGS.seed)
  elif FLAGS.system == 'hubbard':
    system = Hubbard(
        U=FLAGS.U,
        t=FLAGS.t,
        mu=FLAGS.mu,
        max_V=FLAGS.max_potential,
        nsites=FLAGS.nsites,
        fixed_nparticles=FLAGS.fixed_nparticles,
        nparticles=FLAGS.nparticles,
        fixed_Ms=FLAGS.fixed_Ms,
        Ms=FLAGS.Ms,
        seed=FLAGS.seed)
  system.construct()

  # -- training data ------
  data_train = Data(
      system=system,
      ndata=FLAGS.ntrain,
      input_type=FLAGS.input_type,
      output_type=FLAGS.output_type,
      load=FLAGS.load_train_data_csv,
      save=FLAGS.save_train_data_csv,
      path='data_train.csv',
      const_potential_sum=FLAGS.const_potential_sum,
      potential_sum_val=FLAGS.potential_sum_val)

  # -- validation data ------
  if FLAGS.nvalidation > 0:
    data_valid = Data(
        system=system,
        ndata=FLAGS.nvalidation,
        input_type=FLAGS.input_type,
        output_type=FLAGS.output_type,
        load=FLAGS.load_valid_data_csv,
        save=FLAGS.save_valid_data_csv,
        path='data_valid.csv',
        const_potential_sum=FLAGS.const_potential_sum,
        potential_sum_val=FLAGS.potential_sum_val)
  else:
    data_valid = None

  # -- test data ------
  data_test = Data(
      system=system,
      ndata=FLAGS.ntest,
      input_type=FLAGS.input_type,
      output_type=FLAGS.output_type,
      load=FLAGS.load_test_data_csv,
      save=FLAGS.save_test_data_csv,
      path='data_test.csv',
      const_potential_sum=FLAGS.const_potential_sum,
      potential_sum_val=FLAGS.potential_sum_val)

  ninput = data_train.ninput
  noutput = data_train.noutput

  # Linear networks
  if FLAGS.net_type == 'linear':
    layer_widths = [int(s) for s in FLAGS.layer_widths]
    layers_list = networks.create_linear_layers(
        ninput,
        layer_widths,
        noutput,
        wf_output = FLAGS.output_type == 'wave_function')
    net = networks.LinearNet(
        layers_list,
        FLAGS.activation_fn)
  # Convolutional networks
  elif FLAGS.net_type == 'conv':
    output_channels = [int(s) for s in FLAGS.output_channels]
    layers_list = networks.create_conv1d_layers(
        ninput,
        noutput,
        num_in_channels=1,
        num_out_channels=output_channels,
        kernel_size=FLAGS.kernel_size)
    net = networks.ConvNet(
        layers_list,
        ninput,
        FLAGS.activation_fn)

  if FLAGS.load_net:
    net.load(FLAGS.load_path)

  if FLAGS.output_type == 'wave_function':
    criterion = train.Infidelity()
  else:
    criterion = nn.L1Loss()

  optimizer = optim.Adam(
    net.parameters(),
    lr=FLAGS.lr,
    amsgrad=False)

  train.train(
      net=net,
      data_train=data_train,
      data_validation=data_valid,
      data_test=data_test,
      criterion=criterion,
      optimizer=optimizer,
      nepochs=FLAGS.nepochs,
      batch_size=FLAGS.batch_size,
      save_net=FLAGS.save_net,
      save_root=FLAGS.save_root,
      save_net_every=FLAGS.save_net_every)

  if FLAGS.save_final_net:
    net.save(FLAGS.save_final_path)

  train.print_net_accuracy(
      net,
      data_train,
      data_test,
      criterion)

  data_label = 0
  train.print_data_comparison(
      net,
      data_test,
      data_label)

  if FLAGS.assess_energy_from_wf and FLAGS.output_type == 'wave_function':
    train.assess_predicted_energies(
        net=net,
        data=data_test,
        criterion=nn.L1Loss())

if __name__ == '__main__':
  app.run(main)
