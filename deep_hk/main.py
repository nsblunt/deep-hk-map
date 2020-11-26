from absl import app
from absl import flags
from data import Data
from system import SpinlessHubbard
from wave_function import WaveFunction
import networks
import json

import torch
import torch.nn as nn
import torch.optim as optim

FLAGS = flags.FLAGS
flag_dict_init = FLAGS.flag_values_dict()

flags.DEFINE_integer('nsites', 4, 'Number of lattice sites.')
flags.DEFINE_integer('nparticles', 2, 'Number of particles.')
flags.DEFINE_float('U', 1.0, 'Parameter U in the spinless Hubbard model.')
flags.DEFINE_float('t', 1.0, 'Parameter t in the spinless Hubbard model.')
flags.DEFINE_float('mu', 0.0, 'Chemical potential parameter.')
flags.DEFINE_float('max_potential', 2.0, 'The maximum absolute value of '
    'random potentials applied on any given site.')
flags.DEFINE_float('seed', 7, 'Seed for the random number generator.')
flags.DEFINE_boolean(
    'fixed_nparticles', True, 'True is using a fixed number of particles. '
    'False if using all particle sectors.')

flags.DEFINE_integer('ntrain', 12800, 'Number of training samples to '
                      'generate.')
flags.DEFINE_integer('ntest', 100, 'Number of test samples to generate.')
flags.DEFINE_integer('batch_size', 128, 'Number of samples per training batch.')
flags.DEFINE_boolean('load_train_data_csv', False, 'If true, read the training '
    'data from a CSV file, instead of generating it.')
flags.DEFINE_boolean('load_test_data_csv', False, 'If true, read the test '
    'data from a CSV file, instead of generating it.')
flags.DEFINE_boolean('save_train_data_csv', True, 'If true, save the generated '
    'training data to a CSV file.')
flags.DEFINE_boolean('save_test_data_csv', True, 'If true, save the generated '
    'test data to a CSV file.')

flags.DEFINE_integer('nepochs', 100, 'Number of training epochs to perform.')
flags.DEFINE_float('lr', 0.001, 'The learning rate for the optimizer.')

flags.DEFINE_enum(
    'input_type', 'potential', ['potential', 'density', '1-rdm'], 'Specify '
    'which object we pass into the network input.')
flags.DEFINE_enum('output_type', 'energy',
    ['energy', 'wave_function', 'potential', 'density', '1-rdm'],
    'Specify which object should be output by the network.')
flags.DEFINE_list('layer_widths', [100], 'The number of hidden units in '
    'each layer of the network, input as comma-separated values.')

flags.DEFINE_bool('save_final_net', True, 'If True, then save the final '
    'trained network to a file.')
flags.DEFINE_string('save_final_path', './network.pt', 'Path and name '
    'for the file used to print the final network parameters.')
flags.DEFINE_bool('save_net', False, 'If True, then save the network '
    'at regular intervals during training.')
flags.DEFINE_string('save_root', './network', 'Path and root for the '
    'files used to print the network parameters, at regular intervals '
    'during training.')
flags.DEFINE_integer('save_net_every', 100, 'The interval at which to '
    'save the network parameters to a file.')
flags.DEFINE_bool('load_net', False, 'If True, then begin by loading the '
    'network from a file.')
flags.DEFINE_string('load_path', './network.pt', 'Path and name for the '
    'file used to print the final network parameters.')

def main(argv):
  del argv

  flag_dict = FLAGS.flag_values_dict()
  # Dictionary of input variables:
  flag_dict_input = {k: v for k, v in flag_dict.items() if k not in flag_dict_init}
  print(json.dumps(flag_dict_input, indent=1), '\n', flush=True)

  sys = SpinlessHubbard(
    U=FLAGS.U,
    t=FLAGS.t,
    mu=FLAGS.mu,
    max_V=FLAGS.max_potential,
    nsites=FLAGS.nsites,
    fixed_nparticles=FLAGS.fixed_nparticles,
    nparticles=FLAGS.nparticles,
    seed=FLAGS.seed
  )
  sys.construct()

  if FLAGS.input_type == 'potential' or FLAGS.input_type == 'density':
    ninput = sys.nsites
  elif FLAGS.input_type == '1-rdm':
    ninput = sys.nsites**2

  wf_output = False
  if FLAGS.output_type == 'energy':
    noutput = 1
  elif FLAGS.output_type == 'wave_function':
    noutput = sys.ndets
    wf_output = True
  elif FLAGS.output_type == 'potential' or FLAGS.output_type == 'density':
    noutput = sys.nsites
  elif FLAGS.output_type == '1-rdm':
    noutput = sys.nsites**2

  data_train = Data(
    system=sys,
    ninput=ninput,
    noutput=noutput,
    ndata=FLAGS.ntrain,
    input_type=FLAGS.input_type,
    output_type=FLAGS.output_type
  )
  if FLAGS.load_train_data_csv:
    data_train.load_csv('data_train.csv')
  else:
    data_train.generate()

  if FLAGS.save_train_data_csv:
    data_train.save_csv('data_train.csv')

  data_test = Data(
    system=sys,
    ninput=ninput,
    noutput=noutput,
    ndata=FLAGS.ntest,
    input_type=FLAGS.input_type,
    output_type=FLAGS.output_type
  )
  if FLAGS.load_test_data_csv:
    data_test.load_csv('data_test.csv')
  else:
    data_test.generate()

  if FLAGS.save_test_data_csv:
    data_test.save_csv('data_test.csv')

  torch.manual_seed(FLAGS.seed)

  layer_widths = [int(s) for s in FLAGS.layer_widths]

  layers_list = networks.create_linear_layers(
      ninput,
      layer_widths,
      noutput,
      wave_function_output = wf_output
  )
  net = networks.LinearNet(layers_list)

  if FLAGS.load_net:
    net.load(FLAGS.load_path)

  criterion = nn.L1Loss()
  optimizer = optim.Adam(net.parameters(), lr=FLAGS.lr, amsgrad=False)

  networks.train(
    net,
    data_train,
    data_test,
    criterion,
    optimizer,
    nepochs=FLAGS.nepochs,
    batch_size=FLAGS.batch_size,
    save_net=FLAGS.save_net,
    save_root=FLAGS.save_root,
    save_net_every=FLAGS.save_net_every
  )

  if FLAGS.save_final_net:
    net.save(FLAGS.save_final_path)

  networks.print_net_accuracy(
    net,
    data_train,
    data_test,
    criterion
  )

if __name__ == '__main__':
  app.run(main)
