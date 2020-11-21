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

flags.DEFINE_enum(
    'input_type', 'potential', ['potential', 'density', '1-rdm'], 'Specify '
    'which object we pass into the network input.')
flags.DEFINE_integer('ntrain', 12800, 'Number of training samples to '
                      'generate.')
flags.DEFINE_integer('ntest', 100, 'Number of test samples to generate.')
flags.DEFINE_integer('batch_size', 128, 'Number of samples per training batch.')
flags.DEFINE_integer('nepochs', 100, 'Number of training epochs to perform.')
flags.DEFINE_float('lr', 0.001, 'The learning rate for the optimizer.')

flags.DEFINE_list('layer_widths', [100], 'The number of hidden units in '
    'each layer of the network, input as comma-separated values.')
flags.DEFINE_bool('save_net', True, 'If True, then save the final trained '
    'network to a file.')
flags.DEFINE_string('save_path', './network.pt', 'Path and name for the file '
    'used to print the final network parameters.')
flags.DEFINE_bool('load_net', False, 'If True, then begin by loading the '
    'network from a file.')
flags.DEFINE_string('load_path', './network.pt', 'Path and name for the file '
    'used to print the final network parameters.')

flag_dict = FLAGS.flag_values_dict()
# Dictionary of input variables:
flag_dict_input = {k: v for k, v in flag_dict.items() if k not in flag_dict_init}

def main(argv):
  del argv

  # Print the inputs variables that specify the simulation
  print(json.dumps(flag_dict_input, indent=1), '\n')

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
    ninput = FLAGS.nsites
  elif FLAGS.input_type == '1-rdm':
    ninput = FLAGS.nsites**2

  data_train = Data(
    system=sys,
    ninput=ninput,
    ndata=FLAGS.ntrain,
    input_type=FLAGS.input_type
  )
  data_train.generate()

  data_test = Data(
    system=sys,
    ninput=ninput,
    ndata=FLAGS.ntest,
    input_type=FLAGS.input_type
  )
  data_test.generate()

  torch.manual_seed(FLAGS.seed)

  layer_widths = [int(s) for s in FLAGS.layer_widths]

  layers_list = networks.create_linear_layers(ninput, layer_widths, 1)
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
    batch_size=FLAGS.batch_size
  )

  if FLAGS.save_net:
    net.save(FLAGS.save_path)

  #networks.print_net_accuracy(
  #  net,
  #  data_train,
  #  data_test,
  #  criterion
  #)

if __name__ == '__main__':
  app.run(main)
