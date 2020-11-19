from absl import app
from absl import flags
from data import Data
from system import SpinlessHubbard
from wave_function import WaveFunction
import networks

import torch
import torch.nn as nn
import torch.optim as optim

FLAGS = flags.FLAGS
flags.DEFINE_integer('nsites', 4, 'Number of lattice sites.')
flags.DEFINE_integer('nparticles', 2, 'Number of particles.')
flags.DEFINE_float('U', 1.0, 'Parameter U in the spinless Hubbard model.')
flags.DEFINE_float('t', 1.0, 'Parameter t in the spinless Hubbard model.')
flags.DEFINE_float('mu', 0.0, 'Chemical potential parameter.')
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

def main(argv):
  del argv

  sys = SpinlessHubbard(
    U=FLAGS.U,
    t=FLAGS.t,
    mu=FLAGS.mu,
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

  networks.print_net_accuracy(
    net,
    data_train,
    data_test,
    criterion
  )

if __name__ == '__main__':
  app.run(main)
