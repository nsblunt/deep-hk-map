from absl import app
from absl import flags
from data import Data
from hamiltonian import SpinlessHubbard
from wave_function import WaveFunction
from networks import LinearNet, train_network
import torch.nn as nn

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

flags.DEFINE_integer('ntrain', 10000, 'Number of training samples to '
                      'generate.')
flags.DEFINE_integer('ntest', 100, 'Number of test samples to generate.')
flags.DEFINE_integer('nbatch', 100, 'Number of samples per training batch.')
flags.DEFINE_integer('nepochs', 100, 'Number of training epochs to perform.')

def main(argv):
  del argv

  system = SpinlessHubbard(
    U=FLAGS.U,
    t=FLAGS.t,
    mu=FLAGS.mu,
    nsites=FLAGS.nsites,
    fixed_nparticles=FLAGS.fixed_nparticles,
    nparticles=FLAGS.nparticles,
    seed=FLAGS.seed
  )
  system.construct()

  data = Data(
    system=system,
    ntrain=FLAGS.ntrain,
    ntest=FLAGS.ntest,
    nbatch=FLAGS.nbatch
  )
  data.gen_training_data()
  data.gen_test_data()

  layers_list = [nn.Linear(FLAGS.nsites, 100), nn.Linear(100, 1)]
  net = LinearNet(layers_list)

  train_network(net, data, FLAGS.nepochs)

if __name__ == '__main__':
  app.run(main)
