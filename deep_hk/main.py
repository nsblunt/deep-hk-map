from absl import app
from absl import flags
from hamiltonian import SpinlessHubbard
from wave_function import WaveFunction

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

def main(argv):
  del argv

  model = SpinlessHubbard(
    U=FLAGS.U,
    t=FLAGS.t,
    mu=FLAGS.mu,
    nsites=FLAGS.nsites,
    fixed_nparticles=FLAGS.fixed_nparticles,
    nparticles=FLAGS.nparticles,
    seed=FLAGS.seed
  )

  model.construct()

  # find and print eigenvectors and energies
  wf = WaveFunction(model)
  wf.solve_eigenvalue(model.hamil)
  wf.print_energies()
  wf.print_ground()

if __name__ == '__main__':
  app.run(main)
