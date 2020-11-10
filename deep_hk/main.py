from absl import app
from absl import flags
import numpy as np
from hamiltonian import SpinlessHubbard

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

  # Printing the solutions:
  e, psi = np.linalg.eigh(model.hamil)

  print("Energies:")
  for i in range(model.ndets):
    print(i, e[i])

  print("WF:")
  for i in range(model.ndets):
    i_bin = model.dets[i]
    if abs(psi[i,0]) > 1.e-10:
      npart = i_bin.count('1')
      print(i_bin, psi[i,0])
    else:
      print(i_bin, 0)

if __name__ == '__main__':
  app.run(main)
