import numpy as np

class WaveFunction:
  """Class to store a wave function and methods to calculate properties."""

  def __init__(self, model):
    self.nsites = model.nsites
    self.fixed_nparticles = model.fixed_nparticles
    self.nparticles = model.nparticles

    self.dets = model.dets
    self.ndets = model.ndets

    self.coeffs = None
    self.energies = None

    self.density_gs = None

  def solve_eigenvalue(self, hamil):
    """Solve the eigenvalue problem for the provided Hamiltonian. The
       results are stored internally in energies and coeffs.

    Args:
      hamil: numpy ndarray which should be of size (ndets, ndets), which
        holds the Hamiltonian matrix
    """
    self.energies, self.coeffs = np.linalg.eigh(hamil)

  def calc_gs_density(self):
    """Calculate the local density from the ground state wave function.
       Also print the total sum of the densities, which should be the
       total number of particles."""
    self.density_gs = np.zeros( self.nsites )
    for det, coeff in zip(self.dets, self.coeffs[:,0]):
      for site in det:
        self.density_gs[site] += coeff**2

  def print_energies(self):
    """Print the list of energies to screen."""
    print("Energies:")
    for i in range(self.ndets):
      print('{:6d}  {: .8e}'.format(i, self.energies[i]))

  def print_gs_density(self):
    """Print the ground-state local density."""
    print("Ground-state local density:")
    total = 0.0
    for i in range(self.nsites):
      total += self.density_gs[i]
      print('{:6d}  {: .8e}'.format(i, self.density_gs[i]))
    print('Summation: {:6.2f}'.format(total))

  def print_ground(self):
    """Print the ground state wave function coefficients (and corresponding
       occupation lists for each determinant) to the screen.""" 
    print("Ground state wave function:")
    for i in range(self.ndets):
      occ_i = self.dets[i]
      if abs(self.coeffs[i,0]) > 1.e-10:
        print('{:6d}  {}  {: .8e}'.format(i, occ_i, self.coeffs[i,0]))
      else:
        print('{:6d}  {}  {: .8e}'.format(i, occ_i, 0.0))
