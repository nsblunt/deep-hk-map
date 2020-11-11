import numpy as np

class WaveFunction:
  """Class to store wave functions and calculate properties."""

  def __init__(self, nsites, dets):
    """Initialises an object for a wave function of a spinless lattice model.

    Args:
      nsites: int
        The number of lattice sites
      dets: list of (tuple of int)
        List of determinants which span the space under consideration.
        Each determinant is represented as a tuple holding the occupied sites.

    Other Attributes:
      ndets: int
        The total number of determinants
      coeffs: numpy ndarray of size (ndets, ndets)
        Array holding the energy eigenfunctions themselves. The i'th energy
        eigenstate has coefficients coeffs[:,i], with the same ordering as
        the determinants held in dets.
      energies: numpy ndarray of size (ndets)
        The energy eigenvalues
      density_gs: numpy ndarray of size (nsites)
        The local density of the ground-state wave function
      corr_fn_gs: numpy ndarray of size (nsites, nsites)
        The two-point density correlation function of the ground-state
        wave function
    """
    self.nsites = nsites
    self.dets = dets
    self.ndets = len(dets)

    # from the solution of eigenvalue problem
    self.coeffs = None
    self.energies = None

    # properties that can be calculated
    self.density_gs = None
    self.corr_fn_gs = None

  def solve_eigenvalue(self, hamil):
    """Solve the eigenvalue problem for the provided Hamiltonian. The
       results are stored internally in energies and coeffs.

    Args:
      hamil: numpy ndarray of size (ndets, ndets)
        the Hamiltonian matrix
    """
    self.energies, self.coeffs = np.linalg.eigh(hamil)

  def calc_gs_density(self):
    """Calculate the local density from the ground-state wave function."""
    self.density_gs = np.zeros( self.nsites )
    for det, coeff in zip(self.dets, self.coeffs[:,0]):
      for site in det:
        self.density_gs[site] += coeff**2

  def calc_corr_fn_gs(self):
    """Calculate the two-point density correlation function for the ground
       state wave function."""
    self.corr_fn_gs = np.zeros( (self.nsites, self.nsites) )
    for det, coeff in zip(self.dets, self.coeffs[:,0]):
      for site1 in det:
        for site2 in det:
          self.corr_fn_gs[site1,site2] += coeff**2

  def print_energies(self):
    """Print the list of energies to screen."""
    print("Energies:")
    for i in range(self.ndets):
      print('{:6d}  {: .8e}'.format(i, self.energies[i]))

  def print_gs_density(self):
    """Print the ground-state local density. Also print the total sum of
       the densities, which should be the total number of particles."""
    print("Ground-state local density:")
    total = 0.0
    for i in range(self.nsites):
      total += self.density_gs[i]
      print('{:6d}  {: .8e}'.format(i, self.density_gs[i]))
    print('Summation: {:6.2f}'.format(total))

  def print_corr_fn_gs(self):
    """Print the two-point density correlation function of the ground
       state wave function."""
    print(self.corr_fn_gs)
    print("Ground-state two-point correlation function:")
    for i in range(self.nsites):
      for j in range(self.nsites):
        print('({:6d}, {:6d})  {: .8e}'.format(i, j, self.corr_fn_gs[i,j]))

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
