import abc
import numpy as np
from scipy.sparse import csr_matrix
import random

class LatticeHamil(metaclass=abc.ABCMeta):
  """Hamiltonian for a lattice model."""

  def __init__(self,
               mu,
               max_V,
               nsites,
               nspin,
               fixed_Ms,
               Ms,
               fixed_nparticles,
               nparticles,
               seed=7):
    """Initialises an object for the Hamiltonian of a lattice model..

    Args
    ----
    mu : float
      The chemical potential.
    max_V : float
      The maximum absolute value of the potential applied to each site
      (when generating randon potentials).
    nsites : int
      The number of lattice sites.
    nspin : int
      The number of spin states per site. For the spinless Hubbard
      model, this is equal to 1. For the spinned Hubbard model, this is
      equal to 2.
    fixed_Ms : bool
      If true, then only consider determinants in a fixed-Ms sector,
      where Ms is the total (Sz) spin quantum number. This is set to
      True and unused in the case of a spinless model.
    Ms : int
      The spin of the states (in units of electron spin). This is set
      to 0 and unused in the case of a spinless model.
    fixed_nparticles : bool
      True if considering a fixed number of particles. False if
      considering all particle number sectors simultaneously.
    nparticles : int
      The number of particles. This is only used if fixed_nparticles
      is True.
    seed : int
      Seed for the random number generator, used to generate random
      potentials.

    Other Attributes
    ----------------
    norbs : int
      The number of spin orbitals.
    dets : list of (tuple of int)
      List of determinants which span the space under consideration.
      Each determinant is represented as a tuple holding the occupied
      sites.
    ndets : int
      The total number of determinants.
    hamil : scipy csr_matrix
      Hamiltonian for the spinless Hubbard model, stored in sparse
      CSR form.
    hamil_diag : numpy ndarray of size (ndets)
      The diagonal elements of the Hamiltonian matrix.
    hamil_data : list of float
      List of non-zero Hamiltonian elements, in the order that they are
      generated.
    row_ind : list of int
      Row indicies of the non-zero Hamiltonian elements, in the order
      that they are generated.
    col_ind : list of int
      Column indicies of the non-zero Hamiltonian elements, in the order
      that they are generated.
    diag_pos : list of int
      The positions of diagonal elements in the hamil_data list.
    """
    self.mu = mu
    self.max_V = max_V
    self.nsites = nsites
    self.nspin = nspin
    self.norbs = nsites*nspin
    self.fixed_Ms = fixed_Ms
    self.Ms = Ms
    self.fixed_nparticles = fixed_nparticles
    self.nparticles = nparticles
    self.seed = seed

    self.dets = None
    self.ndets = None

    self.hamil = None
    self.hamil_diag = None

    self.hamil_data = []
    self.row_ind = []
    self.col_ind = []
    self.diag_pos = []

    random.seed(self.seed)

  @abc.abstractmethod
  def generate_dets(self):
    """Generate the full list of determinants that span the space."""

  def construct(self):
    """Construct the Hamiltonian, which is a sparse scipy CSR matrix."""

    self.generate_dets()

    self.hamil_diag = np.zeros(self.ndets, dtype=float)

    for i in range(self.ndets):
      count_i = len(self.dets[i])

      for j in range(self.ndets):
        if i == j:
          diag_elem = self.diag_hamil_elem(self.dets[i])
          diag_counter = len(self.hamil_data)

          self.hamil_data.append(diag_elem)
          self.row_ind.append(i)
          self.col_ind.append(i)
          self.diag_pos.append(diag_counter)

          self.hamil_diag[i] = diag_elem
        else:
          # The number of occupied orbitals for each determinant.
          count_j = len(self.dets[j])
          # The Hamiltonian only connects determinants with equal
          # numbers of orbitals occupied.
          if count_i == count_j:
            # Find which orbitals have had their occupation changed.
            # These define the excitation.
            ind_ex_set = set(self.dets[i]).symmetric_difference(set(self.dets[j]))
            ind_ex = tuple(ind_ex_set)
            count_ex = len(ind_ex)
     
            # Can only have a non-zero off-diagonal element for a single
            # excitation, which is this condition:
            if count_ex == 2:
              # If connected then we have a non-zero Hamiltonian element.
              if self.connected(ind_ex):
                hamil_elem = self.off_diag_hamil_elem(
                    self.dets[i],
                    self.dets[j],
                    ind_ex)
                self.hamil_data.append(hamil_elem)
                self.row_ind.append(i)
                self.col_ind.append(j)

    # Make the Hamiltonian in CSR form.
    self.hamil = csr_matrix(
        (self.hamil_data, (self.row_ind, self.col_ind)),
        shape=(self.ndets, self.ndets))

  @abc.abstractmethod
  def diag_hamil_elem(self, occ_list):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by occ_list.
  
    Args
    ----
    occ_list : tuple of int
      Tuple holding all occupied orbitals in the determinant.
    """

  @abc.abstractmethod
  def off_diag_hamil_elem(self, occ_list_1, occ_list_2, ind_ex):
    """Generate and return the off-diagonal element of the Hamiltonian,
       corresponding to determinants represented by occ_list_1 and
       occ_list_2.

       IMPORTANT: This function assumes that the two determinants are
       a single excitation apart, which should be checked before using
       this. ind_ex holds the two orbitals involved in the excitation.

    Args
    ----
    occ_list_1 : tuple of int
      Tuple holding all occupied orbitals in determinant 1.
    occ_list_2 : tuple of int
      Tuple holding all occupied orbitals in determinant 2.
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """

  @abc.abstractmethod
  def connected(self, ind_ex):
    """Return true if two orbitals are connected on the lattice.
  
    Args
    ----
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """

  def parity_single(self, occ_i, occ_j, ind_ex):
    """Calculate the parity (+1 or -1) for the Hamiltonian element
       between the two determinants represented by bit strings i and j,
       which are a single excitation apart from each other.
  
    Args
    ----
    occ_i : tuple of int
      Tuple holding all occupied orbitals in determinant i.
    occ_j : tuple of int
      Tuple holding all occupied orbitals in determinant j.
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """
    if ind_ex[0] in occ_i:
      ind_i = ind_ex[0]
      ind_j = ind_ex[1]
    else:
      ind_i = ind_ex[1]
      ind_j = ind_ex[0]
  
    # Find the positions of the two electrons involved in the lists of
    # occupied orbitals.
    iel = occ_i.index(ind_i)
    jel = occ_j.index(ind_j)
  
    # If this is odd then the Hamiltonian element gains a factor of -1.
    par = iel + jel
  
    return 1 if par%2 == 0 else -1

  def gen_rand_potential(self,
                         const_potential_sum=False,
                         potential_sum_val=0.0):
    """Generate a random potential, where the potential on each site is
       a random number between -self.max_V and +self.max_V.

       If const_potential_sum is true, then the generated potential is
       shifted uniformly so that the total summed potential is equal to
       potential_sum_val.

    Args
    ----
    const_potential_sum : bool
      If true, then uniformly shift the final potential so that the
      summed value (over all sites) is constant.
    potential_sum_val : float
      The value of the summed potential, if const_potential_sum is true.

    Returns
    -------
    V : numpy ndarray of size (nsites)
      A random external potential.
    """
    V = np.zeros( self.nsites )
    for i in range(self.nsites):
      V[i] = random.uniform(-self.max_V, self.max_V)

    if const_potential_sum:
      sum_V = sum(V)
      diff = potential_sum_val - sum_V
      diff_each_site = diff/self.nsites
      V += diff_each_site
    return V

  def add_potential_to_hamil(self, V):
    """Add the potential V into the Hamiltonian object, hamil.

    Args
    ----
    V : numpy ndarray of size (nsites)
      An external potential.
    """
    for i in range(self.ndets):
      diag_pos = self.diag_pos[i]
      self.hamil.data[diag_pos] = self.hamil_diag[i]
      # Loop over all occupied sites in determinant i.
      for orb in self.dets[i]:
        # Convert orbital index to site index.
        site = orb // self.nspin
        self.hamil.data[diag_pos] += V[site]

  def calc_energy(self, wf, V):
    """Calculate the expectation value of the Hamiltonian, with respect
       to the provided wave function.

    Args
    ----
    wf : numpy ndarray of size (ndets)
      The wave function to be used in the expectation value.
    V : numpy ndarray of size (nsites)
      The external potential.
    """
    self.add_potential_to_hamil(V)
    energy = np.dot(wf, self.hamil.multiply(wf))
    return energy
