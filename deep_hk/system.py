import numpy as np
import itertools
import random

class SpinlessHubbard:
  """Hamiltonian for a spinless Hubbard model."""

  def __init__(self, U, t, mu, nsites, fixed_nparticles, nparticles, seed):
    """Initialises an object for the Hamiltonian of a spinless Hubbard model.

    Args:
      U: float
        The density-density interaction parameter
      t: float
        The hopping amplitude parameter
      mu: float
        The chemical potential
      nsites: int
        The number of lattice sites
      fixed_nparticles: bool
        True if considering a fixed number of particles. False if
        considering all particle number sectors simultaneously.
      nparticles: int
        The number of particles. This is only used if fixed_nparticles
        is True.
      seed: int
        Seed for the random number generator, used to generate random
        potentials.

    Other Attributes:
      dets: list of (tuple of int)
        List of determinants which span the space under consideration.
        Each determinant is represented as a tuple holding the occupied sites.
      ndets: int
        The total number of determinants
      hamil: numpy ndarray of size (ndets, ndets)
        Hamiltonian matrix for the spinless Hubbard model
      hamil_diag: numpy ndarray of size (ndets)
        The diagonal elements of the Hamiltonian matrix
    """
    self.U = U
    self.t = t
    self.mu = mu
    self.nsites = nsites
    self.fixed_nparticles = fixed_nparticles
    self.nparticles = nparticles
    self.seed = seed

    self.dets = None
    self.ndets = None
    self.hamil = None
    self.hamil_diag = None

    random.seed(self.seed)

  def generate_dets(self):
    """Generate the full list of determinants that span the space."""
    self.dets = []

    if self.fixed_nparticles:
      # generate all determinants with nparticles fermions in nsites orbitals
      r = itertools.combinations(range(self.nsites), self.nparticles)
      for item in r:
        self.dets.append(item)
    else:
      # generate all determinants of all occupations numbers, from 0 to nsites
      for i in range(2**self.nsites):
        # binary string representation of determinant i
        i_bin = bin(i)[2:].zfill(self.nsites)
        occ_list = [ind for ind,a in enumerate(i_bin) if a == '1']
        occ_tuple = tuple(occ_list)
        self.dets.append(occ_tuple)

    self.ndets = len(self.dets)

  def construct(self):
    """Construct the Hamiltonian, which is a numpy ndarray."""

    self.generate_dets()

    self.hamil = np.zeros((self.ndets, self.ndets), dtype=float)
    self.hamil_diag = np.zeros(self.ndets, dtype=float)

    for i in range(self.ndets):
      count_i = len(self.dets[i])

      for j in range(self.ndets):
        if i == j:
          self.hamil[i,i] = self.diag_hamil_elem(self.dets[i])
          self.hamil_diag[i] = self.hamil[i,i]
        else:
          if j < i:
            continue
          # The number of occupied sites for each determinant
          count_j = len(self.dets[j])
          # The Hamiltonian only connects determinants with equal numbers of
          # orbitals occupied
          if count_i == count_j:
            # Find which sites have had their occupation changed, which are
            # the excitations
            ind_ex_set = set(self.dets[i]).symmetric_difference(set(self.dets[j]))
            ind_ex = tuple(ind_ex_set)
            count_ex = len(ind_ex)
     
            # Can only have a non-zero off-diagonal element for a single
            # excitation, which is this condition:
            if count_ex == 2:
              # If connected then we have a non-zero Hamiltonian element
              if self.connected(ind_ex):
                par = self.parity_single(self.dets[i], self.dets[j], ind_ex)
                self.hamil[i,j] = -self.t * par
                self.hamil[j,i] = -self.t * par

  def diag_hamil_elem(self, occ_list):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by i_bin.
  
    Args:
      occ_list: tuple of int
        tuple holding all occupied sites in the determinant
    """
    nparticles = len(occ_list)
  
    if nparticles == 0:
      return 0.0
  
    # count the number of 1-1 bonds
    nbonds = 0
    if nparticles > 1:
      for ind in range(nparticles-1):
        if occ_list[ind]+1 == occ_list[ind+1]:
          nbonds += 1
      # account for periodicity
      if occ_list[0] == 0 and occ_list[nparticles-1] == self.nsites-1:
        nbonds += 1
  
    diag_elem = (self.U * nbonds) - (self.mu * nparticles)
    return diag_elem

  def connected(self, ind_ex):
    """Return true if two sites are connected on the lattice.
  
    Args:
      ind_ex: tuple of int
        the two sites who occupation changes in the excitation
    """
    # connected if nearest neighbours
    if ind_ex[1] == ind_ex[0]+1:
      return True
    elif ind_ex[0] == 0 and ind_ex[1] == self.nsites-1: # account for periodicity
      return True
    else:
      return False

  def parity_single(self, occ_i, occ_j, ind_ex):
    """Calculate the parity (+1 or -1) for the Hamiltonian element between
       the two determinants represented by bit strings i and j, which are a
       single excitation apart from each other.
  
    Args:
      occ_i: tuple of int
        tuple holding all occupied sites in determinant i
      occ_j: tuple of int
        tuple holding all occupied sites in determinant j
      ind_ex: tuple of int
        the two sites whose occupation changes in the excitation
    """
    if ind_ex[0] in occ_i:
      ind_i = ind_ex[0]
      ind_j = ind_ex[1]
    else:
      ind_i = ind_ex[1]
      ind_j = ind_ex[0]
  
    # find the positions of the two electrons involved in the lists of
    # occupied sites
    iel = occ_i.index(ind_i)
    jel = occ_j.index(ind_j)
  
    # if this is odd then the Hamiltonian element gains a factor of -1
    par = iel + jel
  
    return 1 if par%2 == 0 else -1

  def gen_rand_potential(self):
    """Generate a random potential, where the potential on each site is a
       random number between -0.5 and 0.5

    Returns:
      V: numpy ndarray of size (nsites)
        a random external potential
    """
    V = np.zeros( self.nsites )
    for i in range(self.nsites):
      V[i] = random.uniform(-0.5, 0.5)
    return V

  def add_potential_to_hamil(self, V):
    """Add the potential V into the Hamiltonian object, hamil.

    Args:
      V: numpy ndarray of size (nsites)
        an external potential
    """
    for i in range(self.ndets):
      # loop over all occupied sites in determinant i:
      self.hamil[i,i] = self.hamil_diag[i]
      for site in self.dets[i]:
        self.hamil[i,i] += V[site]
