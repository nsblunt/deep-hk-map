from hamiltonian import LatticeHamil
import itertools
import numpy as np
import random
from scipy.sparse import csr_matrix

class SpinlessHubbard(LatticeHamil):
  """Hamiltonian for a spinless Hubbard model."""

  def __init__(self,
               U,
               t,
               mu,
               max_V,
               nsites,
               fixed_nparticles,
               nparticles,
               seed=7):
    """Initialises an object for the Hamiltonian of a spinless Hubbard
       model.

    Args
    ----
    U : float
      The density-density interaction parameter.
    t : float
      The hopping amplitude parameter.

    Other arguments and attributes are defined in the base class
    (LatticeHamil) docstring.
    """
    self.U = U
    self.t = t

    super().__init__(
        mu=mu,
        max_V=max_V,
        nsites=nsites,
        nspin=1,
        fixed_Ms=True,
        Ms=0,
        fixed_nparticles=fixed_nparticles,
        nparticles=nparticles,
        seed=seed)

  def generate_dets(self):
    """Generate the full list of determinants that span the space."""
    self.dets = []

    if self.fixed_nparticles:
      # Generate all determinants with nparticles fermions in norb
      # orbitals.
      r = itertools.combinations(range(self.norbs), self.nparticles)
      for item in r:
        self.dets.append(item)
    else:
      # Generate all determinants of all occupations numbers, from 0 to
      # norbs.
      for i in range(2**self.norbs):
        # Binary string representation of determinant i.
        i_bin = bin(i)[2:].zfill(self.norbs)
        occ_list = [ind for ind,a in enumerate(i_bin) if a == '1']
        occ_tuple = tuple(occ_list)
        self.dets.append(occ_tuple)

    self.ndets = len(self.dets)

  def diag_hamil_elem(self, occ_list):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by occ_list.
  
    Args
    ----
    occ_list : tuple of int
      Tuple holding all occupied orbitals in the determinant.
    """
    nparticles = len(occ_list)
  
    if nparticles == 0:
      return 0.0
  
    # Count the number of 1-1 bonds.
    nbonds = 0
    if nparticles > 1:
      for ind in range(nparticles-1):
        if occ_list[ind]+1 == occ_list[ind+1]:
          nbonds += 1
      # Account for periodicity.
      if occ_list[0] == 0 and occ_list[nparticles-1] == self.norbs-1:
        nbonds += 1
  
    diag_elem = (self.U * nbonds) - (self.mu * nparticles)
    return diag_elem

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
    par = self.parity_single(occ_list_1, occ_list_2, ind_ex)
    return -self.t * par

  def connected(self, ind_ex):
    """Return true if two orbitals are connected on the lattice.
  
    Args
    ----
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """
    # Sites/orbitals are connected if nearest neighbours.
    if ind_ex[1] == ind_ex[0]+1:
      return True
    elif ind_ex[0] == 0 and ind_ex[1] == self.norbs-1: # (periodicity)
      return True
    else:
      return False
