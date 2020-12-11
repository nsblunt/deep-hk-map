from hamiltonian import LatticeHamil
from heapq import merge
import itertools
import numpy as np
import random
from scipy.sparse import csr_matrix

class Hubbard(LatticeHamil):
  """Hamiltonian for a Hubbard model."""

  def __init__(self,
               U,
               t,
               mu,
               max_V,
               nsites,
               fixed_nparticles,
               nparticles,
               fixed_Ms=True,
               Ms=0,
               seed=7):
    """Initialises an object for the Hamiltonian of a Hubbard model.

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
        nspin=2,
        fixed_Ms=fixed_Ms,
        Ms=Ms,
        fixed_nparticles=fixed_nparticles,
        nparticles=nparticles,
        seed=seed)

  def generate_dets(self):
    """Generate the full list of determinants that span the space."""
    self.dets = []

    if self.fixed_nparticles:
      if self.fixed_Ms:
        # Generate all determinants with nparticles fermions in norb
        # spin-orbitals *and* a fixed value of spin, Ms.

        nup = (self.nparticles + self.Ms)//2
        ndown = (self.nparticles - self.Ms)//2

        # Generate all spin-up and spin-down combinations:
        rup = itertools.combinations(range(self.nsites), nup)
        rdown = itertools.combinations(range(self.nsites), ndown)

        # Lists of all determinants formed from up/down-spin orbitals
        # only.
        dets_up_list = []
        dets_down_list = []

        # Convert from site indices to orbital indices.
        for sites_up in rup:
          orbs_up = tuple(2*site for site in sites_up)
          dets_up_list.append(orbs_up)
        for sites_down in rdown:
          orbs_down = tuple(2*site+1 for site in sites_down)
          dets_down_list.append(orbs_down)

        # Now create the final list of determinants, dets.
        for det_up, det_down in itertools.product(dets_up_list, dets_down_list):
          det = tuple(merge(det_up, det_down))
          self.dets.append(det)
      else:
        # Generate all determinants with nparticles fermions in norb
        # spin-orbitals.
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
      Tuple holding all occupied spin orbitals in the determinant.
    """
    nparticles = len(occ_list)
  
    if nparticles == 0:
      return 0.0
  
    # Count the number of doubly occupied orbitals.
    ndouble = 0
    if nparticles > 1:
      for ind in range(nparticles-1):
        # Alpha (beta) orbitals have an even (odd) index.
        # If this is an alpha electron:
        if occ_list[ind]%2 == 0:
          if occ_list[ind]+1 == occ_list[ind+1]:
            ndouble += 1
  
    diag_elem = (self.U * ndouble) - (self.mu * nparticles)
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
    """Return true if two orbitals are connected on the lattice. Two
       orbitals are connected if they are on neighbouring lattice sites
       and also have the same spin.
  
    Args
    ----
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """
    # Orbitals are connected if nearest neighbours and have the same
    # spin. The spin ordering is (alpha, beta, alpha, beta, ...).
    if ind_ex[1] == ind_ex[0]+2:
      return True
    elif ind_ex[0] == 0 and ind_ex[1] == self.norbs-2:
      # periodicity (alpha, alpha)
      return True
    elif ind_ex[0] == 1 and ind_ex[1] == self.norbs-1:
      # periodicity (beta, beta)
      return True
    else:
      return False
