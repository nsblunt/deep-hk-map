import itertools
from heapq import merge
import numpy as np
import random
from scipy.sparse import csr_matrix

class Hubbard:
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
    mu : float
      The chemical potential.
    max_V : float
      The maximum absolute value of the potential applied to each site
      (when generating randon potentials).
    nsites : int
      The number of lattice sites.
    fixed_nparticles : bool
      True if considering a fixed number of particles. False if
      considering all particle number sectors simultaneously.
    nparticles : int
      The number of particles. This is only used if fixed_nparticles
      is True.
    fixed_Ms : bool
      If true, then only consider determinants in a fixed-Ms sector,
      where Ms is the total (Sz) spin quantum number.
    Ms : int
      The spin of the states (in units of electron spin).
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
      Hamiltonian for the Hubbard model, stored in sparse
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
    self.U = U
    self.t = t
    self.mu = mu
    self.max_V = max_V
    self.nsites = nsites
    self.norbs = 2*nsites
    self.fixed_nparticles = fixed_nparticles
    self.nparticles = nparticles
    self.fixed_Ms = fixed_Ms
    self.Ms = Ms
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
                par = self.parity_single(self.dets[i], self.dets[j], ind_ex)
                hamil_elem = -self.t * par
                self.hamil_data.append(hamil_elem)
                self.row_ind.append(i)
                self.col_ind.append(j)

    # Make the Hamiltonian in CSR form.
    self.hamil = csr_matrix(
        (self.hamil_data, (self.row_ind, self.col_ind)),
        shape=(self.ndets, self.ndets))

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
        site = orb // 2
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
