"""Define and construct Hamiltonian objects for lattice models."""

import abc
import bisect
from heapq import merge
import itertools
import numpy as np
import random
from scipy.sparse import csr_matrix
from math import isqrt

def generate_all_dets(norbs):
  """Generate all determinants with all combinations of nparticles (the
     number of particles) and Ms (the total spin).

  Args
  ----
  norbs : int
    The number of orbitals.

  Returns
  -------
  dets : list of (tuple of int)
    List of determinants generated.
  """
  dets = []
  for i in range(2**norbs):
    # Binary string representation of determinant i.
    i_bin = bin(i)[2:].zfill(norbs)
    occ_list = [ind for ind,a in enumerate(i_bin) if a == '1']
    occ_tuple = tuple(occ_list)
    dets.append(occ_tuple)
  return dets

def generate_all_dets_fixed_n_and_ms(nsites, nparticles, Ms):
  """Generate all determinants with nparticles fixed and also a fixed
     value of the total spin, Ms.

     IMPORTANT: This function should be used for lattice models with
     both spin up and down particles. This is assumed.

  Args
  ----
  nsites : int
    The number of sites on the lattice.
  nparticles : int
    The number of particles in each determinant.
  Ms : int
    The total spin of each determinant (in units of electron spin).

  Returns
  -------
  dets : list of (tuple of int)
    List of determinants generated.
  """
  nup = (nparticles + Ms)//2
  ndown = (nparticles - Ms)//2

  # Generate all spin-up and spin-down combinations:
  rup = itertools.combinations(range(nsites), nup)
  rdown = itertools.combinations(range(nsites), ndown)

  # Lists of all determinants formed from up/down-spin orbitals only.
  dets_up_list = []
  dets_down_list = []

  # Convert from site indices to orbital indices.
  for sites_up in rup:
    orbs_up = tuple(2*site for site in sites_up)
    dets_up_list.append(orbs_up)
  for sites_down in rdown:
    orbs_down = tuple(2*site+1 for site in sites_down)
    dets_down_list.append(orbs_down)

  dets = []
  # Now create the final list of determinants, dets.
  for det_up, det_down in itertools.product(dets_up_list, dets_down_list):
    det = tuple(merge(det_up, det_down))
    dets.append(det)

  return dets


class LatticeHamil(metaclass=abc.ABCMeta):
  """Abstract base class for the Hamiltonian of a lattice model."""

  def __init__(self,
               mu,
               max_V,
               nsites,
               nspin,
               fixed_Ms,
               Ms,
               fixed_nparticles,
               nparticles,
               nonlocal_pot=False,
               lattice_type='1d',
               seed=7):
    """Initialises an object for the Hamiltonian of a lattice model.

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
    nonlocal_pot : bool
      If True then the system is set up to allow non-local potentials
      to be applied. If False, then it is assumed local potentials will
      be used.
    lattice_type : string
      String specifying the lattice type, i.e. '1d' or 'square'.
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
    dets_dict : dictionary
      Maps determinants in dets to their position in that list.
    configs : list of (ndarray of size norbs)
      The same list of determinants stored in dets, but stored in a
      different representation. Here, each configuration is a tuple of
      integers, where 0 represents that the orbital is unoccupied, 1
      that it is occupied. This is only created if generate_configs
      is called.
    connected_arr : numpy integer ndarray of size (nsites,nsites)
      Defines the connectivity of the lattice. If connected_arr[i,j] is
      equal to 1 then sites i and j are connected as nearest neighbours,
      otherwise this element is equal to 0 and they are not connected.
    connected_pairs : list of (tuple of two ints)
      A list of every pair of sites which are connected in the lattice.
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
    self.nonlocal_pot = nonlocal_pot
    self.lattice_type = lattice_type
    self.seed = seed

    self.dets = None
    self.dets_dict = None
    self.configs = None
    self.ndets = None
    self.connected_arr = None
    self.connected_pairs = None

    self.hamil = None
    self.hamil_diag = None

    self.hamil_data = []
    self.row_ind = []
    self.col_ind = []
    self.diag_pos = []

    # Holds the positions of Hamiltonian elements H_{ij} where i and j
    # are up to a single excitation apart, within the sparse
    # representation of the Hamiltonian. These are the elements that
    # are modified when a non-local potential is added.
    self.nonlocal_pos = []
    # Holds the values of the Hamiltonian elements where a non-local
    # potential will be added later, in the same order that they appear
    # with the spare representation of the Hamiltonian matrix.
    self.hamil_without_nonlocal = []

    random.seed(self.seed)

  @abc.abstractmethod
  def generate_dets(self):
    """Generate the full list of determinants that span the space."""

  def construct(self):
    """Construct the Hamiltonian, which is a sparse scipy CSR matrix."""

    self.generate_dets()
    self.make_connected()

    self.hamil_diag = np.zeros(self.ndets, dtype=float)

    for i in range(self.ndets):
      count_i = len(self.dets[i])

      positions_j, dets_j = self.gen_excitations(self.dets[i])

      #for j in range(self.ndets):
      for j, det_j in zip(positions_j, dets_j):
        if i == j:
          diag_elem = self.diag_hamil_elem(self.dets[i])
          diag_counter = len(self.hamil_data)

          self.hamil_data.append(diag_elem)
          self.row_ind.append(i)
          self.col_ind.append(i)

          self.diag_pos.append(diag_counter)
          self.hamil_diag[i] = diag_elem
          if self.nonlocal_pot:
            self.nonlocal_pos.append(diag_counter)
            self.hamil_without_nonlocal.append(diag_elem)
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
              # If applying random non-local potentials, then we store
              # the positions of the single excitations in the sparse
              # Hamiltonian matrix.
              if self.nonlocal_pot:
                offdiag_counter = len(self.hamil_data)
                self.nonlocal_pos.append(offdiag_counter)

              # If connected then we have a non-zero Hamiltonian element.
              if self.connected(ind_ex):
                hamil_elem = self.off_diag_hamil_elem(
                    self.dets[i],
                    self.dets[j],
                    ind_ex)
                self.hamil_data.append(hamil_elem)
                self.row_ind.append(i)
                self.col_ind.append(j)
              else:
                if self.nonlocal_pot:
                  hamil_elem = 0.0
                  self.hamil_data.append(hamil_elem)
                  self.row_ind.append(i)
                  self.col_ind.append(j)

              if self.nonlocal_pot:
                # Store the value of the Hamiltonian at an element where
                # a non-local potential may later be added.
                self.hamil_without_nonlocal.append(hamil_elem)


    # Make the Hamiltonian in CSR form.
    self.hamil = csr_matrix(
        (self.hamil_data, (self.row_ind, self.col_ind)),
        shape=(self.ndets, self.ndets))

  def gen_excitations(self, occ_i):
    """Generate a list of determinants which are excitations from the
       determinant represented by occ_i, including occ_i itself.

    Args
    ----
    occ_i : tuple of int
      Tuple holding all occupied orbitals in the determinant.

    Returns
    -------
    sorted_positions : list of int
      The positions of the excitations in the main list of determinants
      (self.dets), sorted in the same order.
    sorted_excitations : list of (tuple of int)
      A list of determinants which are the excitations of occ_i,
      including occ_i. They are sorted in the same order as the main
      list of determinants (self.dets).
    """
    excitations = [occ_i]
    occ_i_pos = self.dets_dict[occ_i]
    positions = [occ_i_pos]

    all_orbs = range(self.norbs)
    unocc_orbs = set(all_orbs).difference(set(occ_i))

    for ind_i, i in reversed(list(enumerate(occ_i))):
      for a in unocc_orbs:
        # Make sure the excitation conserves spin
        if i%self.nspin == a%self.nspin:
          occ_j_list = list(occ_i)
          occ_j_list.pop(ind_i)
          # Insert a and keep the list sorted
          bisect.insort(occ_j_list, a)

          occ_j = tuple(occ_j_list)
          occ_j_pos = self.dets_dict[occ_j]
          excitations.append(occ_j)
          positions.append(occ_j_pos)

    # Now we sort the excitations to be in the same order as their
    # positions in the full list of determinants
    sorted_combined = sorted(zip(positions, excitations))
    sorted_positions = [x[0] for x in sorted_combined]
    sorted_excitations = [x[1] for x in sorted_combined]

    return sorted_positions, sorted_excitations

  @abc.abstractmethod
  def diag_hamil_elem(self, occ):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by occ.

    Args
    ----
    occ : tuple of int
      Tuple holding all occupied orbitals in the determinant.
    """

  @abc.abstractmethod
  def off_diag_hamil_elem(self, occ_1, occ_2, ind_ex):
    """Generate and return the off-diagonal element of the Hamiltonian,
       corresponding to determinants represented by occ_1 and
       occ_2.

       IMPORTANT: This function assumes that the two determinants are
       a single excitation apart, which should be checked before using
       this. ind_ex holds the two orbitals involved in the excitation.

    Args
    ----
    occ_1 : tuple of int
      Tuple holding all occupied orbitals in determinant 1.
    occ_2 : tuple of int
      Tuple holding all occupied orbitals in determinant 2.
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """

  def make_connected(self):
    """Construct the self.connected array, which defines which
       lattice sites are nearest neighbours.
    """

    self.connected_arr = np.zeros((self.nsites,self.nsites), dtype=int)

    if self.lattice_type == '1d':
      for i in range(self.nsites):
        j = (i+1)%self.nsites
        self.connected_arr[i,j] = 1

        j = (i-1)%self.nsites
        self.connected_arr[i,j] = 1

    elif self.lattice_type == 'square':
      width = isqrt(self.nsites)
      if width**2 != self.nsites:
        raise ValueError('Number of lattice sites is not a square number.')

      for i_1 in range(width):
        for j_1 in range(width):
          ind_a = i_1*width + j_1

          i_2 = (i_1+1)%width
          j_2 = j_1
          ind_b = i_2*width + j_2
          self.connected_arr[ind_a,ind_b] = 1

          i_2 = (i_1-1)%width
          j_2 = j_1
          ind_b = i_2*width + j_2
          self.connected_arr[ind_a,ind_b] = 1

          i_2 = i_1
          j_2 = (j_1+1)%width
          ind_b = i_2*width + j_2
          self.connected_arr[ind_a,ind_b] = 1

          i_2 = i_1
          j_2 = (j_1-1)%width
          ind_b = i_2*width + j_2
          self.connected_arr[ind_a,ind_b] = 1

    elif self.lattice_type == 'from_file':
      f = open('connected.txt')
      for line in f:
        if '#' not in line:
          inds = line.split()
          if len(inds) == 0:
            # Empty line
            continue
          elif len(inds) == 2:
            i = int(inds[0])
            j = int(inds[1])
            if i >= self.nsites or j >= self.nsites:
              raise ValueError('Site label in connected.txt is higher '
                  'than the maximum lattice site index (zero-indexed).')
            self.connected_arr[i,j] = 1
            self.connected_arr[j,i] = 1
          else:
            raise ValueError('connected.txt should consist of parirs of'
                'integers, each on one line of the file.')
      f.close()

    self.make_connected_pairs()

  def make_connected_pairs(self):
    """Make the connected_pairs array, which is a list of pairs of
       connected sites on the lattice."""
    self.connected_pairs = []

    for i in range(self.nsites):
      for j in range(i):
        if self.connected_arr[i,j] == 1:
          self.connected_pairs.append((j,i))

  def connected(self, ind_ex):
    """Return true if two orbitals are connected on the lattice.

    Args
    ----
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.

    Returns
    -------
    nearest_neighbours : bool
      True is the two orbitals correspond to sites which are
      nearest neighbours on the lattice, as defined by the
      self.connected array.
    """

    site_1 = ind_ex[0] // self.nspin
    site_2 = ind_ex[1] // self.nspin

    nearest_neighbours = self.connected_arr[site_1,site_2] == 1

    return nearest_neighbours

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
      A random external local potential.
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

  def gen_rand_nonlocal_potential(self):
    """Generate a random non-local potential, where the potential on
       each site is a random number between -self.max_V and +self.max_V.

       The potential matrix is symmetric, as required to make the
       Hamiltonian Hermitian.

    Returns
    -------
    V : numpy ndarray of size (nsites,nsites)
      A random external non-local potential.
    """
    V = np.zeros( (self.nsites, self.nsites) )
    for i in range(self.nsites):
      for j in range(i+1):
        V[i,j] = random.uniform(-self.max_V, self.max_V)
        V[j,i] = V[i,j]

    return V

  def compress_potential(self, V):
    """Take a non-local potential stored in V, which is a
       two-dimensional symmetric matrix, and convert it into a
       1D array without the repeated elements.

    Args
    ----
    V : numpy ndarray of size (nsites, nsites)
      A non-local external potential, stored as a 2D array.

    Returns
    -------
    V_compressed : numpy ndarray of size (nsites*(nsites+1)/2)
      V converted into a compressed 1D array.
    """

    nelem = int(self.nsites*(self.nsites+1)/2)
    V_compressed = np.zeros(nelem)

    for i in range(self.nsites):
      for j in range(i+1):
        ind = int(i*(i+1)/2) + j
        V_compressed[ind] = V[i,j]

    return V_compressed

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

  def add_nonlocal_potential_to_hamil(self, V):
    """Add a non-local potential V into the Hamiltonian object, hamil.

    Args
    ----
    V : numpy ndarray of size (nsites,nistes)
      An external non-local potential.
    """

    for i, hamil_ind in enumerate(self.nonlocal_pos):
      det_i_ind = self.row_ind[hamil_ind]
      det_j_ind = self.col_ind[hamil_ind]

      det_i = self.dets[det_i_ind]
      det_j = self.dets[det_j_ind]

      # Get the orbitals involved in the excitation
      ind_ex_set = set(det_i).symmetric_difference(set(det_j))
      ind_ex = tuple(ind_ex_set)
      count_ex = len(ind_ex)

      self.hamil.data[hamil_ind] = self.hamil_without_nonlocal[i]

      if count_ex == 0:
        # Diagonal element
        # Loop over all occupied sites in the determinant
        for orb in det_i:
          # Convert orbital index to site index
          site = orb // self.nspin
          self.hamil.data[hamil_ind] += V[site,site]
      elif count_ex == 2:
        # Single excitation
        # Make sure spin is conserved
        if ind_ex[0]%self.nspin == ind_ex[1]%self.nspin:
          # Convert orbital indices to site indices
          site_1 = ind_ex[0] // self.nspin
          site_2 = ind_ex[1] // self.nspin
          par = self.parity_single(det_i, det_j, ind_ex)
          self.hamil.data[hamil_ind] += par * V[site_1,site_2]
      else:
        raise ValueError('Greater than single excitation found in '
            'non-local potential calculation.')

  def generate_configs(self):
    """Generate the configurations, stored as tuples of 0's and 1's,
       where 0 indicates that an orbital is unoccupied, 1 that it is
       occupied.
    """

    self.configs = []

    for det in self.dets:
      config = np.zeros(self.norbs, dtype=float)
      for orb in det:
        config[orb] = 1
      self.configs.append(config)

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
    energy = np.dot(wf, self.hamil.dot(wf)) / np.dot(wf, wf)
    return energy

  def remove_sign_problem(self):
    """Make all off-diagonal Hamiltonian elements negative, which
       removes the sign problem in the Hamiltonian.
    """
    for i in range(len(self.hamil.data)):
      # Off-diagonal element
      if self.row_ind[i] != self.col_ind[i]:
        if self.hamil.data[i] > 0.0:
          self.hamil.data[i] *= -1.0

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
               nonlocal_pot=False,
               lattice_type='1d',
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
        nonlocal_pot=nonlocal_pot,
        lattice_type=lattice_type,
        seed=seed)

  def generate_dets(self):
    """Generate the full list of determinants that span the space."""
    self.dets = []

    if self.fixed_nparticles:
      if self.fixed_Ms:
        self.dets = generate_all_dets_fixed_n_and_ms(
          nsites=self.nsites,
          nparticles=self.nparticles,
          Ms=self.Ms)
      else:
        # Generate all determinants with nparticles fermions in norb
        # spin-orbitals.
        r = itertools.combinations(range(self.norbs), self.nparticles)
        for item in r:
          self.dets.append(item)
    else:
      self.dets = generate_all_dets(self.norbs)

    self.ndets = len(self.dets)

    # Generate dictionary to allow us to efficiently find the index
    # of a given determinant in the full list.
    self.dets_dict = {}
    for i, det_i in enumerate(self.dets):
      self.dets_dict[det_i] = i

  def diag_hamil_elem(self, occ):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by occ.

    Args
    ----
    occ : tuple of int
      Tuple holding all occupied spin orbitals in the determinant.
    """
    nparticles = len(occ)

    if nparticles == 0:
      return 0.0

    # Count the number of doubly occupied orbitals.
    ndouble = 0
    if nparticles > 1:
      for ind in range(nparticles-1):
        # Alpha (beta) orbitals have an even (odd) index.
        # If this is an alpha electron:
        if occ[ind]%2 == 0:
          if occ[ind]+1 == occ[ind+1]:
            ndouble += 1

    diag_elem = (self.U * ndouble) - (self.mu * nparticles)
    return diag_elem

  def off_diag_hamil_elem(self, occ_1, occ_2, ind_ex):
    """Generate and return the off-diagonal element of the Hamiltonian,
       corresponding to determinants represented by occ_1 and
       occ_2.

       IMPORTANT: This function assumes that the two determinants are
       a single excitation apart, which should be checked before using
       this. ind_ex holds the two orbitals involved in the excitation.

    Args
    ----
    occ_1 : tuple of int
      Tuple holding all occupied orbitals in determinant 1.
    occ_2 : tuple of int
      Tuple holding all occupied orbitals in determinant 2.
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """
    par = self.parity_single(occ_1, occ_2, ind_ex)
    return -self.t * par


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
               nonlocal_pot=False,
               lattice_type='1d',
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
        nonlocal_pot=nonlocal_pot,
        lattice_type=lattice_type,
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
      self.dets = generate_all_dets(self.norbs)

    self.ndets = len(self.dets)

    # Generate dictionary to allow us to efficiently find the index
    # of a given determinant in the full list.
    self.dets_dict = {}
    for i, det_i in enumerate(self.dets):
      self.dets_dict[det_i] = i

  def diag_hamil_elem(self, occ):
    """Generate and return the diagonal element of the Hamiltonian,
       corresponding to determinant represented by occ.

    Args
    ----
    occ : tuple of int
      Tuple holding all occupied orbitals in the determinant.
    """
    nparticles = len(occ)

    if nparticles == 0:
      return 0.0

    # Count the number of 1-1 bonds.
    nbonds = 0
    if nparticles > 1:
      # Loop over all connected pairs of sites
      for pair in self.connected_pairs:
        site_1 = pair[0]
        site_2 = pair[1]
        if site_1 in occ and site_2 in occ:
          nbonds += 1

    diag_elem = (self.U * nbonds) - (self.mu * nparticles)
    return diag_elem

  def off_diag_hamil_elem(self, occ_1, occ_2, ind_ex):
    """Generate and return the off-diagonal element of the Hamiltonian,
       corresponding to determinants represented by occ_1 and
       occ_2.

       IMPORTANT: This function assumes that the two determinants are
       a single excitation apart, which should be checked before using
       this. ind_ex holds the two orbitals involved in the excitation.

    Args
    ----
    occ_1 : tuple of int
      Tuple holding all occupied orbitals in determinant 1.
    occ_2 : tuple of int
      Tuple holding all occupied orbitals in determinant 2.
    ind_ex : tuple of int
      The two orbitals whose occupation changes in the excitation.
    """
    par = self.parity_single(occ_1, occ_2, ind_ex)
    return -self.t * par
