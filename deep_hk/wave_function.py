"""Define the class to store wave function solutions to the Hamiltonian
   eigenvalue problem, and to calculate corresponding properties."""

import numpy as np
from scipy.sparse.linalg import eigsh
from itertools import count
import random

class WaveFunction:
  """Class to store wave functions and calculate properties."""

  def __init__(self, nsites, nspin, dets):
    """Initialises an object for a wave function of a spinless lattice
       model.

    Args
    ----
    nsites : int
      The number of lattice sites.
    nspin : int
      The number of spin states per site (1 or 2).
    dets : list of (tuple of int)
      List of determinants which span the space under consideration.
      Each determinant is represented as a tuple holding the occupied
      sites.

    Other Attributes
    ----------------
    ndets : int
      The total number of determinants.
    coeffs : numpy ndarray of size (ndets, ndets)
      Array holding the energy eigenfunctions themselves. The i'th
      energy eigenstate has coefficients coeffs[:,i], with the same
      ordering as the determinants held in dets.
    energies : numpy ndarray of size (ndets)
      The energy eigenvalues.
    density_gs : numpy ndarray of size (nsites)
      The local density of the ground-state wave function.
    corr_fn_gs : numpy ndarray of size (nsites, nsites)
      The two-point density correlation function of the ground-state
      wave function.
    rdm1_gs : numpy ndarray of size (nsites, nsites)
      One-body reduced density matrix for the ground-state wave function.
    """
    self.nsites = nsites
    self.nspin = nspin
    self.dets = dets
    self.ndets = len(dets)

    # Data from the solution of eigenvalue problem.
    self.coeffs = None
    self.energies = None

    # Properties that can be calculated.
    self.density_gs = None
    self.corr_fn_gs = None
    self.rdm1_gs = None

  def solve_eigenvalue(self, hamil):
    """Solve the eigenvalue problem for the provided Hamiltonian. The
       results are stored internally in energies and coeffs.

    Args
    ----
    hamil : scipy sparse CSR matrix
      The Hamiltonian matrix.
    """
    self.energies, self.coeffs = eigsh(hamil, k=1, which='SA')

  def calc_gs_density(self):
    """Calculate the local density from the ground-state wave function."""
    self.density_gs = np.zeros( self.nsites )
    for det, coeff in zip(self.dets, self.coeffs[:,0]):
      for orb in det:
        site = orb//self.nspin
        self.density_gs[site] += coeff**2

  def calc_corr_fn_gs(self):
    """Calculate the two-point density correlation function for the ground
       state wave function.
    """
    self.corr_fn_gs = np.zeros( (self.nsites, self.nsites) )
    for det, coeff in zip(self.dets, self.coeffs[:,0]):
      for orb_1 in det:
        site_1 = orb_1//self.nspin
        for orb_2 in det:
          site_2 = orb_2//self.nspin
          self.corr_fn_gs[site_1,site_2] += coeff**2

  def calc_rdm1_gs(self):
    """Calculate the one-body reduced density matrix for the ground-state
       wave function.
    """
    self.rdm1_gs = np.zeros( (self.nsites, self.nsites) )

    for det_1, coeff_1 in zip(self.dets, self.coeffs[:,0]):
      nel_1 = len(det_1)

      # Contributions for the diagonal of the density matrix:
      coeff_sq = coeff_1**2
      for orb_p in det_1:
        site_p = orb_p//self.nspin
        self.rdm1_gs[site_p,site_p] += coeff_sq

      # Contributions for the off-diagonal of the density matrix:
      for det_2, coeff_2 in zip(self.dets, self.coeffs[:,0]):
        if det_2 < det_1:
          continue

        nel_2 = len(det_2)
        # Find the list of sites involved in the excitation between
        # the two determinants.
        ind_ex_set = set(det_1).symmetric_difference(set(det_2))
        ind_ex = tuple(ind_ex_set)
        count_ex = len(ind_ex)
        if count_ex == 2 and nel_1 == nel_2:
          orb_p = ind_ex[0]
          orb_q = ind_ex[1]
          site_p = orb_p//self.nspin
          site_q = orb_q//self.nspin
          contrib = coeff_1*coeff_2
          self.rdm1_gs[site_p,site_q] += contrib
          self.rdm1_gs[site_q,site_p] += contrib

  def sign_flip_fraction(self):
    """Calculate the fraction of signs that are flipped in the ground
       state wave function, relative to a wave function with entirely
       fixed signs.

    Returns
    -------
    frac_flipped : float
      The fraction of opposite signs in the ground-state wave function.
      If there are the same number of positive and negative signs, then
      frac_flipped is equal to 1. If all coefficients are positive or
      all negative, then it is equal to 0.
    """

    wf = self.coeffs[:,0]
    num_positive = np.sum(wf > 0.0)
    num_negative = self.ndets - num_positive

    smaller_val = min(num_positive, num_negative)
    larger_val = max(num_positive, num_negative)

    frac_flipped = 2.0*smaller_val/self.ndets

    return frac_flipped

  def average_coeff(self):
    """Calculate the average sign of the ground-state wave function.

    Returns
    -------
    av_coeff : float
      The average values of the coefficients in the ground-state wave
      function.
    """
    wf = self.coeffs[:,0]
    av_coeff = np.mean(wf)

    return av_coeff

  def print_energies(self):
    """Print the list of energies to screen."""
    print("Energies:")
    for i, energy in enumerate(self.energies):
      print('{:6d}  {: .8e}'.format(i, energy))

  def print_gs_density(self):
    """Print the ground-state local density. Also print the total sum of
       the densities, which should be the total number of particles.
    """
    print("Ground-state local density:")
    total = 0.0
    for i, density in enumerate(self.density_gs):
      total += density
      print('{:6d}  {: .8e}'.format(i, density))
    print('Summation: {:6.2f}'.format(total))

  def print_corr_fn_gs(self):
    """Print the two-point density correlation function of the ground
       state wave function.
    """
    print("Ground-state two-point correlation function:")
    for i in range(self.nsites):
      for j in range(self.nsites):
        print('({:6d}, {:6d})  {: .8e}'.format(i, j, self.corr_fn_gs[i,j]))

  def print_rdm1_gs(self):
    """Print the one-body reduced density matrix for the ground-state
       wave function.
    """
    print("Ground-state one-body reduced density matrix:")
    for i in range(self.nsites):
      for j in range(self.nsites):
        print('({:6d}, {:6d})  {: .8e}'.format(i, j, self.rdm1_gs[i,j]))

  def print_ground(self):
    """Print the ground state wave function coefficients (and corresponding
       occupation lists for each determinant) to the screen.
    """
    print("Ground-state wave function:")
    for i, occ_i, coeff_i in zip(count(), self.dets, self.coeffs[:,0]):
      if abs(coeff_i) > 1.e-10:
        print('{:6d}  {}  {: .8e}'.format(i, occ_i, coeff_i))
      else:
        print('{:6d}  {}  {: .8e}'.format(i, occ_i, 0.0))

  def select_single_random_config(self, cum_coeffs):
    """Choose a random configuration from the distribution defined
       by cum_coeffs.

    Args
    ----
    cum_coeffs : list of floats
      The cumulative values of the wave function amplitudes squared.
    """
    cum_total = cum_coeffs[-1]

    rand = random.uniform(0.0, cum_total)

    det_chosen_ind = None
    for i, cum_total in enumerate(cum_coeffs):
      if rand < cum_total:
        det_chosen_ind = i
        break

    if det_chosen_ind is None:
      det_chosen_ind = self.ndets-1

    return det_chosen_ind

  def select_random_configs(self, n):
    """Select n configurations from the distribution of wave function
       amplitudes squared.

    Args
    ----
    n : int
      The number of unique configurations to be selected.
    """

    if n > self.ndets:
      raise ValueError('Number of unique configurations requested is '
          'larger than the total number of configurations.')

    # Generate list of cumulative squared coefficients
    cum_coeffs = []
    cum_total = 0.0
    for coeff in self.coeffs[:,0]:
      cum_total += coeff**2
      cum_coeffs.append(cum_total)

    inds_chosen = []

    n_chosen = 0
    while n_chosen < n:
      new_ind = self.select_single_random_config(cum_coeffs)
      if new_ind not in inds_chosen:
        inds_chosen.append(new_ind)
        n_chosen += 1

    assert n_chosen == n

    return inds_chosen
