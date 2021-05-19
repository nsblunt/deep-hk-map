"""Class for generating and storing data sets to use during learning."""

from deep_hk.wave_function import WaveFunction
from torch.utils.data import Dataset
import numpy as np
import torch
import ast
import csv
import random
import time

class Data(Dataset):
  """Object for generating and storing training/validation/test data."""

  def __init__(self,
               system,
               npot,
               input_type='potential',
               output_type='energy',
               nonlocal_pot=False,
               remove_sign_problem=False,
               all_configs=True,
               nconfigs_per_pot=1,
               load=False,
               save=True,
               path='data.csv',
               const_potential_sum=False,
               potential_sum_val=0.0,
               print_timing=False):
    """Initialises an object for storing data to learn from.

    Args
    ----
    system : SpinlessHubbard object
      The definition of the lattice model used.
    npot : int
      The number of potentials to be read or generated.
    input_type : string
      String specifying what object is passed into the network.
    output_type : string
      String specifying what object is passed out of the network.
    nonlocal_pot : bool
      If True then apply a non-local potential. If False then a local
      potential is used.
    remove_sign_problem : bool
      If True then after generating the Hamiltonian, make all of its
      off-diagonal elements negative, which removes any sign problem.
    all_configs : bool
      When predicting individual coefficients as output, if True
      then every configuration is used as a data point for each
      potential. If False then a random selection is taken.
    nconfigs_per_pot : int
      When predicting individual coefficients as output, if all_configs
      is False then this specifies how many random configurations to
      use for each potential generated.
    load : bool
      If True, then the data is read from a file.
    save : bool
      If True, then the generated/loaded data is saved to a file.
    path : string
      Path specifying where data is loaded from or saved to.
    const_potential_sum : bool
      If True, then generated potentials will be shifted so that the
      total summmed potential is a constant value (potential_sum_val).
    potential_sum_val : float
      If const_potential_sum is True, then this is the total summed
      value that is enforced.
    print_timing : bool
      If true, print timing information.

    Other attributes
    ----------------
    ndata_tot : int
      The total number of data points in the data set. Usually this is
      just equal to npot. The exception is where we are predicting
      individual wave function coefficients, in which case we may have
      multiple configurations as data points per potential.
    ninput : int
      The number of values passed into the network (for each data point).
    noutput : int
      The number of values passed out of the network.
    inputs : torch tensor of type torch.float and size (ndata_tot, ninput)
      Tensor holding the input data points in its rows.
    outputs : torch tensor of type torch.float and size (ndata_tot, noutput)
      Tensor holding the predicted labels in its rows.
    potentials : list of numpy ndarrays, each of size (system.nsites)
      Holds the potentials applied to generate each data point.
    energies : list of floats
      Holds the ground-state energies for each potential applied
    """

    self.system = system
    self.npot = npot
    self.input_type = input_type
    self.output_type = output_type

    self.nonlocal_pot = nonlocal_pot
    self.remove_sign_problem = remove_sign_problem

    self.inputs = None
    self.labels = None
    self.potentials = None
    self.energies = None

    self.print_timing = print_timing

    # The number of data points for each potential generated.
    # This is usually 1, but will be larger if passing in each
    # configuration to the network also, as there are many
    # configurations contributing to each wave function.
    self.nconfigs_per_pot = 1

    self.all_configs = all_configs

    # The number of inputs that define each potential
    if self.nonlocal_pot:
      if self.input_type == 'potential_compressed':
        self.npot_vals = int(system.nsites*(system.nsites+1)/2)
      else:
        self.npot_vals = system.nsites**2
    else:
      self.npot_vals = system.nsites

    self.coeff_out = False
    if output_type == 'coeff':
      self.coeff_out = True
      # Do we use all configurations as data points, or a sample?
      if all_configs:
        self.nconfigs_per_pot = system.ndets
      else:
        self.nconfigs_per_pot = nconfigs_per_pot

    self.ndata_tot = self.npot * self.nconfigs_per_pot

    if 'potential' in input_type:
      self.ninput = self.npot_vals
    if 'density' in input_type:
      self.ninput = system.nsites
    elif '1-rdm' in input_type:
      self.ninput = system.nsites**2

    if 'config' in input_type:
      self.ninput += system.norbs
    elif 'occ_str' in input_type:
      self.ninput += system.nparticles
    elif 'det_ind' in input_type:
      self.ninput += 1

    if output_type == 'energy':
      self.noutput = 1
    elif 'wave_function' in output_type:
      self.noutput = system.ndets
    elif output_type == 'potential' or output_type == 'density':
      self.noutput = system.nsites
    elif output_type == '1-rdm':
      self.noutput = system.nsites**2
    elif output_type == 'corr_fn':
      self.noutput = system.nsites**2
    elif output_type == 'coeff':
      self.noutput = 1

    if load:
      self.load_csv(path)
    else:
      self.generate(
          const_potential_sum=const_potential_sum,
          potential_sum_val=potential_sum_val)

    if save:
      self.save_csv(path)

  def __len__(self):
    """Return the number of data points."""
    return len(self.labels)

  def __getitem__(self, index):
    """Return the input labelled by index, and the associated label."""
    return self.inputs[index], self.labels[index]

  def generate(self, const_potential_sum=False, potential_sum_val=0.0):
    """Generate all data.

    Args
    ----
    const_potential_sum : bool
      If True, then generated potentials will be shifted so that the
      total summed potential is a constant value (potential_sum_val).
    potential_sum_val : float
      If const_potential_sum is True, then this is the total summed
      value that is enforced.
    """
    system = self.system
    self.inputs = torch.zeros(self.ndata_tot, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata_tot, self.noutput, dtype=torch.float)
    self.potentials = []
    self.energies = []

    # Generate the configurations from the determinants stored,
    # which will be the same every for every potential applied.
    if 'config' in self.input_type:
      system.generate_configs()

    t1 = time.perf_counter()

    tot_frac_sign_flip = 0.0
    tot_av_coeff = 0.0

    # Loop over randomly-generated potentials
    for i in range(self.npot):

      if self.nonlocal_pot:
        V = system.gen_rand_nonlocal_potential()
        system.add_nonlocal_potential_to_hamil(V)
      else:
        V = system.gen_rand_potential(
            const_potential_sum,
            potential_sum_val)
        system.add_potential_to_hamil(V)

      if self.remove_sign_problem:
        system.remove_sign_problem()
      
      wf = WaveFunction(
          nsites=system.nsites,
          nspin=system.nspin,
          dets=system.dets)

      wf.solve_eigenvalue(system.hamil)

      frac_sign_flip = wf.sign_flip_fraction()
      tot_frac_sign_flip += frac_sign_flip
      av_coeff = wf.average_coeff()
      tot_av_coeff += abs(av_coeff)

      self.potentials.append(V)
      self.energies.append(wf.energies[0])

      if self.input_type == 'potential' or self.input_type == 'potential_compressed':
        if self.nonlocal_pot:
          if self.input_type == 'potential_compressed':
            V_compressed = system.compress_potential(V)
            self.inputs[i,:] = torch.from_numpy(V_compressed)
          else:
            self.inputs[i,:] = torch.from_numpy(V.flatten())
        else:
          self.inputs[i,:] = torch.from_numpy(V)
      elif self.input_type == 'density':
        wf.calc_gs_density()
        self.inputs[i,:] = torch.from_numpy(wf.density_gs)
      elif self.input_type == '1-rdm':
        wf.calc_rdm1_gs()
        self.inputs[i,:] = torch.from_numpy(wf.rdm1_gs.flatten())
      elif self.coeff_out:
        # If outputting a wave function coefficient
        if 'potential' in self.input_type:
          inp_array = torch.from_numpy(V)
          inp_length = system.npot_vals
        if 'density' in self.input_type:
          wf.calc_gs_density()
          inp_array = torch.from_numpy(wf.density_gs)
          inp_length = system.nsites
        if '1-rdm' in self.input_type:
          wf.calc_rdm1_gs()
          inp_array = torch.from_numpy(wf.rdm1_gs.flatten())
          inp_length = system.nsites**2

        # Generate the indices of the determinants to be used
        if self.all_configs:
          det_inds = list(range(self.nconfigs_per_pot))
        else:
          det_inds = wf.select_random_configs(self.nconfigs_per_pot)

        for j, det_ind in enumerate(det_inds):
          ind = i*self.nconfigs_per_pot + j
          self.inputs[ind,0:inp_length] = inp_array
          if 'config' in self.input_type:
            config = system.configs[det_ind]
            self.inputs[ind,inp_length:] = torch.from_numpy(config)
          elif 'occ_str' in self.input_type:
            self.inputs[ind,inp_length:] = torch.from_numpy(np.asarray(system.dets[det_ind]))
          elif 'det_ind' in self.input_type:
            self.inputs[ind,inp_length:] = det_ind

      if self.output_type == 'energy':
        self.labels[i,:] = wf.energies[0]
      elif self.output_type == 'wave_function':
        self.labels[i,:] = torch.from_numpy(wf.coeffs[:,0])
      elif self.output_type == 'abs_wave_function':
        self.labels[i,:] = torch.from_numpy(abs(wf.coeffs[:,0]))
      elif self.output_type == 'potential':
        self.labels[i,:] = torch.from_numpy(V)
      elif self.output_type == 'density':
        wf.calc_gs_density()
        self.labels[i,:] = torch.from_numpy(wf.density_gs)
      elif self.output_type == '1-rdm':
        wf.calc_rdm1_gs()
        self.labels[i,:] = torch.from_numpy(wf.rdm1_gs.flatten())
      elif self.output_type == 'corr_fn':
        wf.calc_corr_fn_gs()
        self.labels[i,:] = torch.from_numpy(wf.corr_fn_gs.flatten())
      elif self.output_type == 'coeff':
        for j, det_ind in enumerate(det_inds):
          ind = i*self.nconfigs_per_pot + j
          self.labels[ind,0] = wf.coeffs[det_ind,0] / np.sign(wf.coeffs[0,0])

    tot_frac_sign_flip /= self.npot
    tot_av_coeff /= self.npot

    print('Sign flip fraction: ', tot_frac_sign_flip)
    print('Average coefficient: ', tot_av_coeff)

    t2 = time.perf_counter()

    if self.print_timing:
      print('Time to generate data: {:.6f}\n'.format(t2-t1))

  def save_csv(self, filename):
    """Save the data to a CSV file.

    Args
    ----
    filename : string
      The name of the file where data will be saved.
    """
    with open(filename, 'w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow([self.input_type, self.output_type])
      for input, label in zip(self.inputs, self.labels):
        writer.writerow([input.tolist(),label.tolist()])

  def load_csv(self, filename):
    """Load the data from a CSV file.

    Args
    ----
    filename : string
      The name of the file where data will be loaded from.
    """
    self.inputs = torch.zeros(self.ndata_tot, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata_tot, self.noutput, dtype=torch.float)
    with open(filename, 'r', newline='') as csv_file:
      reader = csv.reader(csv_file)
      # skip the header:
      next(reader, None)
      for i, row in enumerate(reader):
        self.inputs[i,:] = torch.FloatTensor(ast.literal_eval(row[0]))
        self.labels[i,:] = torch.FloatTensor(ast.literal_eval(row[1]))


class MultipleDatasets(Data):
  """Object for storing multiple different datasets, which can be
     accessed simultanesouly by a DataLoader object."""

  def __init__(self, datasets):
    self.datasets = datasets
    self.ndatasets = len(datasets)
    self.ndata_tot = sum(len(d) for d in self.datasets)

    # Displacements of each data set within the full list
    self.displs = [0] * self.ndatasets
    for i in range(1, self.ndatasets):
      self.displs[i] = self.displs[i-1] + self.datasets[i-1].ndata_tot

    # Dictionary used to define the ordering of data from the
    # various datasets.
    self.indices = {}

    # Use a random ordering, using random.shuffle.
    self.indices = self.create_random_ordering()

  def __len__(self):
    """Return the number of data points."""
    return self.ndata_tot

  def __getitem__(self, index):
    """Return the input labelled by index, and the associated label."""
    set_ind, data_ind = self.indices[index]
    return self.datasets[set_ind].inputs[data_ind], \
           self.datasets[set_ind].labels[data_ind]

  def create_random_ordering(self):
    indices = {}
    data_inds = [i for i in range(self.ndata_tot)]
    random.shuffle(data_inds)
    for i in range(self.ndata_tot):
      n = data_inds[i]
      set_ind = find_pos(n, self.displs)
      data_ind = n - self.displs[set_ind]
      indices[i] = (set_ind, data_ind)
    return indices

def find_pos(ind, displs):
  """Find the position of ind relative to the displacement values stored
     in the displs list. This is used by MultipleDatasets objects; here
     displs[n] refers to the index of the first data point from dataset n.

    Args
    ----
    ind : int
      A position index.
    displs : list of int
      An increasing list of integers, representing the first position
      of each piece of data from n'th data set, where n is an index
      of displs.
  """

  for i in range(1, len(displs)):
    if ind < displs[i]:
      return i-1

  return len(displs) - 1
