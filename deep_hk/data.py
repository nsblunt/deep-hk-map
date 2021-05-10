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
               ndata,
               input_type='potential',
               output_type='energy',
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
    ndata : int
      The number of potentials to be read or generated.
    input_type : string
      String specifying what object is passed into the network.
    output_type : string
      String specifying what object is passed out of the network.
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
    self.ndata = ndata
    self.input_type = input_type
    self.output_type = output_type

    self.inputs = None
    self.labels = None
    self.potentials = None
    self.energies = None

    self.print_timing = print_timing

    # The number of data points for each potential generated.
    # This is usually 1, but will be larger if passing in each
    # configuration to the network also, as there are many
    # configurations contributing to each wave function.
    self.ndata_per_potential = 1

    self.all_configs = all_configs
    self.nconfigs_per_pot = nconfigs_per_pot

    if input_type == 'potential' or input_type == 'density':
      self.ninput = system.nsites
    elif input_type == '1-rdm':
      self.ninput = system.nsites**2
    elif input_type == 'potential_and_config':
      self.ninput = system.nsites + system.norbs
      if all_configs:
        self.ndata_per_potential = system.ndets
      else:
        self.ndata_per_potential = nconfigs_per_pot
    elif input_type == 'potential_and_occ_str':
      self.ninput = system.nsites + system.nparticles
      self.ndata_per_potential = system.ndets
    elif input_type == 'potential_and_det_ind':
      self.ninput = system.nsites + 1
      self.ndata_per_potential = system.ndets

    self.ndata_tot = self.ndata * self.ndata_per_potential

    if output_type == 'energy':
      self.noutput = 1
    elif output_type == 'wave_function':
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
    det_ind = None

    # Generate the configurations from the determinants stored,
    # which will be the same every for every potential applied.
    if self.input_type == 'potential_and_config':
      system.generate_configs()

    t1 = time.perf_counter()

    for i in range(self.ndata):
      V = system.gen_rand_potential(
          const_potential_sum,
          potential_sum_val)

      system.add_potential_to_hamil(V)
      
      wf = WaveFunction(
          nsites=system.nsites,
          nspin=system.nspin,
          dets=system.dets)

      wf.solve_eigenvalue(system.hamil)

      self.potentials.append(V)
      self.energies.append(wf.energies[0])

      if self.input_type == 'potential':
        self.inputs[i,:] = torch.from_numpy(V)
      elif self.input_type == 'density':
        wf.calc_gs_density()
        self.inputs[i,:] = torch.from_numpy(wf.density_gs)
      elif self.input_type == '1-rdm':
        wf.calc_rdm1_gs()
        self.inputs[i,:] = torch.from_numpy(wf.rdm1_gs.flatten())
      elif self.input_type == 'potential_and_config':
        if self.all_configs:
          for j in range(system.ndets):
            ind = i*system.ndets + j
            config = system.configs[j]
            self.inputs[ind,0:system.nsites] = torch.from_numpy(V)
            self.inputs[ind,system.nsites:] = torch.from_numpy(config)
        else:
          inds_chosen = wf.select_random_configs(self.nconfigs_per_pot)
          for j, det_ind in enumerate(inds_chosen):
            ind = i*self.nconfigs_per_pot + j
            config = system.configs[det_ind]
            self.inputs[ind,0:system.nsites] = torch.from_numpy(V)
            self.inputs[ind,system.nsites:] = torch.from_numpy(config)
      elif self.input_type == 'potential_and_occ_str':
        if self.all_configs:
          for j in range(system.ndets):
            ind = i*system.ndets + j
            det = torch.from_numpy(np.asarray(system.dets[j]))
            self.inputs[ind,0:system.nsites] = torch.from_numpy(V)
            self.inputs[ind,system.nsites:] = det
      elif self.input_type == 'potential_and_det_ind':
        if self.all_configs:
          for j in range(system.ndets):
            ind = i*system.ndets + j
            self.inputs[ind,0:system.nsites] = torch.from_numpy(V)
            self.inputs[ind,system.nsites:] = j

      if self.output_type == 'energy':
        self.labels[i,:] = wf.energies[0]
      elif self.output_type == 'wave_function':
        self.labels[i,:] = torch.from_numpy(wf.coeffs[:,0])
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
        if self.all_configs:
          for j in range(system.ndets):
            ind = i*system.ndets + j
            self.labels[ind,0] = wf.coeffs[j,0] / np.sign(wf.coeffs[0,0])
        else:
          for j, det_ind in enumerate(inds_chosen):
            ind = i*self.nconfigs_per_pot + j
            self.labels[ind,0] = wf.coeffs[det_ind,0] / np.sign(wf.coeffs[0,0])

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
