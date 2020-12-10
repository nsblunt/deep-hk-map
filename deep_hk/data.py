from wave_function import WaveFunction
from torch.utils.data import Dataset
import torch
import ast
import csv
import time

class Data(Dataset):
  """Object for generating and storing training/validation/test data."""

  def __init__(self,
               system,
               ndata,
               input_type='potential',
               output_type='energy',
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
      The number of data points to be read or generated.
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
    inputs : torch tensor of type torch.float and size (ndata, ninput)
      Tensor holding the input data points in its rows.
    outputs : torch tensor of type torch.float and size (ndata, noutput)
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

    if input_type == 'potential' or input_type == 'density':
      self.ninput = system.nsites
    elif input_type == '1-rdm':
      self.ninput = system.nsites**2

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
    self.inputs = torch.zeros(self.ndata, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata, self.noutput, dtype=torch.float)
    self.potentials = []
    self.energies = []

    t1 = time.perf_counter()

    for i in range(self.ndata):
      V = system.gen_rand_potential(
          const_potential_sum,
          potential_sum_val)

      system.add_potential_to_hamil(V)
      
      wf = WaveFunction(
          nsites=system.nsites,
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
    self.inputs = torch.zeros(self.ndata, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata, self.noutput, dtype=torch.float)
    with open(filename, 'r', newline='') as csv_file:
      reader = csv.reader(csv_file)
      # skip the header:
      next(reader, None)
      for i, row in enumerate(reader):
        self.inputs[i,:] = torch.FloatTensor(ast.literal_eval(row[0]))
        self.labels[i,:] = torch.FloatTensor(ast.literal_eval(row[1]))
