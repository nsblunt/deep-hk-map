from system import SpinlessHubbard
from wave_function import WaveFunction
from torch.utils.data import Dataset
import torch
import ast
import csv

class Data(Dataset):

  def __init__(self, system, ninput, ndata, input_type='potential'):
    self.sys = system
    self.ninput = ninput
    self.ndata = ndata
    self.input_type = input_type

    self.inputs = None
    self.labels = None

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.inputs[index], self.labels[index]

  def generate(self):
    sys = self.sys
    self.inputs = torch.zeros(self.ndata, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata, 1)

    for i in range(self.ndata):
      V = sys.gen_rand_potential()
      sys.add_potential_to_hamil(V)
      
      wf = WaveFunction(
        nsites=sys.nsites,
        dets=sys.dets
      )

      wf.solve_eigenvalue(sys.hamil)

      if self.input_type == 'potential':
        self.inputs[i,:] = torch.from_numpy(V)
      elif self.input_type == 'density':
        wf.calc_gs_density()
        self.inputs[i,:] = torch.from_numpy(wf.density_gs)
      elif self.input_type == '1-rdm':
        wf.calc_rdm1_gs()
        self.inputs[i,:] = torch.from_numpy(wf.rdm1_gs.flatten())

      self.labels[i,0] = wf.energies[0]

      #sample = {'density': list(wf.density_gs), 'energy': wf.energies[0]}
      #sample = {'density': torch_density, 'energy': wf.energies[0]}
      #data.append(sample)

  def print_data(self, data):
    keys = data[0].keys()
    with open('data.csv', 'w', newline='') as csv_file:
      dict_writer = csv.DictWriter(csv_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(data)

  def read_data(self):
    data_in = []
    with open('data.csv', 'r', newline='') as csv_file:
      dict_reader = csv.DictReader(csv_file)
      for row in dict_reader:
        density = ast.literal_eval(row['density'])
        energy = float(row['energy'])
        new_dict = {'density': density, 'energy': energy}
        data_in.append(new_dict)
