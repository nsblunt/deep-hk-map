from system import SpinlessHubbard
from wave_function import WaveFunction
from torch.utils.data import Dataset
import torch
import ast
import csv

class Data(Dataset):

  def __init__(self, system, ntrain, ntest, nbatch):
    self.sys = system
    self.ntrain = ntrain
    self.ntest = ntest
    self.nbatch = nbatch

    self.inputs_train = None
    self.labels_train = None

    self.inputs_test = None
    self.labels_test = None

  def __len__(self):
    return len(self.labels_train)

  def __getitem__(self, index):
    return self.inputs_train[index], self.labels_train[index]

  def generate(self, data_type, input_type='potential'):
    sys = self.sys
    inputs = torch.zeros(self.ntrain, sys.nsites, dtype=torch.float)
    labels = torch.zeros(self.ntrain, 1)

    if data_type == 'train':
      self.inputs_train = inputs
      self.labels_train = labels
    elif data_type == 'test':
      self.inputs_test = inputs
      self.labels_test = labels

    for i in range(self.ntrain):
      V = sys.gen_rand_potential()
      sys.add_potential_to_hamil(V)
      
      wf = WaveFunction(
        nsites=sys.nsites,
        dets=sys.dets
      )

      # find and print eigenvectors and energies
      wf.solve_eigenvalue(sys.hamil)

      if input_type == 'potential':
        inputs[i,:] = torch.from_numpy(V)
      elif input_type == 'density':
        wf.calc_gs_density()
        inputs[i,:] = torch.from_numpy(wf.density_gs)

      labels[i,0] = wf.energies[0]

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
