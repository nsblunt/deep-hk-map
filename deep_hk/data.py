from hamiltonian import SpinlessHubbard
from wave_function import WaveFunction
import torch
import ast
import csv

class Data:

  def __init__(self, system, ntrain, ntest, nbatch):
    self.system = system
    self.ntrain = ntrain
    self.ntest = ntest
    self.nbatch = nbatch

    self.inputs_train = None
    self.labels_train = None

    self.inputs_test = None
    self.labels_test = None

  def gen_training_data(self):
    system = self.system
    self.inputs_train = torch.zeros(self.ntrain, system.nsites, dtype=torch.float)
    self.labels_train = torch.zeros(self.ntrain, 1)

    for i in range(self.ntrain):
      V = system.gen_rand_potential()
      system.add_potential_to_hamil(V)
      
      wf = WaveFunction(
        nsites=system.nsites,
        dets=system.dets
      )

      # find and print eigenvectors and energies
      wf.solve_eigenvalue(system.hamil)
      # find and print properties
      wf.calc_gs_density()

      #torch_density = torch.from_numpy(wf.density_gs)
      torch_density = torch.from_numpy(V)
      self.inputs_train[i,:] = torch_density
      self.labels_train[i,0] = wf.energies[0]

      #sample = {'density': list(wf.density_gs), 'energy': wf.energies[0]}
      #sample = {'density': torch_density, 'energy': wf.energies[0]}
      #data.append(sample)

  def gen_test_data(self):
    system = self.system
    self.inputs_test = torch.zeros(self.ntest, system.nsites, dtype=torch.float)
    self.labels_test = torch.zeros(self.ntest, 1)

    for i in range(self.ntest):
      V = system.gen_rand_potential()
      system.add_potential_to_hamil(V)
      
      wf = WaveFunction(
        nsites=system.nsites,
        dets=system.dets
      )

      # find and print eigenvectors and energies
      wf.solve_eigenvalue(system.hamil)
      # find and print properties
      wf.calc_gs_density()

      torch_density = torch.from_numpy(wf.density_gs)
      torch_density = torch.from_numpy(V)
      self.inputs_test[i,:] = torch_density
      self.labels_test[i,0] = wf.energies[0]

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
