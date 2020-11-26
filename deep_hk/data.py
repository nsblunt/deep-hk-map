from system import SpinlessHubbard
from wave_function import WaveFunction
from torch.utils.data import Dataset
import torch
import ast
import csv

class Data(Dataset):

  def __init__(self,
               system,
               ninput,
               noutput,
               ndata,
               input_type='potential',
               output_type='energy'):

    self.sys = system
    self.ninput = ninput
    self.noutput = noutput
    self.ndata = ndata
    self.input_type = input_type
    self.output_type = output_type

    self.inputs = None
    self.labels = None

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.inputs[index], self.labels[index]

  def generate(self):
    sys = self.sys
    self.inputs = torch.zeros(self.ndata, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata, self.noutput, dtype=torch.float)

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

      if self.output_type == 'energy':
        self.labels[i,:] = wf.energies[0]
      elif self.output_type == 'wave_function':
        self.labels[i,:] = torch.from_numpy(wf.coeffs[:,0])
      elif self.input_type == 'potential':
        self.labels[i,:] = torch.from_numpy(V)
      elif self.input_type == 'density':
        self.labels[i,:] = torch.from_numpy(wf.density_gs)
      elif self.input_type == '1-rdm':
        self.labels[i,:] = torch.from_numpy(wf.rdm1_gs.flatten())

  def save_csv(self, filename):
    with open(filename, 'w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['density','energy'])
      for i in range(self.ndata):
        writer.writerow([self.inputs[i,:].tolist(),self.labels[i,:].tolist()])

  def load_csv(self, filename):
    self.inputs = torch.zeros(self.ndata, self.ninput, dtype=torch.float)
    self.labels = torch.zeros(self.ndata, 1)
    with open(filename, 'r', newline='') as csv_file:
      reader = csv.reader(csv_file)
      # skip the header:
      next(reader, None)
      for i, row in enumerate(reader):
        self.inputs[i,:] = torch.FloatTensor(ast.literal_eval(row[0]))
        self.labels[i,:] = torch.FloatTensor(ast.literal_eval(row[1]))
