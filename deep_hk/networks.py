import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearNet(nn.Module):
  def __init__(self, layers_list):
    super(LinearNet, self).__init__()
    self.layers = nn.ModuleList(layers_list)

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    # output layer
    x = self.layers[-1](x)
    return x

def create_linear_layers(num_input, num_hidden, num_output):
  layers_list = []
  # input layer:
  layers_list.append( nn.Linear(num_input, num_hidden[0]) )
  # hidden layers:
  for i in range(1, len(num_hidden)):
    layers_list.append( nn.Linear(num_hidden[i-1], num_hidden[i]) )
  # output layer:
  layers_list.append( nn.Linear(num_hidden[-1], num_output) )

  return layers_list

def train(net, data, nepochs):
  ntrain = data.ntrain
  nbatch = data.nbatch
  ntest = data.ntest

  criterion = nn.L1Loss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  # train network
  for epoch in range(nepochs):
    total_loss = 0.0
    for i in range(int(ntrain/nbatch)):
      inputs = data.inputs_train[nbatch*i:nbatch*(i+1)-1, :]
      labels = data.labels_train[nbatch*i:nbatch*(i+1)-1, :]

      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    print('%d %.5f' % (epoch, total_loss), flush=True)

  # final network applied to the training data
  outputs_train = net(data.inputs_train)
  train_loss = criterion(outputs_train, data.labels_train)
  print('Final training loss: %.5f' % train_loss)

  # final network applied to the test data
  outputs_test = net(data.inputs_test)
  test_loss = criterion(outputs_test, data.labels_test)
  print('Final test loss: %.5f' % test_loss)

  # print the exact values against the predicted values for the
  # test data
  print('# 1. iteration  2. exact  3. predicted')
  for i in range(ntest):
    print(i, float(data.labels_test[i][0]), float(outputs_test[i][0]))
