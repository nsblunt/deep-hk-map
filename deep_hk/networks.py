import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

def train(net, data_train, data_test, criterion, optimizer, nepochs, batch_size):
  print('# 1. Epoch' + 6*' ' + '2. Loss')

  data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False,
                            num_workers=0)

  # train network
  for epoch in range(nepochs):
    total_loss = 0.0

    for batch_inputs, batch_labels in data_loader:
      optimizer.zero_grad()
      batch_outputs = net(batch_inputs)
      loss = criterion(batch_outputs, batch_labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    print('{:10d}   {:10.6f}'.format(epoch, total_loss), flush=True)
  print(flush=True)

def print_net_accuracy(net, data_train, data_test, criterion):
  # apply network to the training data
  outputs_train = net(data_train.inputs)
  train_loss = criterion(outputs_train, data_train.labels)
  print('Training loss: {:.5f}'.format(train_loss))

  # apply network to the test data
  outputs_test = net(data_test.inputs)
  test_loss = criterion(outputs_test, data_test.labels)
  print('Test loss: {:.5f}\n'.format(test_loss))

  # print the exact labels against the predicted labels for the test data
  print('# 1. Iter.' + 8*' ' + '2. Exact' + 6*' ' + '3. Predicted')
  for i in range(data_test.ndata):
    print('{:8d}   {: .8e}   {: .8e}'.format(i,
        float(data_test.labels[i][0]), float(outputs_test[i][0])))
