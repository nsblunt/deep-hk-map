import torch.nn as nn
import torch.nn.functional as F

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

def train(net, data, criterion, optimizer, nepochs):
  ntrain = data.ntrain
  nbatch = data.nbatch
  ntest = data.ntest

  print('# 1. Epoch' + 6*' ' + '2. Loss')

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
    print('{:10d}   {:10.6f}'.format(epoch, total_loss), flush=True)
  print(flush=True)

def print_net_accuracy(net, data, criterion):
  # apply network to the training data
  outputs_train = net(data.inputs_train)
  train_loss = criterion(outputs_train, data.labels_train)
  print('Training loss: {:.5f}'.format(train_loss))

  # apply network to the test data
  outputs_test = net(data.inputs_test)
  test_loss = criterion(outputs_test, data.labels_test)
  print('Test loss: {:.5f}\n'.format(test_loss))

  # print the exact labels against the predicted labels for the test data
  print('# 1. Iter.' + 8*' ' + '2. Exact' + 6*' ' + '3. Predicted')
  for i in range(data.ntest):
    print('{:8d}   {: .8e}   {: .8e}'.format(i,
        float(data.labels_test[i][0]), float(outputs_test[i][0])))
