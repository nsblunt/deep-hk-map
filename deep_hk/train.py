import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Infidelity(nn.Module):
  def __init__(self):
    super(Infidelity, self).__init__()

  def forward(self, outputs, labels):
    dot_products = torch.sum(outputs * labels, dim=1)
    loss = 1 - torch.mean(torch.abs(dot_products))
    return loss

def train(net,
          data_train,
          data_test,
          criterion,
          optimizer,
          nepochs,
          batch_size,
          save_net=False,
          save_root='./network',
          save_net_every=100):

  print('# 1. Epoch' + 6*' ' + '2. Loss')

  data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False,
                            num_workers=0)

  # train network
  for epoch in range(nepochs):
    total_loss = 0.0
    nbatches = 0

    for batch_inputs, batch_labels in data_loader:
      optimizer.zero_grad()
      batch_outputs = net(batch_inputs)
      loss = criterion(batch_outputs, batch_labels)
      #loss = new_loss(batch_outputs, batch_labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      nbatches += 1
    av_loss = total_loss/nbatches
    print('{:10d}   {:10.6f}'.format(epoch, av_loss), flush=True)

    if save_net:
      if epoch % save_net_every == save_net_every-1:
        nepochs_done = epoch+1
        filename = save_root + '_' + str(nepochs_done) + '.pt'
        net.save(filename)

  print(flush=True)

def calc_output_norms(outputs):
  norms = []
  for row in outputs:
    norms.append(torch.norm(row))
  return norms

def print_net_accuracy(net, data_train, data_test, criterion):
  # apply network to the training data
  outputs_train = net(data_train.inputs)
  train_loss = criterion(outputs_train, data_train.labels)
  print('Training loss: {:.5f}'.format(train_loss))

  # apply network to the test data
  outputs_test = net(data_test.inputs)
  test_loss = criterion(outputs_test, data_test.labels)
  print('Test loss: {:.5f}\n'.format(test_loss))

  output_norms = calc_output_norms(outputs_test)
  labels_norms = calc_output_norms(data_test.labels)

  # print the exact labels against the predicted labels for the test data
  print('# 1. Index' + 10*' ' + '2. Exact' + 6*' ' + '3. Predicted' + 5*' ' +
         '4. Exact_Norm' + 3*' ' + '5. Predicted_Norm')
  for i in range(data_test.ndata):
    print('{:10d}   {: .8e}   {: .8e}   {: .8e}     {: .8e}'.format(i,
        float(data_test.labels[i][0]), float(outputs_test[i][0]),
        float(labels_norms[i]), float(output_norms[i])))
