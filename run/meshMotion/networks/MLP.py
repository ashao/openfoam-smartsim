import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
  def __init__(self, input_size, output_size, activation_fn, num_layers, layer_width):
      super(MLP, self).__init__()

      layers = []
      layers.append(nn.Linear(input_size, layer_width))
      layers.append(activation_fn)

      for _ in range(num_layers - 2):
          layers.append(nn.Linear(layer_width, layer_width))
          layers.append(activation_fn)

      layers.append(nn.Linear(layer_width, output_size))
      self.layers = nn.Sequential(*layers)

  def forward(self, x):
      return self.layers(x)

class MLPTrainer:
  def __init__(self, model, X, y_true, radius_power, lr=1e-3, loss_stop=5e-5):
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr=lr)
    self.loss_stop = loss_stop
    self.loss_value = None
    self.radius_power = radius_power
    self.X = X
    self.y_true = y_true

  def loss(self):
    inner = self.y_true != 0.
    center = torch.mean(self.X[inner], dim=0)
    scaled_dist = torch.sqrt(torch.sum((self.X-center)**2, dim=1))**self.radius_power
    wts = scaled_dist/torch.sum(scaled_dist)

    y_pred = self.model(self.X)
    return torch.sum(wts*torch.sum(torch.sqrt((self.y_true-y_pred)**2), dim=1))

  def training_step(self):
     self.optimizer.zero_grad()
     loss_value = self.loss()
     self.loss_value = loss_value
     loss_value.backward()
     self.optimizer.step()

     return loss_value, self.model

  def converged(self):
     if self.loss_value.item() < self.loss_stop:
        return True
     return False
