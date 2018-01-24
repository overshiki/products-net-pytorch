
import torch
from torch.autograd import Variable

class LR(torch.nn.Module):
	def __init__(self, input_dim, output_dim=2):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.linear(x)
		x = self.sigmoid(x)
		return x
