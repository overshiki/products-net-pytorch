'''
SOME BASIC FUNCTION AS MODULES
'''
import torch
from .basic import num2pivot
class Transpose(torch.nn.Module):
	def __init__(self, transpose_index):
		super().__init__()
		self.transpose_index = transpose_index

	def forward(self, x):
		x = x.transpose(self.transpose_index[0], self.transpose_index[1])
		return x

class View(torch.nn.Module):
	def __init__(self, view_size):
		super().__init__()
		self.view_size = view_size

	def forward(self, x):
		x = x.view(self.view_size)
		return x

class Print(torch.nn.Module):
	def __init__(self, _type):
		super().__init__()
		self.type = _type

	def forward(self, x):
		if(self.type=="size"):
			print(x.size())
		elif(self.type=="type"):
			print(x.data.type())
		elif(self.type=="verbose"):
			print("size: {}, type: {}, dim: {}".format(x.size(), x.data.type(), x.data.dim()))
		return x

class Split(torch.nn.Module):
	def __init__(self, fields):
		super().__init__()
		self.fields = [int(x) for x in fields]
		self.pivot = [int(x) for x in num2pivot(fields)]
		self.end = sum(fields)

	def forward(self, x):
		LIST = []
		for i in range(len(self.fields)-1):
			start = self.pivot[i]
			end = self.pivot[i+1]
			LIST.append(x[:,start:end].contiguous())
		start = self.pivot[-1]
		end = self.end 
		LIST.append(x[:,start:end].contiguous())
		return LIST

import torch.nn as nn

class ListModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


if __name__ == '__main__':
	from torch.autograd import Variable
	# a = Variable(torch.ones(10,1,1000,1000)).long()
	# print(a[:,:,10:100,:])

	fields = [int(x) for x in range(10)]

	_len = sum(fields)

	a = Variable(torch.ones(10, _len)).long()
	model = Split(fields)
	ouputs = model(a)

	print([x.shape for x in ouputs])

