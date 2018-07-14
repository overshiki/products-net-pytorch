
'''
IPNN MODEL TRANSLATED FROM TENSORFLOW CONTERPARTS

STEPS:
	1. UNDERTAKE EMBEDDING AT EACH FIELD
'''
import torch
from .torch_utils.basic_modules import Split
from torch.autograd import Variable

class linear_field(torch.nn.Module):
	'''
	input: b*N
	output: b*num_inputs*embed_size (batch * num * k)
	'''
	def __init__(self, fields, embedding_dim):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.fields = fields
		self.split = Split(fields)
		self.embedding_layer = []
		for num in fields:
			self.embedding_layer.append(torch.nn.Linear(num, embedding_dim).cuda())

	def forward(self, x):
		LIST = self.split(x)
		re_LIST = []
		for index, item in enumerate(LIST):
			re_LIST.append(self.embedding_layer[index](item))
		x = torch.stack(re_LIST)
		return x.transpose(0,1)



class embedding_field(torch.nn.Module):
	'''
	input: b*N
	output: b*num_inputs*embed_size (batch * num * k)
	'''
	def __init__(self, fields, embedding_dim):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.fields = fields
		self.split = Split(fields)
		self.embedding_layer = []
		for num in fields:
			self.embedding_layer.append(torch.nn.EmbeddingBag(num, embedding_dim).cuda())

	def forward(self, x):
		LIST = self.split(x)
		re_LIST = []
		for index, item in enumerate(LIST):
			re_LIST.append(self.embedding_layer[index](item))
		x = torch.stack(re_LIST)
		return x.transpose(0,1)



class IPNN(torch.nn.Module):
	def __init__(self, fields, embedding_method=linear_field, embedding_dim=10, layer_size1=500, layer_size2=2):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_inputs = len(fields)
		self.denseLayer_dim = int(self.num_inputs*(self.num_inputs-1)/2)+self.num_inputs*self.embedding_dim
		self.layer_size1 = layer_size1
		self.layer_size2 = layer_size2

		self.embedding = embedding_method(fields, embedding_dim)
		self.inner_p = inner_products(self.num_inputs, self.embedding_dim)

		self.layer1 = self.ensemble(self.denseLayer_dim, self.layer_size1)
		self.layer2 = self.ensemble(self.layer_size1, self.layer_size2)

		self.sigmoid = torch.nn.Sigmoid()


	def forward(self, x):
		x = self.embedding(x)
		x = self.inner_p(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.sigmoid(x)
		return x

	def ensemble(self, in_dim, out_dim):
		return torch.nn.Sequential(
			torch.nn.Linear(in_dim, out_dim),
			torch.nn.ReLU(),
			torch.nn.Dropout()
			)


class inner_products(torch.nn.Module):
	def __init__(self, num_inputs, embedding_dim):
		super().__init__()
		self.num_inputs = num_inputs
		self.embedding_dim = embedding_dim

	def forward(self, x):
		num_inputs = x.data.shape[1]
		embedding_dim = x.data.shape[2]
		row, col = [], []
		for i in range(self.num_inputs-1):
			for j in range(i+1, self.num_inputs):
				row.append(i)
				col.append(j)

		row, col = torch.LongTensor(row), torch.LongTensor(col)
		row, col = Variable(x.data.new(row.shape).copy_(row).long()), Variable(x.data.new(col.shape).copy_(col).long())

		# batch * pair * k
		p = torch.index_select(x, 1, row)
		q = torch.index_select(x, 1, col)

		ip = torch.sum(p*q, -1)

		x = x.contiguous().view(-1, self.num_inputs*self.embedding_dim)
		x = torch.cat([x, ip], 1)

		return x






if __name__ == '__main__':
	from utils import data_loader, field_size
	_loader, column_path = data_loader(refresh=False, minibatch=200, shuffle=True)


	fields = field_size(column_path)
	embedding_dim = 10
	model = IPNN(fields, embedding_dim).cuda()

	_loader.create_iter()
	for tensor, labels in _loader._iter:
		tensor = tensor.long()
		print(tensor.shape, labels.shape, tensor.type(), labels.type())
		outputs = model(Variable(tensor))
		print(outputs.data.type(), outputs.data.shape)
		break
