

'''
basic functions for pytorch training 
'''
import numpy, torch
import numpy as np
import os, re
from timeit import default_timer
from .basic import childPath

from timeit import default_timer as timer
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import *
from collections import OrderedDict



class general_loader():
	'''
	prepare iterator with given data format and given rule
	'''
	def __init__(self, data, labels, indices, data_rule, indices_rule, shuffle=True, minibatch=100):
		self.data = data 
		self.data_rule = data_rule
		self.labels = labels

		self.indices_rule = indices_rule
		self.indices = indices

		self.shuffle = shuffle
		self.minibatch = minibatch

		self.indices_list = indices_rule(indices, shuffle=shuffle, minibatch=minibatch)

	def refresh(self):
		self.indices_list = self.indices_rule(self.indices, shuffle=self.shuffle, minibatch=self.minibatch)

	def loader(self):
		for indices in self.indices_list:
			label, data = self.data_rule(self.data, self.labels, indices)
			yield label, data




class general_train():
	def __init__(self, loader, model, loss=None, opti=None, logging=None, epoch=100, verbose=True, cuda=True, save=True, gpu=0, params='', savePath="./"):
		self.loader = loader
		self.logging = logging
		self.epoch = epoch 
		self.verbose = verbose
		self.cuda = cuda
		self.gpu = gpu
		self.save = save
		self.savePath = savePath
		self.params = params
		self.opti_fun = opti
		self.model = model
		self.loss = loss



		# if(self.opti_fun!=None):
		# 	if(type(self.params)==str):
		# 		self.opti = self.opti_fun(self.model.parameters(), 0.001)
		# 	else:
		# 		self.opti = self.opti_fun(params, 0.001)

	def allocate(self):
		if(self.cuda==True):
			if(type(self.gpu)==list):
				self.model = DataParallel(self.model, device_ids=self.gpu).cuda()
				if(self.loss!=None):
					self.loss = self.loss.cuda()
			else:
				self.model = self.model.cuda(self.gpu)
				if(self.loss!=None):
					self.loss = self.loss.cuda(self.gpu)



	def update_opti(self, lr):
		if(type(self.params)==str):
			self.opti = self.opti_fun(self.model.parameters(), lr)
		else:
			self.opti = self.opti_fun(self.params, lr)

	def load_checkpoint(self, checkpoint, removeModule=False):
		if(removeModule==False):
			self.model.load_state_dict(checkpoint)
		else:
			new_state_dict = OrderedDict()
			for k, v in checkpoint.items():
			    name = k[7:] # remove `module.`
			    new_state_dict[name] = v
			self.model.load_state_dict(new_state_dict)


	def classify(self, loadPath, description=''):
		# self.logging.update(description, key="start")
		checkpoint = torch.load(loadPath)
		print(checkpoint)

		self.allocate()
		# self.model.load_state_dict(checkpoint)
		self.load_checkpoint(checkpoint, removeModule=True)

		self.loader.refresh()
		correct = 0
		_len = 0
		for index, (X, Y) in enumerate(self.loader.loader()):
			end = timer()
			# if(self.verbose==True):
			# 	string = "loading data: {}, {}, {}, time: {}".format(index, X.shape, Y.shape, timer()-end)
			# 	self.logging.update(string, key=str(epoch).zfill(3))
			# end = timer()
			X, Y = Variable(X), Variable(Y)
			_len = _len+Y.data.shape[0]

			if(self.cuda==True):
				if(type(self.gpu)==list):
					X = X.cuda()
					Y = Y.cuda()
				else:
					X = X.cuda(self.gpu)
					Y = Y.cuda(self.gpu)

			R = self.model(X)
			pred = R.data.max(1)[1]
			correct += pred.eq(Y.data).sum()
			_correct = correct*1./_len
			print("index: {}, num: {}, correct_so_far: {}, time: {}".format(index, Y.data.shape[0], _correct, timer()-end))

		correct = 100. * correct / _len
		print("correct: ", correct)

	def train(self, loadPath=None, description=''):
		offset = 0
		if(loadPath!=None):
			checkpoint = torch.load(loadPath)
			print(checkpoint)
			self.model.load_state_dict(checkpoint)
			offset = int(childPath(loadPath).split(".")[0])

		self.allocate()
		self.update_opti(0.001)

		turningPoint = False
		self.logging.update(description, key="start")
		for epoch in range(self.epoch):
			_epoch = epoch+offset
			self.loader.refresh()
			end = timer()
			correct = 0
			_len = 0
			for index, (X, Y) in enumerate(self.loader.loader()):
				if(self.verbose==True):
					string = "loading data: {}, {}, {}, time: {}".format(index, X.shape, Y.shape, timer()-end)
					self.logging.update(string, key=str(_epoch).zfill(3))

				end = timer()
				X, Y = Variable(X), Variable(Y)
				if(self.cuda==True):
					if(type(self.gpu)==list):
						X = X.cuda()
						Y = Y.cuda()
					else:
						X = X.cuda(self.gpu)
						Y = Y.cuda(self.gpu)

				R = self.model(X)
				L = self.loss(R, Y)
				self.opti.zero_grad()
				L.backward()
				self.opti.step()

				if(self.verbose==True):

					pred = R.data.max(1)[1]
					correct += pred.eq(Y.data).sum()
					_len = _len+Y.data.shape[0]

					_correct = correct*100./_len

					string = "result: loss {}, accuracy so far {}, time: {}".format(L.data[0], _correct, timer()-end)
					self.logging.update(string, key="INEPOCH")

					if(turningPoint==False):
						if(_correct>95):
							self.update_opti(0.00001)
							turningPoint = True
						elif(_correct>80):
							self.update_opti(0.0001)
							turningPoint = True

				end = timer()


			###
			# save
			###
			if(self.save==True):
				if(_epoch%5==0):
					state = self.model.state_dict()
					torch.save(state, self.savePath+str(_epoch).zfill(6)+".pth.tar")

			if(self.verbose==True):
				correct = 100. * correct / _len
				string = "epoch: {}, accuracy: {}".format(_epoch, correct)
				self.logging.update(string, key="OUTEPOCH")



from torch.nn.modules.module import _addindent
def torch_summarize(model, show_weights=True, show_parameters=True):
	"""Summarizes torch model by showing trainable parameters and weights."""
	tmpstr = model.__class__.__name__ + ' (\n'
	for key, module in model._modules.items():
		# if it contains layers let call it recursively to get params and weights
		if type(module) in [
			torch.nn.modules.container.Container,
			torch.nn.modules.container.Sequential
		]:
			modstr = torch_summarize(module)
		else:
			modstr = module.__repr__()
		modstr = _addindent(modstr, 2)

		params = sum([np.prod(p.size()) for p in module.parameters()])
		weights = tuple([tuple(p.size()) for p in module.parameters()])

		tmpstr += '==>(' + key + '): ' + modstr 
		if show_weights:
			tmpstr += ', weights={}'.format(weights)
		if show_parameters:
			tmpstr +=  ', parameters={}'.format(params)
		tmpstr += '\n\n'   

	tmpstr = tmpstr + ')'
	return tmpstr

from datetime import datetime as today
from .basic import formatPath, writeobj
from functools import reduce
class logging():
	def __init__(self, targetDir, onscreen=True, onfile=True, key='normal'):
		self.filename = formatPath(targetDir)+"/"+key+"_"+today.now().strftime('%m%d-%H%M')+".log"
		self.onscreen = onscreen
		self.onfile = onfile
	def update(self, data, key=None):
		if(type(data)==dict):

			string = reduce(lambda x,y:x+": {} "+y, [str(x) for x in _dict.keys()])+": {}"
			string = string.format(*[_dict[x] for x in _dict.keys()])
			string = string

		elif(type(data)==str):
			string = data
		elif(type(data)==list):
			string = data

		if(self.onfile==True):
			if(key!=None):
				writeobj(self.filename, "[{}] ".format(key))

			writeobj(self.filename, string)
			writeobj(self.filename, "\n")
		if(self.onscreen==True):
			if(key!=None):
				print("[{}] ".format(key), string)
			else:
				print(string)



if __name__ == '__main__':
	pass