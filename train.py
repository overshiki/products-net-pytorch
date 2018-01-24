import torch
from torch.autograd import Variable
from utils import data_loader, field_size
from timeit import default_timer
from PYTORCH.utils import logging

def train(model, logger, data_index=1458, _type='IPNN', embedding_method=None, refresh=False, minibatch=200, device_ids=[0]):
	logger.update("data_index: {}".format(data_index), key='define')
	_loader, column_path = data_loader(data_index=data_index, refresh=refresh, minibatch=minibatch, shuffle=True, gpu=True)
	fields = field_size(column_path)
	input_dims = sum(fields)+1

	if(_type=='IPNN'):
		logger.update("method: IPNN", key='define')
		logger.update("embedding_method to be: {}".format(embedding_method.__name__), key='define')
		model = model(fields, embedding_method=embedding_method, embedding_dim=10).cuda()
	elif(_type=='LR'):
		logger.update("method: LR", key='define')
		model = model(input_dims).cuda()

	# model = torch.nn.DataParallel(model, device_ids=device_ids)

	loss = torch.nn.NLLLoss().cuda()
	lr = 1e-3
	solver = torch.optim.Adam(model.parameters(), lr=lr)

	for _iter in range(1000):
		end = default_timer()
		_loader.create_iter()

		LOSS, sub_accumulate_correct_pre, sub_accumulate_correct_after = 0, 0, 0

		_sum = {}
		_sum_pre = {}

		index = 1
		for tensor, _labels in _loader._iter:
			if(embedding_method.__name__=='linear_field'):
				tensor = tensor.float()
			else:
				tensor = tensor.long()

			for inner_iter in range(1000):
				inputs = Variable(tensor)
				labels = Variable(_labels)
				outputs = model(inputs)

				_loss = loss(outputs, labels)

				LOSS = LOSS+_loss.data[0]

				_loss.backward()
				solver.step()

				if(inner_iter==0):
					pred_pre = outputs.data.max(1)[1]
					pred = outputs.data.max(1)[1]
					correct_pre = pred.eq(labels.data).sum()
					sub_accumulate_correct_pre += pred.eq(labels.data).sum()

				if(inner_iter==999):
					pred = outputs.data.max(1)[1]
					correct_after = pred.eq(labels.data).sum()
					sub_accumulate_correct_after += pred.eq(labels.data).sum()

			for x in range(pred.shape[0]):
				if pred[x] not in list(_sum.keys()):
					_sum[pred[x]] = 1 
				else:
					_sum[pred[x]] = _sum[pred[x]]+1

			for x in range(pred_pre.shape[0]):
				if pred_pre[x] not in list(_sum_pre.keys()):
					_sum_pre[pred_pre[x]] = 1 
				else:
					_sum_pre[pred_pre[x]] = _sum_pre[pred_pre[x]]+1

			time = default_timer()-end
			end = default_timer()

			string_pre = "index: {}, time: {}, sub_dict: {}, correct: {}, sub_accumulate_correct: {}".format(index, time, _sum_pre, correct_pre*1./(labels.data.shape[0]), sub_accumulate_correct_pre*1./(index*labels.data.shape[0])  )

			string_after = "index: {}, time: {}, sub_dict: {}, correct: {}, sub_accumulate_correct: {}".format(index, time, _sum, correct_after*1./(labels.data.shape[0]), sub_accumulate_correct_after*1./(index*labels.data.shape[0])  )

			logger.update(string_pre, key='pre')
			logger.update(string_after, key='after')

			if(index==350):
				break

			index = index+1

		break

		correct = 100.*correct/_loader._len

		LOSS = LOSS/_loader._len 

		print("loss: {:6f}, accuracy: {:6f}, len: {}, step_time: {:4f}".format(LOSS, correct, _loader._len, default_timer()-end))
		print("... dict: {}".format(_sum))




def debug(model, minibatch=200, device_ids=[0]):
	_loader, column_path = data_loader(debug=True, refresh=True, minibatch=minibatch, shuffle=False, gpu=True, truncate=True)
	fields = field_size(column_path)

	embedding_dim = 10
	model = model(fields, embedding_dim).cuda()

	# model = torch.nn.DataParallel(model, device_ids=device_ids)

	loss = torch.nn.NLLLoss().cuda()
	lr = 1e-3
	solver = torch.optim.Adam(model.parameters(), lr=lr)

	for _iter in range(10):
		end = default_timer()
		_loader.create_iter()

		LOSS, correct = 0, 0

		_sum = {}

		for tensor, labels in _loader._iter:
			tensor = tensor.long()

			inputs = Variable(tensor)
			labels = Variable(labels)
			outputs = model(inputs)

			_loss = loss(outputs, labels)

			LOSS = LOSS+_loss.data[0]

			_loss.backward()
			solver.step()

			pred = outputs.data.max(1)[1]
			correct += pred.eq(labels.data).sum()

			for x in range(pred.shape[0]):
				if pred[x] not in list(_sum.keys()):
					_sum[pred[x]] = 1 
				else:
					_sum[pred[x]] = _sum[pred[x]]+1

		break

		correct = 100.*correct/_loader._len

		LOSS = LOSS/_loader._len 

		print("loss: {:6f}, accuracy: {:6f}, len: {}, step_time: {:4f}".format(LOSS, correct, _loader._len, default_timer()-end))
		print("... dict: {}".format(_sum))


	# _loader.create_iter()
	# for tensor, labels in _loader._iter:
	# 	tensor = tensor.long()
	# 	print(tensor.shape, labels.shape, tensor.type(), labels.type())
	# 	outputs = model(Variable(tensor))
	# 	print(outputs.data.type(), outputs.data.shape)
	# 	break



if __name__ == '__main__':
	from IPNN import IPNN, linear_field, embedding_field
	from LR import LR

	# debug(IPNN, minibatch=20, device_ids=[0])
	# train(IPNN, embedding_field, minibatch=200, device_ids=[0])
	for index in [2821, 2997, 3358, 3386, 3427, 3476]:
		targetDir = "./log/"
		logger = logging(targetDir, onscreen=True)
		train(IPNN, logger, data_index=index, _type='IPNN', refresh=True, embedding_method=linear_field, minibatch=200, device_ids=[0])

		targetDir = "./log/"
		logger = logging(targetDir, onscreen=True)	
		train(LR, logger, data_index=index, _type='LR', refresh=False, embedding_method=linear_field, minibatch=200, device_ids=[0])
