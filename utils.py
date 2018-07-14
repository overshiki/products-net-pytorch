from PYTORCH.utils import loader
import numpy as np
from torch_utils.basic import file2pandas, line2pos_value


def column_info(file):
	pd = file2pandas(file, sep='\t')
	_min = pd[1].as_matrix().min()
	_max = pd[1].as_matrix().max()
	_len = _max-_min+1
	return _len


def field_size(file):
	FIELD_SIZES = [0] * 26
	with open(file) as fin:
		for i,line in enumerate(fin):
			line = line.strip().split(':')
			if len(line) > 1:
				f = int(line[0]) - 1
				FIELD_SIZES[f] += 1
	return FIELD_SIZES

def _filter(line, column_len):
	s = line.replace(':', ' ').split()
	y=int(s[0])
	feats = sorted([int(s[j]) for j in range(2, len(s), 2)])

	return line2pos_value(feats, column_size=column_len, reshape=True), y

def line_iter(text_path):
	with open(text_path, 'r') as file:
		for index, _line in enumerate(file):
			yield index, _line

def loader_function(text_path=None, column_path=None, truncate=False):
	files = []
	header = []
	column_len = column_info(column_path)
	print("loading from text file {} ...".format(text_path))
	for index, line in line_iter(text_path):
		data, y = _filter(line, column_len)
		files.append(data)
		header.append(y)
		print("index: {}, label: {}".format(index, y))
		if(truncate==True)and(index==100):
			break

	print("finished loading text file")
	return files, header

from PYTORCH.utils import generate_fun, loader 
from IO.basic import mkdir
def data_loader(data_index=1458, refresh=False, minibatch=2, shuffle=True, gpu=True, truncate=False, debug=False):
	column_path = "../data/source/"+str(data_index)+"/featindex.txt"
	source_path = "../data/source/"+str(data_index)+"/train.yzx.txt"
	path = "../data/pos_value/"+str(data_index)

	if(debug==True):
		path = "../data/pos_value/"+str(data_index)+"/debug"
		mkdir(path)

	_len = column_info(column_path)
	def afterprocess(data):
		after_size = data.shape[1]*data.shape[2]
		data = data.view(-1, after_size)
		data = data[:,0:_len]
		return data


	genrator = generate_fun(loader_function, text_path=source_path, column_path=column_path, truncate=truncate)

	_loader = loader(
		path,
		generate_fun=genrator,
		load=True,
		refresh=refresh,
		minibatch=minibatch,
		shuffle=shuffle,
		_type='pos_value', 
		gpu=gpu,
		afterprocess = afterprocess
		)

	return _loader, column_path


if __name__ == '__main__':
	# column_path = "../data/source/1458/featindex.txt"
	# _len = column_info(column_path)
	# source_path = "../data/source/1458/train.yzx.txt"
	# path = "../data/pos_value/1458/"
	# files, header = loader_function(text_path=source_path, column_path=column_path)

	_loader = data_loader(refresh=False, minibatch=1000, shuffle=True)
	_loader.create_iter()
	for tensor, labels in _loader._iter:
		print(tensor.shape, labels.shape)

	# from MATRIX.sparse import sparse2matrix, pos_value2tensor
	# from GPU.prime import factorParallel
	# import math
	# column_path = "../data/source/1458/featindex.txt"
	# source_path = "../data/source/1458/train.yzx.txt"
	# column_len = column_info(column_path)
	# print("loading from text file {} ...".format(source_path))
	# for index, line in line_iter(source_path):
	# 	data, y = _filter(line, column_len)
	# 	if(index>=1):
	# 		break

	# # print(data)
	# new_data = [0,1]
	# size = data[1][1]

	# balance_x = math.floor(math.sqrt(size))
	# balance_y = math.ceil(size/balance_x)
	# print(size, balance_x*balance_y)


	# new_data[0] = data[0]
	# new_data[1] = [balance_x, balance_y]
	# # print(new_data)


	# sparse = pos_value2tensor(new_data).cuda()
	# print(sparse)

	# dense = sparse.to_dense()
	# print(dense)

	# dense = dense.view(-1, balance_x*balance_y)
	# print(dense)
