import shutil
import os, copy, re
import pickle
import scipy, math
import scipy.sparse
import scipy.io
# import hdf5storage

import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	import hdf5storage


import numpy
def num2pivot(num):
	num = numpy.array(num)
	pivot = numpy.zeros(num.shape)
	for i in range(1, num.shape[0]):
		pivot[i] = pivot[i-1]+num[i-1]
	return pivot.tolist()




def clear_dir(_dir):
	shutil.rmtree(_dir)
	os.mkdir(_dir)

def mkdir(_dir):
	if not os.path.exists(_dir):
		os.makedirs(_dir)

def rm_file(_file):
	if os.path.exists(_file):
		os.remove(_file)

from functools import reduce
def writeobj(file, data):
	if(type(data)==str):
		line = data 
	elif(type(data)==list):
		line = reduce(lambda x,y: str(x)+" , "+str(y), data)
	with open(file, 'a') as f:
		f.write(line)

def formatPath(input_path):
	#no matter the input_path is ./a or ./a/, always return ./a
	parentPath=input_path
	parentPath=os.path.split(parentPath)
	if(parentPath[1]==''):
		parentPath=parentPath[0]
	else:
		parentPath=parentPath[0]+"/"+parentPath[1]
	return parentPath

def parentPath(input_path):
	input_path = formatPath(input_path)
	parentPath=os.path.split(input_path)[0]
	return parentPath

def childPath(input_path):
	input_path = formatPath(input_path)
	parentPath=os.path.split(input_path)[1]
	return parentPath


def save_obj(name, obj):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'rb') as f:
		return pickle.load(f)

def save_sparse(obj, name):
	if(re.search('.npz', name)):
		scipy.sparse.save_npz(name, obj)
	else:
		scipy.sparse.save_npz(name+".npz", obj)

def load_sparse(name):
	if(re.search('.npz', name)):
		X = scipy.sparse.load_npz(name)
	else:
		X = scipy.sparse.load_npz(name+".npz")
	return X

def load_mat(name, key=None, header='old'):
	if(header=='old'):
		M = scipy.io.loadmat(name)
	elif(header=='hdf5'):
		M = hdf5storage.loadmat(name)
	if(key!=None):
		M = M[key]
	return M



def dir_travers(current_path, key):
	'''
	directory travers with bfs
	'''
	dirs = sorted([current_path+"/"+x for x in os.listdir(current_path)])
	_next = []
	_next.extend(dirs)

	i = 0
	while len(_next)>0:
		current_next = sorted(copy.deepcopy(_next))
		for _dir in current_next:
			if os.path.isdir(_dir):
				_next.remove(_dir)
				dirs = sorted([_dir+"/"+x for x in os.listdir(_dir)])
				_next.extend(dirs)
			else:
				_next.remove(_dir)
				if re.search(key, _dir):
					yield _dir

def shuffle_list(_list):
	indices = [x for x in range(len(_list))]
	shuffle(indices)
	new_list = []
	for index in indices:
		new_list.append(_list[index])
	return new_list

from random import shuffle
def split_files(files, num, by='num', shuffle=False):
	if(shuffle==True):
		shuffle_list(files)

	if(by=='num'):
		num_to_split = num
		minibatch = max(1, math.ceil(len(files)/num))
	elif(by=='minibatch'):
		num_to_split = max(1, math.ceil(len(files)/num))
		minibatch = num

	file_list = []

	for index_group in range(num_to_split):		
		index_0 = index_group*minibatch
		index_1 = (index_group+1)*minibatch
		if(index_1>len(files)):
			index_1 = len(files)

		file_list.append(files[index_0:index_1])
	
	return file_list

import pandas
def file2pandas(path, header=None, sep=',', low_memory=True):
	df = pandas.read_csv(path, header=header, sep=sep, low_memory=low_memory)
	if(header==None):
		columns = [x for x in range(len(df.keys()))]
		df.columns = columns
	return df

import numpy
def line2pos_value(M, column_size=None, reshape=False):
	column = numpy.array(M)

	if(column_size==None):
		column_size = column.shape[0]

	if(reshape==True):
		balance_x = math.floor(math.sqrt(column_size))
		balance_y = math.ceil(column_size/balance_x)
		shape = [balance_x, balance_y]
		row_source = column
		row = numpy.floor(row_source/balance_y).astype('int')
		column = (row_source%balance_y)
		value = numpy.ones(row_source.shape[0])
	else:
		shape = [1, column_size]
		row = numpy.ones(column.shape[0])
		value = numpy.ones(column.shape[0])

	return [value, [row, column]], shape



if __name__ == '__main__':
	# import numpy
	# _dict = {}
	# for i in range(10):
	# 	_dict[i] = numpy.random.randn(10)

	# # print(_dict)
	# save_obj(_dict, "./data/test_dict_numpyArray")
	# data = load_obj("./data/test_dict_numpyArray")
	# print(data)
	# input_path = "./a/a//c"
	# print(childPath(input_path))
	path = "../DATA/O.mat"
	M = load_mat(path)
	print(type(M['O']))

