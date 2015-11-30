import caffe
import sys
import argparse, pprint
import time
import numpy as np
import scipy.misc as scmk
from easydict import EasyDict as edict
from multiprocessing import Pool
from collections import deque
from pycaffe_config import cfg
from os import path as osp
sys.path.append(osp.join(cfg.BILLIARDS_CODE_PATH, 'code/physicsEngine'))
import dataio as dio
import pdb

def wrapper_fetch_data(rndSeed, **kwargs):
	ds  = dio.DataSaver(randSeed=rndSeed, **kwargs)	
	ims = ds.fetch()
	return ims

class DataFetch(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Try Layer')
		parser.add_argument('--mnBallSz', default=15, type=int)
		parser.add_argument('--mxBallSz', default=35, type=int)
		parser.add_argument('--mnSeqLen', default=10, type=int)
		parser.add_argument('--mxSeqLen', default=100, type=int)
		parser.add_argument('--mnForce',  default=1e+3, type=float)
		parser.add_argument('--mxForce',  default=1e+3, type=float)
		parser.add_argument('--isRect' ,  default=True, type=bool)
		parser.add_argument('--wTheta' ,  default=30,   type=float)
		parser.add_argument('--mxWLen' ,  default=600, type=int)
		parser.add_argument('--mnWLen' ,  default=200, type=int)
		parser.add_argument('--arenaSz',  default=667, type=int)
		parser.add_argument('--batchSz',  default=16, type=int)
		parser.add_argument('--imSz'   ,  default=227, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using World Config:')
		pprint.pprint(args)
		return edict(vars(args))	
	
	def setup(self, bottom, top):
		#Get the parameters
		self.params_ = DataFetch.parse_args(self.param_str) 
		#Shape the output blobs
		top[0].reshape(self.params_.batchSz, self.params_.imSz, self.params_.imSz, 3)
		top[1].reshape(self.params_.batchSz,)
		#Start the pool of workers
		self.pool_   = Pool(8)
		self.jobs_   = deque()	
		#Start loading the data
		self.fetch_data()

	def fetch_data(self):
		rnd = int(time.time()) 
		for i in range(self.params_.batchSz):
			#ds  = dio.DataSaver(randSeed=rnd+i, **self.params_)	
			self.jobs_.append(self.pool_.apply_async(wrapper_fetch_data, 
					[rnd + i], self.params_))
		print ('Data Fetch started')

	def forward(self, bottom, top):
		#Get the data from the already launched processes
		for i in range(self.params_.batchSz):
			data = self.jobs_.popleft()
			data = data.get()  
			top[0].data[i,:,:,:] = scm.imresize(data[0], (self.params_.imSz, self.params_.imSz))
		#Launch new processes
		self.fetch_data()	

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
