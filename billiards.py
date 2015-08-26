import caffe
import sys
import argparse, pprint
import time
import numpy as np
from easydict import EasyDict as edict
from multiprocessing import Pool
from collections import deque
sys.path.append('/work4/pulkitag-code/code/physicsEngine')
import dataio as dio

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
		self.params_ = TryLayer.parse_args(self.param_str) 
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
			ds  = dio.DataSaver(randSeed=rnd+i, **self.params_)	


	def forward(self, bottom, top):
		#Get the data from the already launched processes

		#Launch new processes
		self.fetch_data()	
		#Fill in the top
		top[0].data[...] = top[0].data + self.params_.aa * np.ones(top[0].shape)	
	

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
