import caffe
import sys
import argparse, pprint
import time
import numpy as np
import scipy.misc as scm
from easydict import EasyDict as edict
from multiprocessing import Pool
from collections import deque
sys.path.append('/work4/pulkitag-code/code/physicsEngine')
import dataio as dio
import pdb
import Queue

def wrapper_fetch_data(args):
	rndSeed, params = args
	params['randSeed'] = rndSeed
	ds  = dio.DataSaver(**params)	
	ims = ds.fetch(params['imSz'])
	return ims

class DataFetchLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Try Layer')
		parser.add_argument('--mnBallSz', default=15, type=int)
		parser.add_argument('--mxBallSz', default=35, type=int)
		parser.add_argument('--mnSeqLen', default=10, type=int)
		parser.add_argument('--mxSeqLen', default=100, type=int)
		parser.add_argument('--mnForce',  default=1e+3, type=float)
		parser.add_argument('--mxForce',  default=1e+5, type=float)
		parser.add_argument('--isRect' ,  default=True, type=bool)
		parser.add_argument('--wTheta' ,  default=30,   type=float)
		parser.add_argument('--mxWLen' ,  default=600, type=int)
		parser.add_argument('--mnWLen' ,  default=200, type=int)
		parser.add_argument('--arenaSz',  default=667, type=int)
		parser.add_argument('--batchSz',  default=16, type=int)
		parser.add_argument('--imSz'   ,  default=227, type=int)
		parser.add_argument('--lookAhead',  default=10, type=int)
		parser.add_argument('--history',  default=4, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using World Config:')
		pprint.pprint(args)
		return edict(vars(args))	
	
	def setup(self, bottom, top):
		#Get the parameters
		self.params_ = DataFetchLayer.parse_args(self.param_str) 
		#Shape the output blobs
		top[0].reshape(self.params_.batchSz, self.params_.imSz,
											 self.params_.imSz, 3 * self.params_.history)
		top[1].reshape(self.params_.batchSz, self.params_.lookAhead, 2, 1)	
		#Start the pool of workers
		self.pool_   = Pool(16)
		self.jobs_   = deque()	
		#Make a Queue for storing the game plays
		self.play_cache_ = Queue.Queue(maxsize=self.params_.batchSz)
		#The datastreams
		self.plays_ = []
		self.plays_toe_ = [] #Stores the time of end of each play
		self.plays_tfs_ = [] #Stores the time of start of each play
		for j in range(self.params_.batchSz):
			self.plays_.append([])
			self.plays_toe_.append(0)
			self.plays_tfs_.append(0)
		#Prepare the data and label vectors
		self.imdata_ = np.zeros((self.params_.batchSz,self.params_.imSz, 
										self.params_.imSz, 3 * self.params_.history), np.float32) 
		self.labels_ = np.zeros((self.params_.batchSz, self.params_.lookAhead, 
										2, 1), np.float32) 
		#Start loading the data
		self.prefetch()
		

	def prefetch(self):
		rnd     = int(time.time())
		jobArgs = [] 
		for i in range(self.params_.batchSz):
			#ds  = dio.DataSaver(randSeed=rnd+i, **self.params_)	
			jobArgs.append([rnd + i, self.params_])
		try:
			self.jobs_ = self.pool_.map_async(wrapper_fetch_data, jobArgs)
			print ('Data Fetch started')
		except KeyboardInterrupt:
			print 'Keyboard Interrupt received - terminating in launch jobs'
			self.pool_.terminate()	

	def get_cache_data(self):
		#If the plays_ Queue is empty get the data from prefetch
		#and populate the queue. 
		if self.play_cache_.empty():
			try:
				data = self.jobs_.get()
			except KeyboardInterrupt:
				print 'Keyboard Interrupt received - terminating'
				self.pool_.terminate()
			for d in data:
				self.play_cache_.put(d)	
			self.prefetch()
		#Return one of the plays from the Queue
		return self.play_cache_.get()

	def get_next_sample(self):
		for j in range(self.params_.batchSz):
			#Ensure samples can be taken for all batches
			if self.plays_toe_[j] - self.params_.lookAhead < 0:
				print ('I AM HERE', j)
				self.plays_[j]     = self.get_cache_data()	
				self.plays_toe_[j] = len(self.plays_[j][0][0])
				self.plays_tfs_[j] = 0
			imBalls, force, _ = self.plays_[j]
			#For now just chose the first ball
			imBall = imBalls[0]
			for h in range(self.params_.history):
				stCh = h * 3
				enCh = stCh + 3
				#print (stCh, enCh, h) 
				h = max(0, len(self.plays_[j]) - self.plays_toe_[j] + h)
				#print (stCh, enCh, h, imBall[h].shape)
				self.imdata_[j,:,:,stCh:enCh] = imBall[h][:,:,0:3]
			stLbl = self.plays_tfs_[j]
			enLbl = stLbl + self.params_.lookAhead
			for l in range(stLbl, enLbl):
				#Choose the first ball
				self.labels_[j,l-stLbl, 0:2] = force[0:2,l].reshape((2,1))
			#Update the counts
			self.plays_tfs_[j] += 1
			self.plays_toe_[j] -= 1
			
							
	def forward(self, bottom, top):
		#Get the data from the already launched processes
		t1 = time.time()
		self.get_next_sample()
		t2= time.time()
		tFetch = t2 - t1
		print ('Waiting for fetch: %f'  %tFetch)
		top[0].data[...] = self.imdata_[...]
		top[1].data[...] = self.labels_[...]

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
