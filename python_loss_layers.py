import caffe
import numpy as np
import argparse, pprint
import scipy.misc as scm
from os import path as osp
from easydict import EasyDict as edict
import time
import glog
import pdb
import pickle

class L1LossLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L1 Loss Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LossLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		predShape = bottom[0].data.shape
		gtShape   = bottom[1].data.shape
		for i in range(len(predShape)):
			assert predShape[i] == gtShape[i], 'Mismatch: %d, %d' % (predShape[i], gtShape[i])
		#Get the batchSz
		self.batchSz_ = gtShape[0]
		#Form the top
		assert len(top)==1, 'There should be only one output blob'
		top[0].reshape(1,1,1,1)
		
	def forward(self, bottom, top):
		top[0].data[...] = self.param_.loss_weight * np.sum(np.abs(bottom[0].data[...]\
													 - bottom[1].data[...]))/float(self.batchSz_)	
		glog.info('Loss is %f' % top[0].data[0])
	
	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[...] = self.param_.loss_weight * np.sign(bottom[0].data[...]\
														 - bottom[1].data[...])/float(self.batchSz_)
		
	def reshape(self, bottom, top):
		top[0].reshape(1,1,1,1)
		pass

class L1LossWithIgnoreLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L1 Loss With Ignore Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LossWithIgnoreLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		assert len(top) == 1, 'There should be 1 top blobs'
		assert (bottom[0].num == bottom[1].num)
		assert (bottom[0].channels + 1 == bottom[1].channels)
		assert (bottom[0].width == bottom[1].width)
		assert (bottom[0].height== bottom[1].height)
		#Get the batchSz
		self.batchSz_ = bottom[0].num
		#Form the top
		assert len(top)==1, 'There should be only one output blob'
		top[0].reshape(1)
		
	def forward(self, bottom, top):
		loss, count = 0, 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				loss += np.sum(np.abs(bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()))
				count += 1
		#pickle.dump({'pd': bottom[0].data, 'gt': bottom[1].data}, open('data_dump.pkl','w'))
		#glog.info('%f, %d, %d' % (loss, count, self.batchSz_))
		#glog.info('%f, %f, %d' % (bottom[0].data[b,0], bottom[1].data[b,0], b))
		if count == 0:
			top[0].data[...] = 0.0
		else:
			top[0].data[...] = self.param_.loss_weight * loss /float(count)	
		#glog.info('Loss is %f, count: %d' % (top[0].data[0], count))
	
	def backward(self, top, propagate_down, bottom):
		count = 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				count += 1
				bottom[0].diff[b] = np.sign(bottom[0].data[b] - bottom[1].data[b,0:-1].squeeze())
		if count == 0:
			bottom[0].diff[...] = 0
		else:
			bottom[0].diff[...] = self.param_.loss_weight * bottom[0].diff[...]/float(count)	

		
	def reshape(self, bottom, top):
		top[0].reshape(1)
		pass


