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

##
#Simple L1 loss layer
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
		assert bottom[0].data.squeeze().ndim == bottom[1].data.squeeze().ndim, 'Shape Mismatch'
  	#Get the batchSz
		self.batchSz_ = gtShape[0]
		#Form the top
		assert len(top)==1, 'There should be only one output blob'
		top[0].reshape(1,1,1,1)
		
	def forward(self, bottom, top):
		top[0].data[...] = self.param_.loss_weight * np.sum(np.abs(bottom[0].data[...].squeeze()\
													 - bottom[1].data[...].squeeze()))/float(self.batchSz_)	
		glog.info('Loss is %f' % top[0].data[0])
	
	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[...] = self.param_.loss_weight * np.sign(bottom[0].data[...].squeeze()\
														 - bottom[1].data[...].squeeze())/float(self.batchSz_)
		
	def reshape(self, bottom, top):
		top[0].reshape(1,1,1,1)
		pass

##
#L1 loss layer, which the ability to ignore the loss computation for some examples
#This can be done - by making gt labels of dimension N + 1, whereas the vectors
#between which the error is being computed are N-D. If the (N+1)th dimension is set
#to 1, it means that the example should be included otherwise not. 
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
		bottom[0].diff[...] = 0
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

##
#L1Log loss layer, which the ability to ignore the loss computation for some examples
#This can be done - by making gt labels of dimension N + 1, whereas the vectors
#between which the error is being computed are N-D. If the (N+1)th dimension is set
#to 1, it means that the example should be included otherwise not. 
#L1Log loss layer is more robust than L1Loss layer
class L1LogLossWithIgnoreLayer(caffe.Layer):
	'''
		if err = abs(err) if abs(err) <= 1
					 = 1 + log(abs(err)) if abs(err) > 1
	'''
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L1LogLoss With Ignore Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LogLossWithIgnoreLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		assert len(top) == 1, 'There should be 1 top blobs'
		assert (bottom[0].num == bottom[1].num)
		assert bottom[0].channels + 1 == bottom[1].channels,\
           '%d, %d' % (bottom[0].channels + 1, bottom[1].channels)
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
				err   = np.abs(bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze())
				err   = np.array(err)
				idx   = err > 1
				err[idx] = np.log(err[idx]) + 1
				loss     += np.sum(err)
				count    += 1
		if count == 0:
			top[0].data[...] = 0.0
		else:
			top[0].data[...] = self.param_.loss_weight * loss /float(count)	
	
	def backward(self, top, propagate_down, bottom):
		count = 0
		bottom[0].diff[...] = 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				count += 1
				diff   = bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()
				diff   = np.array(diff)
				err    = np.array(np.abs(diff))
				idx    = err > 1
				diff[~idx] = np.sign(diff[~idx])
				diff[idx]  = (1/err[idx]) * np.sign(diff[idx])
				bottom[0].diff[b] = diff[...]
		if count == 0:
			bottom[0].diff[...] = 0
		else:
			bottom[0].diff[...] = self.param_.loss_weight * bottom[0].diff[...]/float(count)	

	def reshape(self, bottom, top):
		top[0].reshape(1)
		pass

##
#L2 loss layer, which the ability to ignore the loss computation for some examples
#This can be done - by making gt labels of dimension N + 1, whereas the vectors
#between which the error is being computed are N-D. If the (N+1)th dimension is set
#to 1, it means that the example should be included otherwise not. 
class L2LossWithIgnoreLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L2Loss With Ignore Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LogLossWithIgnoreLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		assert len(top) == 1, 'There should be 1 top blobs'
		assert (bottom[0].num == bottom[1].num)
		assert bottom[0].channels + 1 == bottom[1].channels,\
           '%d, %d' % (bottom[0].channels + 1, bottom[1].channels)
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
				err   = bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()
				err   = np.array(err)
				loss  += 0.5 * np.sum(err * err)
				count    += 1
		if count == 0:
			top[0].data[...] = 0.0
		else:
			top[0].data[...] = self.param_.loss_weight * loss /float(count)	
	
	def backward(self, top, propagate_down, bottom):
		count = 0
		bottom[0].diff[...] = 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				count += 1
				diff   = bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()
				bottom[0].diff[b] = diff[...]
		if count == 0:
			bottom[0].diff[...] = 0
		else:
			bottom[0].diff[...] = self.param_.loss_weight * bottom[0].diff[...]/float(count)	

	def reshape(self, bottom, top):
		top[0].reshape(1)
		pass

##
#L2 loss layer, which the ability to ignore the loss computation for some examples
#This can be done - by making gt labels of dimension N + 1, whereas the vectors
#between which the error is being computed are N-D. If the (N+1)th dimension is set
#to 1, it means that the example should be included otherwise not. 
class L2LossQuaternionWithIgnoreLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L2LossQuaternion With Ignore Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LogLossWithIgnoreLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		assert len(top) == 1, 'There should be 1 top blobs'
		assert (bottom[0].num == bottom[1].num)
		assert bottom[0].channels + 1 == bottom[1].channels,\
           '%d, %d' % (bottom[0].channels + 1, bottom[1].channels)
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
				#nrmlz the gt and pred
				pd  = bottom[0].data[b].squeeze()
				pdZ = np.sqrt(np.sum(pd * pd))
				if pdZ > 0: 
					pd = pd / pdZ
				gt  = bottom[1].data[b,0:-1].squeeze()
				gtZ = np.sqrt(np.sum(gt * gt))
				if gtZ > 0: 
					gt = gt / gtZ
				#q and -q are the same in the quaterion world
				err1 = np.sum((pd - gt) * (pd - gt))
				err2 = np.sum((-pd - gt) * (-pd - gt))
				#err2 = 1000
				#print (err1, err2)
				loss  += 0.5 * min(err1, err2)
				count += 1
		if count == 0:
			top[0].data[...] = 0.0
		else:
			top[0].data[...] = self.param_.loss_weight * loss /float(count)	
	
	def backward(self, top, propagate_down, bottom):
		count = 0
		bottom[0].diff[...] = 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				count += 1
				#nrmlz the gt and pred
				pdU  = bottom[0].data[b].squeeze()
				pdZ  = np.sqrt(np.sum(pdU * pdU))
				if pdZ > 0: 
					pd = pdU / pdZ
				else:
					pd = pdU
				gtU = bottom[1].data[b,0:-1].squeeze()
				gtZ = np.sqrt(np.sum(gtU * gtU))
				if gtZ > 0: 
					gt = gtU / gtZ
				else:
					gt = gtU
				#q and -q are the same in the quaterion world
				err1 = np.sum((pd - gt) * (pd - gt))
				err2 = np.sum((-pd - gt) * (-pd - gt))
				#print err2
				nDim  = bottom[0].data[b].shape[0]
				diff = np.zeros((nDim,), np.float32)
				if err1 < err2:
					#print ('e1')
					if pdZ > 0:
						for i in range(nDim):
							grad    = -pdU[i] * pdU / np.power(pdZ,3)
							grad[i] = grad[i] + 1 / pdZ
							diff[i] = np.dot((pd  - gt).reshape(1, nDim), grad) 
					else:
						diff = (pd - gt)
				else:
					#print ('e2')
					if pdZ > 0:
						for i in range(nDim):
							grad    = -pdU[i] * pdU / np.power(pdZ,3)
							grad[i] = grad[i] + 1 / pdZ
							diff[i] = np.dot((-1) * (-pd  - gt).reshape(1, nDim), grad) 
						#diff = (-pd - gt) * (-1) * (-(pdU * pdU) / np.power(pdZ, 3) + np.ones(pdU.shape) / pdZ)
					else:
						diff = (-pd - gt) * (-1)
				bottom[0].diff[b] = diff.reshape(bottom[0].diff[b].shape)
		if count == 0:
			bottom[0].diff[...] = 0
		else:
			bottom[0].diff[...] = self.param_.loss_weight * bottom[0].diff[...]/float(count)	
		#print bottom[0].diff

	def reshape(self, bottom, top):
		top[0].reshape(1)
		pass



##
#L2 loss layer, which the ability to ignore the loss computation for some examples
#This can be done - by making gt labels of dimension N + 1, whereas the vectors
#between which the error is being computed are N-D. If the (N+1)th dimension is set
#to 1, it means that the example should be included otherwise not. 
class L2LossWithIgnoreLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L2Loss With Ignore Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LogLossWithIgnoreLayer.parse_args(self.param_str)
		assert len(bottom) == 2, 'There should be two bottom blobs'
		assert len(top) == 1, 'There should be 1 top blobs'
		assert (bottom[0].num == bottom[1].num)
		assert bottom[0].channels + 1 == bottom[1].channels,\
           '%d, %d' % (bottom[0].channels + 1, bottom[1].channels)
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
				err   = bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()
				err   = np.array(err)
				loss  += 0.5 * np.sum(err * err)
				count    += 1
		if count == 0:
			top[0].data[...] = 0.0
		else:
			top[0].data[...] = self.param_.loss_weight * loss /float(count)	
	
	def backward(self, top, propagate_down, bottom):
		count = 0
		bottom[0].diff[...] = 0
		for b in range(self.batchSz_):
			if bottom[1].data[b,-1,0,0] == 1.0:
				count += 1
				diff   = bottom[0].data[b].squeeze() - bottom[1].data[b,0:-1].squeeze()
				bottom[0].diff[b] = diff[...].reshape(bottom[0].diff[b].shape)
		if count == 0:
			bottom[0].diff[...] = 0
		else:
			bottom[0].diff[...] = self.param_.loss_weight * bottom[0].diff[...]/float(count)	

	def reshape(self, bottom, top):
		top[0].reshape(1)
		pass



##
#L1 loss layer which allows each dimension of |a - b| to weighted by a seperated weight
#This is useful for instance when there is a lookahead and samples in the future should
#be weighted less than current samples. 
class L1LossWeightedLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python L1 Weighted Loss Layer')
		parser.add_argument('--loss_weight', default=1.0, type=float)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
		
	def setup(self, bottom, top):
		self.param_ = L1LossWeightedLayer.parse_args(self.param_str)
		assert len(bottom) == 3, 'There should be three bottom blobs'
		predShape = bottom[0].data.shape
		gtShape   = bottom[1].data.shape
		wtShape   = bottom[2].data.shape
		for i in range(len(predShape)):
			assert predShape[i] == gtShape[i], 'Mismatch: %d, %d' % (predShape[i], gtShape[i])
			if i > 0:
				assert gtShape[i] == wtShape[i],'Mismatch: %d, %d' % (wtShape[i], gtShape[i])
				
		#Get the batchSz
		self.batchSz_ = gtShape[0]
		#Form the top
		assert len(top)==1, 'There should be only one output blob'
		top[0].reshape(1,1,1,1)
		
	def forward(self, bottom, top):
		wtErr = np.abs(bottom[0].data[...] - bottom[1].data[...]) * bottom[2].data[...]
		top[0].data[...] = self.param_.loss_weight * np.sum(wtErr)/float(self.batchSz_)	
		glog.info('Loss is %f' % top[0].data[0])
	
	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[...] = self.param_.loss_weight * bottom[2].data[...] *\
			np.sign(bottom[0].data[...] - bottom[1].data[...])/float(self.batchSz_)
		
	def reshape(self, bottom, top):
		top[0].reshape(1,1,1,1)
		pass

