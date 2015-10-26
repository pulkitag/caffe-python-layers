import caffe
import numpy as np
import argparse, pprint
from multiprocessing import Pool
import scipy.misc as scm
from os import path as osp
import my_pycaffe_io as mpio
import my_pycaffe as mp
from easydict import EasyDict as edict

class PythonWindowDataLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python Window Data Layer')
		parser.add_argument('--source', default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--mean_file', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
	
	def setup(self, bottom, top):
		self.param_ = PythonWindowDataLayer.parse_args(self.param_str) 
		self.wfid_   = mpio.GenericWindowReader(self.param_.source)
		self.numIm_  = self.wfid_.numIm_
		self.lblSz_  = self.wfid_.lblSz_
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
										self.param_.crop_size, self.param_.crop_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_, 1, 1)

		self.mu_ = None
		if len(self.param_.mean_file) > 0:
			#Mean is assumbed to be in BGR format
			self.mu_ = mp.read_mean(self.param_.mean_file)
			self.mu_ = self.mu_.astype(np.float32)
			ch, h, w = self.mu_.shape
			assert (h >= self.param_.crop_size and w >= self.param_.crop_size)
			y1 = int(h/2 - (self.param_.crop_size/2))
			x1 = int(w/2 - (self.param_.crop_size/2))
			y2 = int(y1 + self.param_.crop_size)
			x2 = int(x1 + self.param_.crop_size)
			self.mu_ = self.mu_[:,y1:y2,x1:x2]

	def forward(self, bottom, top):
		for b in range(self.param_.batch_size):
			imNames, lbls = self.wfid_.read_next()
			#Read images
			for n in range(self.numIm_):
				#Load images
				imName, ch, h, w, x1, y1, x2, y2 = imNames[n].strip().split()
				imName = osp.join(self.param_.root_folder, imName)
				im     = scm.imread(imName)
				#Process the image
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				#Feed the image	
				cSt = n * self.ch_
				cEn = cSt + self.ch_
				im = scm.imresize(im[y1:y2, x1:x2, :],
									(self.param_.crop_size, self.param_.crop_size))
				im = im[:,:,[2,1,0]].transpose((2,0,1))
				if self.mu_ is not None:
					im = im - self.mu_
				top[0].data[b,cSt:cEn, :, :] = im.astype(np.float32)
			#Read the labels
			top[1].data[b,:,:,:] = lbls.reshape(self.lblSz_,1,1).astype(np.float32) 

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass

##
#Parallel version
class PythonWindowDataParallelLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python Window Data Parallel Layer')
		parser.add_argument('--source', default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', default=False, type=bool)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.wfid_.close()
		self.pool_.terminate()
	
	def setup(self, bottom, top):
		self.param_ = PythonWindowDataLayer.parse_args(self.param_str) 
		self.wfid_   = mpio.GenericWindowReader(self.param_.source)
		self.numIm_  = self.wfid_.numIm_
		self.lblSz_  = self.wfid_.lblSz_
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
										self.param_.crop_size, self.param_.crop_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_, 1, 1)
		#Create the pool
		self.pool_ = Pool(processes=self.numIm_)
		self.launch_jobs()
		
	def load_images(self, imNames, jobid):
		imData = np.zeros((self.param_.batch_size, self.ch_,
							self.param_.crop_size, self.param_.crop_size), np.float32)
		for b in range(self.param_.batch_size):
			#Load images
			imName, ch, h, w, x1, y1, x2, y2 = imNames[b].strip().split()
			imName = osp.join(self.param_.root_folder, imName)
			im     = scm.imread(imName)
			im     = im.astype(np.float32)
			#Process the image
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			#Feed the image	
			cSt = n * self.ch_
			cEn = cSt + self.ch_
			imData[b,cSt:cEn, :, :] = im[y1:y2, x1:x2,[2,1,0]].transpose((2,0,1)) 
		return [jobid, imData]

	def _load_images(self, args):
		return self.load_images(*args) 

	def launch_jobs(self):
		inArgs = []
		self.labels_ = np.zeros((self.param_.batch_size, self.lblSz_,1,1),np.float32)
		for n in range(self.numIm_):
			inArgs.append([])
		for b in range(self.param_.batch_size):
			imNames, lbls = self.wfid_.read_next()
			self.labels_[b,:,:,] = lbls.reshape(self.lblSz_,1,1).astype(np.float32) 
			#Read images
			for n in range(self.numIm_):
				inArgs[n].append([imNames[n], n])
		self.jobs_ = self.pool_.map_async(self._load_images, inArgs)
	
	def forward(self, bottom, top):
		#Load the images
		imRes      = self.jobs_.get()
		for res in imRes:
			jobId, imData = res
			cSt = jobId * self.ch_
			cEn = cSt + self.ch_
			top[0][:,cSt:cEn,:,:] = imData
		#Read the labels
		top[1].data[...] = self.labels_
		self.launch_jobs()

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
