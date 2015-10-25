import caffe
import numpy as np
import argparse, pprint
from multiprocessing import Pool
import scipy.misc as scm

class PythonWindowDataLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Python Window Data Layer')
		parser.add_argument('--source', default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', default=False, type=bool)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
	
	def setup(self, bottom, top):
		self.params_ = PythonWindowDataLayer.parse_args(self.param_str) 
		self.wfid_   = mpio.GenericWindowReader(self.param_.root_folder)
		self.numIm_  = self.wfid_.numIm_
		self.lblSz_  = self.wfid_.lblSz_
		if self.params_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
										self.param_.crop_size, self.param_.crop_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_, 1, 1)

	def forward(self, bottom, top):
		for b in range(self.param_.batch_size):
			imNames, lbls = self.wfid_.read_next()
			#Read images
			for n in range(self.numIm_):
				#Load images
				imName, ch, h, w, x1, y1, x2, y2 = imNames[n].strip().split()
				imName = osp.join(self.param_.root_folder, imName)
				im     = scm.imread(imName)
				im     = im.astype(float)
				#Process the image
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				#Feed the image	
				cSt = n * self.ch_
				cEn = cSt + self.ch_
				top[0].data[b,cSt:cEn, :, :] = im[ySt:yEn, xSt:xEn,:].transpose((2,0,1)) 
			#Read the labels
			top[1].data[b,:,:,:] = lbls.reshape(self.lblSz_,1,1).astype(float) 

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
