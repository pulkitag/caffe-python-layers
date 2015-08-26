import caffe
import numpy as np
import argparse, pprint

class TryLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='Try Layer')
		parser.add_argument('--num_classes', default=20, type=int)
		parser.add_argument('--aa', default=5, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	
	
	def setup(self, bottom, top):
		self.params_ = TryLayer.parse_args(self.param_str) 
		top[0].reshape(2,2)

	def forward(self, bottom, top):
		top[0].data[...] = top[0].data + self.params_.aa * np.ones(top[0].shape)	
	
	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
