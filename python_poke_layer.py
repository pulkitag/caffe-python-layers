import caffe
import numpy as np
import argparse, pprint
from multiprocessing import Pool
import scipy.misc as scm
from os import path as osp
import my_pycaffe_io as mpio
import my_pycaffe as mp
from easydict import EasyDict as edict
import time
import glog
import pdb

def get_jitter(self, coords):
	dx, dy = 0, 0
	if self.param_.jitter_amt > 0:
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * self.param_.jitter_amt, ry * self.param_.jitter_amt
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	
	if self.param_.jitter_pct > 0:
		h, w = [], []
		for n in range(len(coords)):
			x1, y1, x2, y2 = coords[n]
			h.append(y2 - y1)
			w.append(x2 - x1)
		mnH, mnW = min(h), min(w)
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * mnW * self.param_.jitter_pct, ry * mnH * self.param_.jitter_pct
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	return int(dx), int(dy)	


def get_crop_coords(poke, H, W, crpSz):
	#Trivial cropping
	yMid, xMid = H/2, W/2
	y1, x1     = yMid - crpSz/2, xMid - crpSz/2
	y2, x2     = y1 + crpSz, x1 + crpSz
	return x1, y1, x2, y2 	

def image_reader_keys(dbNames, dbKeys, crpSz, isCrop, isGray=False):
	bk,  ak,  pk   = dbKeys
	db  = mpio.MultiDbReader(dbNames)
	N   = len(bk)
	ims   = np.zeros((N, 6, crpSz, crpSz)).astype(np.float32)
	pokes = np.zeros((N, 3, 1, 1)).astype(np.float32) 
	for i in range(N):
		im1, im2, poke = db.read_key([bk[i], ak[i], pk[i]])
		im1  = im1.transpose((0,2,1))
		im2  = im2.transpose((0,2,1))
		poke = poke.reshape((1,3,1,1))
		if isCrop:
			H, W, _ = im1.shape
			x1, y1, x2, y2 = get_crop_coords(poke.squeeze(), H, W, crpSz)
			ims[i, 0:3] = im1[:, y1:y2, x1:x2]
			ims[i, 3:6] = im2[:, y1:y2, x1:x2]
		else:
			ims[i, 0:3] = scm.imresize(im1, (crpSz, crpSz))
			ims[i, 3:6] = scm.imresize(im2, (crpSz, crpSz))
		pokes[i][...] = poke[...]
	db.close()
	return ims, pokes

	
class PythonPokeLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonPokeLayer')
		parser.add_argument('--before', default='', type=str)
		parser.add_argument('--after',  default='', type=str)
		parser.add_argument('--poke',  default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--mean_file', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--is_mirror',  dest='is_mirror', action='store_true', default=False)
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--jitter_pct', default=0, type=float)
		parser.add_argument('--jitter_amt', default=0, type=int)
		parser.add_argument('--ncpu', default=2, type=int)
		parser.add_argument('--randSeed', default=3, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.pool_.terminate()

	def load_mean(self):
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

	def setup(self, bottom, top):
		self.param_ = PythonPokeLayer.parse_args(self.param_str)
		rf  = self.params_.root_folder
		self.dbNames_ = [osp.join(rf, self.params_.before),
									   osp.join(rf, self.params_.after),
									   osp.join(rf, self.params_.poke)] 
		#Read Keys
		self.dbKeys_ = []
		for name in self.dbNames_:
			db   = mpio.DbReader(name)
			keys = db.get_key_all()
			self.dbKeys_.append(keys)
			db.close()
			del db 	
		self.stKey_  = 0
		self.numKey_ = len(self.dbKeys_[0]) 	
		#Poke layer has 2 input images
		self.numIm_ = 2
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
										self.param_.crop_size, self.param_.crop_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_, 1, 1)
		#Load the mean
		self.load_mean()
		#If needed to resume	
		if self.param_.resume_iter > 0:
			N = self.param_.resume_iter * self.param_.batch_size
			N = np.mod(N, self.wfid_.num_)
			print ('SKIPPING AHEAD BY %d out of %d examples,
						  BECAUSE resume_iter is NOT 0'\
							% (N, self.wfid_.num_))
		#Create the pool
		self.pool_ = Pool(processes=self.num_param_.ncpu)
		self.jobs_ = []
	
		#Storing the image data	
		self.imData_ = np.zeros((self.param_.batch_size, 
						self.numIm_ * self.ch_,
						self.param_.crop_size, self.param_.crop_size), np.float32)
		self.labels_ = np.zeros((self.param_.batch_size, 
						self.lblSz_,1,1),np.float32)
		#Function to read the images
		self.readfn_ = image_reader_list
		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

	
	def launch_jobs(self):
		argList = []
		enKey   = self.stKey_ + self.param_.batch_size
		if  enKey > self.numKey_:
			wrap = np.mod(enKey, self.numKey_)
			keys = range(self.stKey_, self.param_.batch_size) + 
						 range(wrap)
			self.stKey_ = wrap
		else:
			keys = range(self.stKey_, enKey)
			self.stKey_ = enKey		
		argList = [self.dbNames_, [self.dbKeys_[0][keys],
						   self.dbKeys_[1][keys], self.dbKeys_[2][keys]], 
							 self.param_.crop_size, True, self.param_.is_gray]
		try:
			self.jobs_ = self.pool_.map_async(self.readfn_, argList)
		except KeyboardInterrupt:
			print 'Keyboard Interrupt received - terminating in launch jobs'
			self.pool_.terminate()	

	def get_prefetch_data(self):
		t1 = time.time()
		try:
			res      = self.jobs_.get()
			im, self.labels_[...]  = res
		except:
			print 'Keyboard Interrupt received - terminating'
			self.pool_.terminate()
			raise Exception('Error/Interrupt Encountered')	
		t2= time.time()
		tFetch = t2 - t1
		if self.mu_ is not None:	
			self.imData_[...] = im - self.mu_
		else:
			self.imData_[...] = im

	def forward(self, bottom, top):
		t1 = time.time()
		tDiff = t1 - self.t_
		#Load the images
		self.get_prefetch_data()
		top[0].data[...] = self.imData_
		t2 = time.time()
		tFetch = t2-t1
		#Read the labels
		top[1].data[:,:,:,:] = self.labels_
		self.launch_jobs()
		t2 = time.time()
		#print ('Forward took %fs in PythonWindowDataParallelLayer' % (t2-t1))
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass
	
	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
