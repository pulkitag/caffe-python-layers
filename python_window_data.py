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
import cv2


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
		parser.add_argument('--resume_iter', default=0, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

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
		self.load_mean()
		#Skip the number of examples so that the same examples
		#are not read back
		if self.param_.resume_iter > 0:
			N = self.param_.resume_iter * self.param_.batch_size
			N = np.mod(N, self.wl_.num_)
			for n in range(N):
				_, _ = self.read_next()	

	def forward(self, bottom, top):
		t1 = time.time()
		tIm, tProc = 0, 0
		for b in range(self.param_.batch_size):
			if self.wfid_.is_eof():	
				self.wfid_.close()
				self.wfid_   = mpio.GenericWindowReader(self.param_.source)
				print ('RESTARTING READ WINDOW FILE')
			imNames, lbls = self.wfid_.read_next()
			#Read images
			for n in range(self.numIm_):
				#Load images
				imName, ch, h, w, x1, y1, x2, y2 = imNames[n].strip().split()
				imName = osp.join(self.param_.root_folder, imName)
				tImSt = time.time() 
				im     = cv2.imread(imName)
				tImEn = time.time() 
				tIm += (tImEn - tImSt)
				#Process the image
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				#Feed the image	
				cSt = n * self.ch_
				cEn = cSt + self.ch_
				im = cv2.resize(im[y1:y2, x1:x2, :],
									(self.param_.crop_size, self.param_.crop_size))
				im = im.transpose((2,0,1))
				if self.mu_ is not None:
					im = im - self.mu_
				top[0].data[b,cSt:cEn, :, :] = im.astype(np.float32)
				tEn = time.time()
				tProc += (tEn - tImEn) 
			#Read the labels
			top[1].data[b,:,:,:] = lbls.reshape(self.lblSz_,1,1).astype(np.float32) 
		t2 = time.time()
		print ('Forward: %fs, Reading: %fs, Processing: %fs' % (t2-t1, tIm, tProc))

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass


def image_reader(args):
	imName, imDims, cropSz, imNum = args
	im     = cv2.imread(imName)
	#Process the image
	x1, y1, x2, y2 = imDims
	im = cv2.resize(im[y1:y2, x1:x2, :],
						(cropSz, cropSz))
	im = im.transpose((2,0,1))
	return (im, imNum)


class WindowLoader(object):
	def __init__(self, root_folder, batch_size, channels,
							 crop_size, mu=None, poolsz=None):
		self.root_folder = root_folder
		self.batch_size  = batch_size
		self.ch    		   = channels 
		self.crop_size   = crop_size
		self.mu          = mu
		self.pool_       = poolsz
	
	def load_images(self, imNames, jobid):
		if self.pool_ is not None:
			return self.load_images_parallel(imNames, jobid)
		imData = np.zeros((self.batch_size, self.ch,
							self.crop_size, self.crop_size), np.float32)
		for b in range(self.batch_size):
			#Load images
			imName, ch, h, w, x1, y1, x2, y2 = imNames[b].strip().split()
			imName = osp.join(self.root_folder, imName)
			#Gives BGR
			im     = cv2.imread(imName)
			#Process the image
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			im = cv2.resize(im[y1:y2, x1:x2, :],
									(self.crop_size, self.crop_size))
			im = im.transpose((2,0,1))
			imData[b,:, :, :] = im
		#Subtract the mean if needed
		if self.mu is not None:
			imData = imData - self.mu
		imData = imData.astype(np.float32)
		return jobid, imData

	def load_images_parallel(self, imNames, jobid):
		imData = np.zeros((self.batch_size, self.ch,
							self.crop_size, self.crop_size), np.float32)
		inArgs = []
		for b in range(self.batch_size):
			#Load images
			imName, ch, h, w, x1, y1, x2, y2 = imNames[b].strip().split()
			imName = osp.join(self.root_folder, imName)
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			inArgs.append([imName, (x1, y1, x2, y2), self.crop_size, b])	
		jobs = READ_POOL.map_async(image_reader, inArgs)
		res  = jobs.get()
		for r in res:
			imData[r[1],:,:,:] = r[0]
		#Subtract the mean if needed
		if self.mu is not None:
			imData = imData - self.mu
		imData = imData.astype(np.float32)
		return jobid, imData


def _load_images(args):
	self, imNames, jobId = args
	return self.load_images(imNames, jobId) 

##
#Parallel version
class PythonWindowDataParallelLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonWindowDataParallel Layer')
		parser.add_argument('--source', default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--mean_file', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--resume_iter', default=0, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.wfid_.close()
		for n in self.numIm_:
			self.pool_[n].terminate()

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
		#Load the mean
		self.load_mean()	
		#Create the window loader
		self.wl_ = WindowLoader(self.param_.root_folder,
								self.param_.batch_size, self.ch_, 
								self.param_.crop_size, self.mu_, poolsz=4)
		#Skip the number of examples so that the same examples
		#are not read back
		if self.param_.resume_iter > 0:
			N = self.param_.resume_iter * self.param_.batch_size
			N = np.mod(N, self.wl_.num_)
			for n in range(N):
				_, _ = self.read_next()	

		#Create the pool
		self.pool_, self.jobs_ = [], []
		for n in range(self.numIm_):
			self.pool_.append(Pool(processes=8))
			self.jobs_.append([])
		self.launch_jobs()
		self.t_ = time.time()	
		
		self.imData_ = np.zeros((self.param_.batch_size, self.numIm_ * self.ch_,
						self.param_.crop_size, self.param_.crop_size), np.float32)
	
	def launch_jobs(self):
		argList = []
		for n in range(self.numIm_):
			argList.append([])
		self.labels_ = np.zeros((self.param_.batch_size, self.lblSz_,1,1),np.float32)
		#Form the list of images and labels
		for b in range(self.param_.batch_size):
			if self.wfid_.is_eof():	
				self.wfid_.close()
				self.wfid_   = mpio.GenericWindowReader(self.param_.source)
				print ('RESTARTING READ WINDOW FILE')
			imNames, lbls = self.wfid_.read_next()
			self.labels_[b,:,:,:] = lbls.reshape(self.lblSz_,1,1).astype(np.float32) 
			#Read images
			for n in range(self.numIm_):
				fName, ch, h, w, x1, y1, x2, y2 = imNames[n].strip().split()
				fName  = osp.join(self.param_.root_folder, fName)
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				argList[n].append([fName, (x1,y1,x2,y2), self.param_.crop_size, b])
		#Launch the jobs
		for n in range(self.numIm_):
			try:
				self.jobs_[n] = self.pool_[n].map_async(image_reader, argList[n])
			except KeyboardInterrupt:
				print 'Keyboard Interrupt received - terminating'
				self.pool_[n].terminate()	

	def get_prefetch_data(self):
		for n in range(self.numIm_):
			cSt = n * self.ch_
			cEn = cSt + self.ch_
			t1 = time.time()
			try:
				imRes      = self.jobs_[n].get()
			except KeyboardInterrupt:
				print 'Keyboard Interrupt received - terminating'
				self.pool_[n].terminate()	
			t2= time.time()
			tFetch = t2 - t1
			for res in imRes:
				if self.mu_ is not None:	
					self.imData_[res[1],cSt:cEn,:,:] = res[0] - self.mu_
				else:
					self.imData_[res[1],cSt:cEn,:,:] = res[0]
			#print ('%d, Fetching: %f, Copying: %f' % (n, tFetch, time.time()-t2))
			#glog.info('%d, Fetching: %f, Copying: %f' % (n, tFetch, time.time()-t2))
					
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
		#glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
	
	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass

	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass
