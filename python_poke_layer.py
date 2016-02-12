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
import pickle
import copy

def get_crop_coords(poke, H, W, crpSz, maxJitter=100):
	'''
		Crop a size of crpSz while assuring that the poke point is 
		inside a central box of side maxJitter in the crop
	'''
	maxJitter = min(maxJitter, crpSz)
	x1 = round(max(0, poke[0] - (crpSz -  maxJitter)/2 - maxJitter))
	x2 = max(x1, round(min(W - crpSz, max(0, poke[0] - (crpSz - maxJitter)/2))))
	y1 = round(max(0, poke[1] - (crpSz -  maxJitter)/2 - maxJitter))
	y2 = max(y1, round(min(H - crpSz, max(0, poke[1] - (crpSz - maxJitter)/2))))
	ySt   = int(np.random.random() * (y2 - y1) + y1)
	xSt   = int(np.random.random() * (x2 - x1) + x1)
	xEn, yEn = xSt + crpSz, ySt + crpSz
	pk     = [poke[0] - xSt, poke[1] - ySt, poke[2]]
	return xSt, ySt, xEn, yEn, pk


#ims   = np.zeros((128, 6, 192, 192)).astype(np.float32)
#pokes = np.zeros((128, 3, 1, 1)).astype(np.float32) 

def find_bin(val, bins):
	bnIdx = np.where(val >=bins)[0]
	if len(bnIdx) == 0:
		bnIdx = 0
	else:
		bnIdx = bnIdx[-1]
	return bnIdx

def transform_poke(pk, **kwargs):
	if kwargs['tfmType'] is None:
		#Normalize pokes to range 0, 1
		pk = copy.deepcopy(pk)
		crpSz = kwargs['crpSz']
		pk[0]  = (pk[0] - crpSz/2.0)/float(crpSz)
		pk[1]  = (pk[1] - crpSz/2.0)/float(crpSz)
		pk[2]  = pk[2] - np.pi/2
		return pk
	elif kwargs['tfmType'] == 'gridCls':
		xBins, yBins, tBins = kwargs['xBins'], kwargs['yBins'], kwargs['tBins']
		xIdx = np.where(pk[0] >= xBins)[0]
		yIdx = np.where(pk[1] >= yBins)[0]
		tIdx = np.where(pk[2] >= tBins)[0]
		if len(xIdx) == 0:
			print('SOMETHING IS WRONG in x: %f' %  pk[0])
		if len(yIdx) == 0:
			print('SOMETHING IS WRONG in y: %f' %  pk[1])
		if len(tIdx) == 0:
			print('SOMETHING IS WRONG in th: %f' %  pk[2])

		xBn = find_bin(pk[0], xBins)
		yBn = find_bin(pk[1], yBins)
		tBn = find_bin(pk[2], tBins)
		nX, nY = kwargs['nX'], kwargs['nY']
		#print ('PokeBin, %d, %d' % (xBn * xBins[1], yBn * yBins[1]))
		bnIdx  = yBn * nY + xBn
		yInfer = int(np.floor(bnIdx / float(len(xBins))))
		xInfer = int(np.mod(bnIdx,len(xBins))) 
		assert yInfer == yBn and xInfer==xBn
		return (bnIdx, tBn)
	elif kwargs['tfmType'] == 'gridCls_loc_debug':
		kwargs = copy.deepcopy(kwargs)
		crpSz = kwargs['crpSz']
		kwargs['tfmType'] = None
		pokePoint = transform_poke(pk, **kwargs)
		pokePoint[0] = pokePoint[0] * crpSz + crpSz/2.0	
		pokePoint[1] = pokePoint[1] * crpSz + crpSz/2.0	
		#print ('PokePoint: %d, %d' % (pokePoint[0], pokePoint[1]))
		kwargs['tfmType'] = 'gridCls'
		pokeCls   = transform_poke(pk, **kwargs)
		return list(pokePoint) + list(pokeCls)
		

#def image_reader_keys(dbNames, dbKeys, crpSz, isCrop, isGray=False):
def image_reader_keys(*args):
	dbNames, dbKeys, crpSz, isCrop, isGray, pkTfm = args
	t1 = time.time()
	bk,  ak,  pk   = dbKeys
	db  = mpio.MultiDbReader(dbNames)
	t15 = time.time()
	openTime = t15 - t1
	N   = len(bk)
	ims   = np.zeros((N, 6, crpSz, crpSz), np.uint8)
	pokes = np.zeros((N, pkTfm['lblSz'], 1, 1), np.float32)
	t2  = time.time()
	preTime  = t2 - t15
	readTime, procTime, tranTime, cropFnTime, cropTime = 0, 0, 0, 0, 0
	for i in range(N):
		t3 = time.time()
		im1, im2, poke = db.read_key([bk[i], ak[i], pk[i]])
		t4 = time.time()
		readTime += t4 - t3
		im1  = im1.transpose((0,2,1))
		im2  = im2.transpose((0,2,1))
		poke = poke.reshape((1,3,1,1))
		t45  = time.time()
		tranTime += t45 - t4
		if isCrop:
			_, H, W = im1.shape
			x1, y1, x2, y2, newPoke = get_crop_coords(poke.squeeze(), H, W, crpSz)
			t47 = time.time()
			cropFnTime += t47 - t45
			ims[i, 0:3] = im1[:, y1:y2, x1:x2]
			ims[i, 3:6] = im2[:, y1:y2, x1:x2]
			t48 = time.time()
			cropTime += t48 - t47
		else:
			ims[i, 0:3] = scm.imresize(im1, (crpSz, crpSz))
			ims[i, 3:6] = scm.imresize(im2, (crpSz, crpSz))
		tfmPoke = transform_poke(newPoke, **pkTfm)
		#print tfmPoke 
		pokes[i][...] = np.array(tfmPoke).reshape((pkTfm['lblSz'],1,1))
		t5 = time.time()
		procTime += t5 - t4
	#db.close()
	print '#####################'
	print 'Open-Time: %f, Pre-Time: %f, Read-Time: %f, Proc-Time: %f' % (openTime, preTime, readTime, procTime)
	print 'CropFnTime: %f, Crop-Time: %f, Transpose Time: %f' % (cropFnTime, cropTime, tranTime)
	print '#####################'
	return ims, pokes

	
class PythonPokeLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonPokeLayer')
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--before', default='', type=str)
		parser.add_argument('--after',  default='', type=str)
		parser.add_argument('--poke',  default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--mean_file', default='None', type=str)
		parser.add_argument('--mean_type', default='3val', type=str)
		parser.add_argument('--poke_tfm_type', default='None', type=str)
		parser.add_argument('--poke_nxGrid', default=15, type=str)
		parser.add_argument('--poke_nyGrid', default=15, type=int)
		parser.add_argument('--poke_thGrid', default=15, type=int)
		parser.add_argument('--poke_mnTheta', default=np.pi/4, type=float)
		parser.add_argument('--poke_mxTheta', default=3*np.pi/4, type=float)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--is_mirror',  dest='is_mirror', action='store_true', default=False)
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--max_jitter', default=0, type=int)
		parser.add_argument('--is_prefetch', default=0, type=int)
		parser.add_argument('--randSeed', default=3, type=int)
		parser.add_argument('--is_debug', dest='is_gray', action='store_true', default=False)
		parser.add_argument('--no-is_debug', dest='is_gray', action='store_false')
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.pool_.terminate()

	def load_mean(self):
		self.mu_ = None
		if self.param_.mean_file is not None:
			print ('READING MEAN FROM %s', self.param_.mean_file)
			if self.param_.mean_file[-3:] == 'pkl':
				meanDat  = pickle.load(open(self.param_.mean_file, 'r'))
				self.mu_ = meanDat['mu'].astype(np.float32).transpose((2,0,1))
			else:
				#Mean is assumbed to be in BGR format
				self.mu_ = mp.read_mean(self.param_.mean_file)
				self.mu_ = self.mu_.astype(np.float32)
			if self.param_.mean_type == '3val':
				self.mu_   = np.mean(self.mu_, axis=(1,2)).reshape(1,3,1,1)
				self.mu_   = np.concatenate((self.mu_, self.mu_), axis=1)
			elif self.param_.mean_type == 'img':
				ch, h, w = self.mu_.shape
				assert (h >= self.param_.crop_size and w >= self.param_.crop_size)
				y1 = int(h/2 - (self.param_.crop_size/2))
				x1 = int(w/2 - (self.param_.crop_size/2))
				y2 = int(y1 + self.param_.crop_size)
				x2 = int(x1 + self.param_.crop_size)
				self.mu_ = self.mu_[:,y1:y2,x1:x2]
				self.mu_ = self.mu_.reshape((1,) + self.mu_.shape)
			else:
				raise Exception('Mean type %s not recognized' % self.param_.mean_type)

	def setup(self, bottom, top):
		self.param_ = PythonPokeLayer.parse_args(self.param_str)
		if self.param_.mean_file == 'None':
			self.param_.mean_file = None
		if self.param_.poke_tfm_type == 'None':
			self.param_.poke_tfm_type = None

		rf  = self.param_.root_folder
		self.dbNames_ = [osp.join(rf, self.param_.before),
									   osp.join(rf, self.param_.after),
									   osp.join(rf, self.param_.poke)] 
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
	
		self.pkParams_ = {}
		self.pkParams_['tfmType'] = self.param_.poke_tfm_type
		if self.param_.poke_tfm_type is None:
			self.pkParams_['xBins'], self.pkParams_['yBins'] = None, None
			self.pkParams_['tBins'] = None
			self.lblSz_ = 3
		elif self.param_.poke_tfm_type in ['gridCls', 'gridCls_loc_debug']:
			#Grid classification
			nXGrid, nYGrid = self.param_.poke_nxGrid, self.param_.poke_nyGrid
			thGrid         = self.param_.poke_thGrid
			nY, nX = self.param_.crop_size, self.param_.crop_size 
			self.pkParams_['xBins']   = np.linspace(0, nX, nXGrid)
			self.pkParams_['yBins']   = np.linspace(0, nY, nYGrid) 
			self.pkParams_['tBins']   = np.linspace(self.param_.poke_mnTheta,
																	self.param_.poke_mxTheta, thGrid)
			self.pkParams_['nX']      = nXGrid
			self.pkParams_['nY']      = nYGrid
			if self.param_.poke_tfm_type == 'gridCls':
				self.lblSz_ = 2
			else:
				self.lblSz_ = 5
		self.pkParams_['lblSz']  = self.lblSz_
		self.pkParams_['crpSz']  = self.param_.crop_size
		
		print self.pkParams_
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
			print ('SKIPPING AHEAD BY %d out of %d examples,\
						  BECAUSE resume_iter is NOT 0'\
							% (N, self.wfid_.num_))
		#Create the pool
		self.isPrefetch_ = bool(self.param_.is_prefetch)
		if self.isPrefetch_:
			self.pool_ = Pool(processes=1)
			self.jobs_ = []
	
		#Storing the image data	
		self.imData_ = np.zeros((self.param_.batch_size, 
						self.numIm_ * self.ch_,
						self.param_.crop_size, self.param_.crop_size), np.float32)
		self.labels_ = np.zeros((self.param_.batch_size, 
						self.lblSz_,1,1),np.float32)
		self.argList_ = []
		#Function to read the images
		self.readfn_ = image_reader_keys
		#Launch the prefetching
		if self.isPrefetch_:	
			self.launch_jobs()
		self.t_ = time.time()	

	def _make_arglist(self):
		self.argList_ = []
		enKey   = self.stKey_ + self.param_.batch_size
		if  enKey > self.numKey_:
			wrap = np.mod(enKey, self.numKey_)
			keys = range(self.stKey_, self.numKey_) +\
						 range(wrap)
			self.stKey_ = wrap
		else:
			keys = range(self.stKey_, enKey)
			self.stKey_ = enKey
		self.argList_ = [self.dbNames_, 
							 [[self.dbKeys_[0][k] for k in keys],
						   [self.dbKeys_[1][k] for k in keys],
							 [self.dbKeys_[2][k] for k in keys]],
               self.param_.crop_size, True, self.param_.is_gray,
							 self.pkParams_]

	def launch_jobs(self):
		self._make_arglist()
		try:
			print ('PREFETCH STARTED')
			self.jobs_ = self.pool_.map_async(self.readfn_, self.argList_)
		except KeyboardInterrupt:
			print 'Keyboard Interrupt received - terminating in launch jobs'
			self.pool_.terminate()	

	def get_prefetch_data(self):
		t1 = time.time()
		if self.isPrefetch_:
			try:
				print ('GETTING PREFECH')
				res      = self.jobs_.get()
				print ('PREFETCH GOT')	
				im, self.labels_[...]  = res
			except:
				print 'Keyboard Interrupt received - terminating'
				self.pool_.terminate()
				raise Exception('Error/Interrupt Encountered')
		else:
			self._make_arglist()
			im, self.labels_[...] = self.readfn_(*self.argList_)
			#self.readfn_(*self.argList_)
	
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
		if self.isPrefetch_:
			self.launch_jobs()
		t2 = time.time()
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass
	
	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass


def test_poke_layer_regression(isPlot=True):
	import vis_utils as vu
	import matplotlib.pyplot as plt
	fig     = plt.figure()
	defFile = 'test/poke_layer.prototxt'
	net     = caffe.Net(defFile, caffe.TEST)
	while True:
		data   = net.forward(blobs=['im', 'poke'])
		im, pk = data['im'], data['poke']
		if isPlot:
			for b in range(10):
				ax = vu.plot_pairs(im[b,0:3], im[b,3:6], isBlobFormat=True, chSwap=(2,1,0), fig=fig)
				ax[0].plot(int(pk[b][0]), int(pk[b][1]), markersize=10, marker='o')
				plt.draw()
				plt.show()
				ip = raw_input()
				if ip == 'q':
					return	

def test_poke_layer_cls(isPlot=True, debugMode=True):
	import vis_utils as vu
	import matplotlib.pyplot as plt
	fig     = plt.figure()
	if debugMode:
		defFile = 'test/poke_cls_debug_layer.prototxt'
	else:
		defFile = 'test/poke_cls_layer.prototxt'
	net     = caffe.Net(defFile, caffe.TEST)
	while True:
		data   = net.forward(blobs=['im', 'poke'])
		im, pk = data['im'], data['poke']
		shp    = im.shape
		print (shp)
		if isPlot:
			for b in range(1):
				pkb = pk[b].squeeze()
				if debugMode:
					x, y, pkBin = pkb[0], pkb[1], pkb[3]
					#x, y = x * shp[3], y * shp[2]
					#x, y = int(round(x + shp[3]/2.0)), int(round(y + shp[2]/2.0))
				else:
					pkBin = pk[b][0].squeeze()
				by    = int(np.floor(pkBin / np.float(15)))
				bx    = int(np.mod(pkBin, 15))
				pkIm  = np.zeros((15, 15, 3)).astype(np.uint8)
				print ('by, bx: %d, %d' % (int(by), int(bx)))
				pkIm[by, bx,:] = np.array((255,0,0))
				imBy, imBx     = by * 16.214, bx * 16.214
				pkIm  = scm.imresize(pkIm, (shp[2], shp[3]), interp='nearest')
				pkIm  = pkIm.transpose((2,0,1)).reshape((1,3, shp[2], shp[3]))
				imBefore = im[b, 0:3] / 2 + pkIm[0] / 2
				ax    = vu.plot_n_ims([imBefore, im[b,3:6], pkIm[0]],\
								 isBlobFormat=True, chSwap=(2,1,0), fig=fig)
				if debugMode:
					ax[2].plot(x, y, 'r', markersize=10, marker='o')
					ax[2].plot(int(imBx), int(imBy), 'g', markersize=10, marker='o')
				#print pk[b]
				plt.draw()
				plt.show()
				ip = raw_input()
				if ip == 'q':
					return	

