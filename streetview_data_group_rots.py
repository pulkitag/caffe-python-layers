import caffe
import numpy as np
import argparse, pprint
from multiprocessing import Pool
import scipy.misc as scm
from os import path as osp
import my_pycaffe_io as mpio
import my_pycaffe as mp
from easydict import EasyDict as edict
from transforms3d.transforms3d import euler  as t3eu
import street_label_utils as slu
import time
import glog
import pdb
import pickle
import math
import copy
try:
	import cv2
except:
	print('OPEN CV not found, resorting to scipy.misc')

MODULE_PATH = osp.dirname(osp.realpath(__file__))

def get_jitter(coords=None, jitAmt=0, jitPct=0):
	dx, dy = 0, 0
	if jitAmt > 0:
		assert (jitPct == 0)
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * jitAmt, ry * jitAmt
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	
	if jitPct > 0:
		h, w = [], []
		for n in range(len(coords)):
			x1, y1, x2, y2 = coords[n]
			h.append(y2 - y1)
			w.append(x2 - x1)
		mnH, mnW = min(h), min(w)
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * mnW * jitPct, ry * mnH * jitPct
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	return int(dx), int(dy)	


def rotate_image(im, theta):
	'''
		theta: in degrees
	'''
	rows, cols, _ = im.shape
	M   = cv2.getRotationMatrix2D((cols/2,rows/2), theta, 1)
	dst = cv2.warpAffine(im, M, (cols, rows))
	return dst

def read_double_images(imName1, imName2, imPrms, rollJitter=None):
	imSz, cropSz = imPrms['imSz'], imPrms['cropSz']
	jitPct  = imPrms['jitter_pct']
	jitAmt  = imPrms['jitter_amt']
	im     = []
	#Read the images:
	try:
		if rollJitter is None:
			im.append(cv2.imread(imName1))
		else:
			r1  = rollJitter[0]
			img = cv2.imread(imName1)
			im.append(rotate_image(img, r1)) 
	except:
		raise Exception('Image %s read incorrectly' % imName1)
	try:
		if rollJitter is None:
			im.append(cv2.imread(imName2))
		else:
			r2  = rollJitter[1]
			img = cv2.imread(imName2)
			im.append(rotate_image(img, r2)) 
	except:
		raise Exception('Image %s read incorrectly' % imName2)
	try:
		h1, w1, ch1 = im[0].shape
		h2, w2, ch2 = im[1].shape
		assert ch1==ch2
	except:
		print (im[0].shape)
		print (im[1].shape)
		raise Exception('Something is wrong in read image')
	ims    = np.concatenate(im, axis=2)
	#Crop the images
	h, w, ch = ims.shape
	x1   = int(max(0, w/2.0 - cropSz/2.0))
	y1   = int(max(0, h/2.0 - cropSz/2.0))
	dx, dy = get_jitter(jitAmt=jitAmt, jitPct=jitPct)
	x1, y1 = max(0,x1 + dx), max(0,y1 + dy )
	x2, y2 = min(w, x1 + cropSz), min(h, y1 + cropSz)
	ims    = ims[y1:y2, x1:x2,:]	
	#Resize and transpose
	ims    = cv2.resize(ims,(imSz, imSz))
	ims    = ims.transpose((2,0,1))
	return ims


def get_rots(gp, imPrms, lbPrms, idx):
	'''
		gp    : group
		lbPrms: parameter for computing the labels
		idx   : tuple (n1,n2) indicating which grp elements
            to extract 
	'''
	rollMax = imPrms['random_roll_max']
	if rollMax == 0:
		rollJitter = None
	else:
		rollJitter = slu.get_roll_jitter(rollMax)
	
	n1, n2 = idx
	if rollJitter is None:
		r1, r2 = gp.data[n1].rots, gp.data[n2].rots
	else:
		r1 = copy.deepcopy(gp.data[n1].rots)
		r2 = copy.deepcopy(gp.data[n2].rots)
		r1[2] = r1[2] + rollJitter[0]
		r2[2] = r2[2] + rollJitter[1]	
	lb     = slu.get_normalized_pose_delta(lbPrms, r1, r2,
						pt1=gp.data[n1].pts.camera, pt2=gp.data[n2].pts.camera,
						debugMode=lbPrms['debugMode'])
	lb     = np.array(lb)
	imFolder = imPrms['imRootFolder'] % gp.folderId
	imName1  = osp.join(imFolder, gp.crpImNames[n1])
	imName2  = osp.join(imFolder, gp.crpImNames[n2])
	im     = read_double_images(imName1, imName2, imPrms, rollJitter=rollJitter)
	return im, lb			

#Sample which image pair to chose from the group
def sample_within_group(gp, lbPrms):
	if gp.num == 1:
		print ('WARNING: Only 1 element in the group')
	l1 = np.random.permutation(gp.num)
	l2 = np.random.permutation(gp.num)
	done = False
	for n1 in l1:
		for n2 in l2:
			#Sample the same image rarely
			if n1 == n2:
				rnd = np.random.random()
				if rnd < 0.85:
					continue
			lb  = slu.get_pose_delta_clip(lbPrms, gp.data[n1].rots,
						  gp.data[n2].rots,
							pt1=gp.data[n1].pts.camera, pt2=gp.data[n2].pts.camera,
              debugMode=lbPrms['debugMode'])
			if lb is None:
				done = False
				continue
			else:
				done = True 
				break
		if done:
			break
	#If valid label is found
	if done:
		return n1, n2
	else:
		return None, None
		

def read_groups(args):
	grp, fetchPrms, lbPrms, idx = args
	if lbPrms['type'] == 'pose':
		im, lb = get_rots(grp, fetchPrms, lbPrms, idx)		
	else:
		raise Exception('Label type %s not recognized' % lbPrms['type'])
	return (im, lb)
	

##
#Read data directly from groups
class PythonGroupDataRotsLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonGroupDataRots Layer')
		parser.add_argument('--im_root_folder', default='', type=str)
		#The file which contains the name of groups
		parser.add_argument('--grplist_file', default='', type=str)
		#File containing information what kind of labels
    #should be extractee etc.
		parser.add_argument('--lbinfo_file', default='', type=str)
		parser.add_argument('--mean_file', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--im_size', default=101, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--random_roll_max', default=0, type=float)
		parser.add_argument('--is_mirror',  dest='is_mirror', action='store_true', default=False)
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--jitter_pct', default=0, type=float)
		parser.add_argument('--jitter_amt', default=0, type=int)
		parser.add_argument('--nrmlz_file', default='None', type=str)
		parser.add_argument('--ncpu', default=2, type=int)
		#For debugging - load a single group
		parser.add_argument('--is_single_grp', dest='is_single_grp',
                        action='store_true', default=False )
		parser.add_argument('--no-is_single_grp', dest='is_single_grp', action='store_false')
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.pool_.terminate()
		del self.jobs_

	def load_mean(self):
		self.mu_ = None
		if self.param_.mean_file == 'None':
			self.mu_ = 128 * np.ones((6, self.param_.im_size, self.param_.im_size), np.float32)
		else:
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
		#Initialize the parameters
		self.param_  = PythonGroupDataRotsLayer.parse_args(self.param_str) 
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
				
		#debug mode
		self.debugMode_ = False
	
		#Read the groups
		print ('Loading Group Data')
		grpNameDat = pickle.load(open(self.param_.grplist_file, 'r'))	
		grpFiles   = grpNameDat['grpFiles']
		self.grpDat_   = []
		self.grpCount_ = []
		numGrp         = 0
		if self.param_.is_single_grp:
			grpFiles = [grpFiles[0]]	 
		for i,g in enumerate(grpFiles):
			self.grpDat_.append(pickle.load(open(g, 'r'))['groups'])
			self.grpCount_.append(len(self.grpDat_[i]))
			print ('Groups in %s: %d' % (g, self.grpCount_[i]))
			numGrp += self.grpCount_[i]
		print ('Total number of groups: %d' % numGrp)
		self.grpSampleProb_ = [float(i)/float(numGrp) for i in self.grpCount_]	
		print (self.grpSampleProb_)
		print (np.sum(np.array(self.grpSampleProb_)))	

		#Define the parameters required to read data
		self.fetchPrms_ = {}
		self.fetchPrms_['isMirror'] = self.param_.is_mirror
		self.fetchPrms_['isGray']   = self.param_.is_gray
		self.fetchPrms_['cropSz'] = self.param_.crop_size
		self.fetchPrms_['imSz']   = self.param_.im_size
		self.fetchPrms_['imRootFolder'] = self.param_.im_root_folder
		self.fetchPrms_['jitter_pct'] = self.param_.jitter_pct
		self.fetchPrms_['jitter_amt'] = self.param_.jitter_amt
		self.fetchPrms_['random_roll_max'] = self.param_.random_roll_max
		self.fetchPrms_['debugMode'] = self.debugMode_
	
		#Parameters that define how labels should be computed
		lbDat = pickle.load(open(self.param_.lbinfo_file))
		self.lbPrms_ = lbDat['lbInfo']
		self.lbPrms_['debugMode'] = self.debugMode_
		self.lblSz_  = self.lbPrms_['lbSz']
		if self.lbPrms_['nrmlz'] is not None:
			nrmlzDat = pickle.load(open(self.lbPrms_['statsFile'], 'r'))
			self.lbPrms_['nrmlzDat']  = edict()
			self.lbPrms_['nrmlzDat']['mu'] = nrmlzDat['mu']
			self.lbPrms_['nrmlzDat']['sd'] = nrmlzDat['sd']
			print (self.lbPrms_)	
		if self.debugMode_:
			self.lblSz_ += 3	
	
		top[0].reshape(self.param_.batch_size, 2 * self.ch_,
										self.param_.im_size, self.param_.im_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_ + 1, 1, 1)
		#Load the mean
		self.load_mean()
	
		#Create pool
		if self.param_.ncpu > 0:
			self.pool_ = Pool(processes=self.param_.ncpu)
			self.jobs_ = None
		
		#placeholders for data
		self.imData_ = np.zeros((self.param_.batch_size, 2 * self.ch_,
						self.param_.im_size, self.param_.im_size), np.float32)
		self.labels_ = np.ones((self.param_.batch_size, self.lblSz_ + 1,1,1),np.float32)

		#Which functions to use for reading images
		if 'cv2' in globals():
			print('OPEN CV FOUND')
			self.readfn_ = read_groups
		else:
			print('OPEN CV NOT FOUND, USING SCM')
			self.readfn_ = read_groups

		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

	#Launch jobs	
	def launch_jobs(self):
		self.argList = []
		#Form the list of groups that should be used
		for b in range(self.param_.batch_size):
			count = 0
			while True:
				count += 1
				rand   =  np.random.multinomial(1, self.grpSampleProb_)
				grpIdx =  np.where(rand==1)[0][0]
				ng     =  np.random.randint(low=0, high=self.grpCount_[grpIdx])
				n1, n2 =  sample_within_group(self.grpDat_[grpIdx][ng], self.lbPrms_)
				if n1 is not None:
					break
				if np.mod(count,100) == 1 and count > 1:
					print ('TRIED %d times, cannot find a sample' % count)
			self.argList.append([self.grpDat_[grpIdx][ng], self.fetchPrms_, self.lbPrms_, (n1,n2)])
		if self.param_.ncpu > 0:
			#Launch the jobs
			try:
				self.jobs_ = self.pool_.map_async(self.readfn_, self.argList)
			except KeyboardInterrupt:
				self.pool_.terminate()
				print 'Keyboard Interrupt received - terminating in launch jobs'
				raise Exception('Error/Interrupt Encountered')

	def normalize_labels(self):
		pass

	def get_prefetch_data(self):
		t1 = time.time()
		if self.param_.ncpu > 0:
			try:
				imRes      = self.jobs_.get(20)
			except:
				self.pool_.terminate()
				raise Exception('Error/Interrupt Encountered')
		else:
			#print (self.argList[0])
			imRes = []
			for b in range(self.param_.batch_size):
				imRes.append(self.readfn_(self.argList[b]))
			#pdb.set_trace()
		t2= time.time()
		tFetch = t2 - t1
		bCount = 0
		for b in range(self.param_.batch_size):
			if imRes[b][1] is not None:
				if self.mu_ is not None:	
					self.imData_[b,:,:,:] = imRes[b][0] - self.mu_
				else:
					self.imData_[b,:,:,:] = imRes[b][0]
				#print (imRes[b][1].shape)
				self.labels_[b,0:self.lblSz_,:,:] = imRes[b][1].reshape(1,self.lblSz_,1,1).astype(np.float32)
				bCount += 1
			else:
				#Donot use the label, image pair
				self.imData_[b,:,:,:] = 0.
				self.labels_[b,:,:,:] = 0.
		print ('Number of valid images in batch: %d' % bCount)
		self.normalize_labels()	
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
		top[1].data[...] = self.labels_
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

def test_group_rots(isPlot=True, debugMode=True):
	import vis_utils as vu
	import matplotlib.pyplot as plt
	fig     = plt.figure()
	defFile = osp.join(MODULE_PATH, 'test/test_group_rots.prototxt')
	net     = caffe.Net(defFile, caffe.TEST)
	while True:
		data   = net.forward(blobs=['pair_data', 'pose_label'])
		im, pk = data['pair_data'], data['pose_label']
		im     += 128
		im     = im.astype(np.uint8)
		if isPlot:
			for b in range(10):
				if debugMode:
					rots = tuple(pk[b].squeeze())[0:6]
				else:
					rots = tuple(pk[b].squeeze())[0:3]
				rots = [(r * 180.)/np.pi for r in rots]
				figTitle = 'yaw: %f,  pitch: %f, roll: %f \n yaw: %f, pitch:%f, roll: %f'\
             % (rots[0], rots[1], rots[2], rots[3], rots[4], rots[5])
				ax = vu.plot_pairs(im[b,0:3], im[b,3:6], isBlobFormat=True,
             chSwap=(2,1,0), fig=fig, figTitle=figTitle)
				plt.draw()
				plt.show()
				ip = raw_input()
				if ip == 'q':
					return	


