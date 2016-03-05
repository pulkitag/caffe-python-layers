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
try:
	import cv2
except:
	print('OPEN CV not found, resorting to scipy.misc')

MODULE_PATH = osp.dirname(osp.realpath(__file__))

IM_DATA = []

def get_jitter(coords):
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


def image_reader(args):
	imName, imDims, cropSz, imNum, isGray, isMirror = args
	x1, y1, x2, y2 = imDims
	im = cv2.imread(imName)
	im = cv2.resize(im[y1:y2, x1:x2, :],
						(cropSz, cropSz))
	if isMirror and np.random.random() >= 0.5:
		im = im[:,::-1,:]
	im = im.transpose((2,0,1))
	#glog.info('Processed')
	return (im, imNum)

def image_reader_list(args):
	outList = []
	for ag in args:
		imName, imDims, cropSz, imNum, isGray, isMirror = ag
		x1, y1, x2, y2 = imDims
		im = cv2.imread(imName)
		im = cv2.resize(im[y1:y2, x1:x2, :],
							(cropSz, cropSz))
		if isMirror and np.random.random() >= 0.5:
			im = im[:,::-1,:]
		outList.append((im.transpose((2,0,1)), imNum))
	#glog.info('Processed')
	return outList

def image_reader_scm(args):
	imName, imDims, cropSz, imNum, isGray, isMirror = args
	x1, y1, x2, y2 = imDims
	im = scm.imread(imName)
	im = scm.imresize(im[y1:y2, x1:x2, :],
						(cropSz, cropSz))
	if isMirror and np.random.random() >= 0.5:
		im = im[:,::-1,:]
	im = im[:,:,[2,1,0]].transpose((2,0,1))
	#glog.info('Processed')
	return (im, imNum)


def estimate_rot_labels(rlb, rollPrms):
	'''
		rlb: angles in radians
	'''
	pitch1, yaw1, roll1, x1, y1, z1, pitch2, yaw2, roll2, x2, y2, z2 = rlb
	pitchNz, yawNz, rollNz, xNz, yNz, zNz = rollPrms['nrmlzSd']
	pitchMu, yawMu, rollMu, xMu, yMu, zMu = rollPrms['nrmlzMu']
	if rollPrms['randomRoll']:
		roll1 = rollPrms['roll1']
		roll2 = rollPrms['roll2']
	rMat1 = t3eu.euler2mat(pitch1, yaw1, roll1, 'szxy')
	rMat2 = t3eu.euler2mat(pitch2, yaw2, roll2, 'szxy')
	dRot  = np.dot(rMat2, rMat1.transpose())
	pitch, yaw, roll = t3eu.euler2mat(dRot, 'szxy')
	x, y, z = x2 - x1, y2 - y1, z2 - z1
	pitch, yaw, roll = pitch - pitchMu, yaw - yawMu, roll - rollMu
	x, y, z = x - xMu, y - yMu, z - zMu
	return pitch/pitchNz, yaw/yawNz, roll/rollNz, x/zNz, y/yNz, z/zNz		
	
##
#Parallel version
class PythonWindowDataRotsLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonWindowDataRots Layer')
		parser.add_argument('--source', default='', type=str)
		parser.add_argument('--root_folder', default='', type=str)
		parser.add_argument('--mean_file', default='', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--is_random_roll',    dest='is_gray', action='store_true', default=True)
		parser.add_argument('--no-is_random_roll', dest='is_gray', action='store_false')
		parser.add_argument('--is_mirror',  dest='is_mirror', action='store_true', default=False)
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--jitter_pct', default=0, type=float)
		parser.add_argument('--jitter_amt', default=0, type=int)
		parser.add_argument('--nrmlz_file', default='None', type=str)
		parser.add_argument('--ncpu', default=2, type=int)
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
		self.param_ = PythonWindowDataRotsLayer.parse_args(self.param_str) 
		self.wfid_   = mpio.GenericWindowReader(self.param_.source)
		self.numIm_  = self.wfid_.numIm_
		self.lblSz_  = self.wfid_.lblSz_
		self.isV2    = False
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		#assert not self.param_.nrmlz_file == 'None'
		self.rotPrms_ = {}
		self.rotPrms_['randomRoll']   = self.param_.randomRoll
		self.rotPrms_['randomRollMx'] = self.param_.randomRollMx
		if self.param_.nrmlz_file:
			nrmlzDat = pickle.load(open(self.param_.nrmlz_file, 'r'))
			self.rotPrms_['nrmlzMu'] = nrmlzDat['nrmlzMu']
			self.rotPrms_['nrmlzSd'] = nrmlzDat['nrmlzSd'] 
			

		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
										self.param_.crop_size, self.param_.crop_size)
		top[1].reshape(self.param_.batch_size, self.lblSz_, 1, 1)
		#Load the mean
		self.load_mean()
		#If needed to resume	
		if self.param_.resume_iter > 0:
			N = self.param_.resume_iter * self.param_.batch_size
			N = np.mod(N, self.wfid_.num_)
			print ('SKIPPING AHEAD BY %d out of %d examples, BECAUSE resume_iter is NOT 0'\
							 % (N, self.wfid_.num_))
			for n in range(N):
				_, _ = self.wfid_.read_next()	
		#Create the pool
		self.pool_, self.jobs_ = [], []
		for n in range(self.numIm_):
			self.pool_.append(Pool(processes=self.param_.ncpu))
			self.jobs_.append([])
		
		self.imData_ = np.zeros((self.param_.batch_size, self.numIm_ * self.ch_,
						self.param_.crop_size, self.param_.crop_size), np.float32)
		if 'cv2' in globals():
			print('OPEN CV FOUND')
			if self.isV2:
				self.readfn_ = image_reader_list
			else:
				self.readfn_ = image_reader
		else:
			print('OPEN CV NOT FOUND, USING SCM')
			self.readfn_ = image_reader_scm
		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

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
				glog.info('RESTARTING READ WINDOW FILE')
			imNames, lbls = self.wfid_.read_next()
			self.labels_[b,:,:,:] = lbls.reshape(self.lblSz_,1,1).astype(np.float32) 
			#Read images
			fNames, coords = [], []
			for n in range(self.numIm_):
				fName, ch, h, w, x1, y1, x2, y2 = imNames[n].strip().split()
				fNames.append(osp.join(self.param_.root_folder, fName))
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				coords.append((x1, y1, x2, y2))
			#Computing jittering if required
			dx, dy = self.get_jitter(coords)
			for n in range(self.numIm_):
				fName = fNames[n]
				x1, y1, x2, y2 = coords[n]
				#Jitter the box
				x1 = max(0, x1 + dx)
				y1 = max(0, y1 + dy)
				x2 = min(w, x2 + dx)
				y2 = min(h, y2 + dy)
				#glog.info('%d, %d, %d, %d' % (x1, y1, x2, y2))
				argList[n].append([fName, (x1,y1,x2,y2), self.param_.crop_size,
									 b, self.param_.is_gray, self.param_.is_mirror])
		#Launch the jobs
		for n in range(self.numIm_):
			try:
				#print (argList[n])
				self.jobs_[n] = self.pool_[n].map_async(self.readfn_, argList[n])
			except KeyboardInterrupt:
				print 'Keyboard Interrupt received - terminating in launch jobs'
				self.pool_[n].terminate()	

	
	def get_prefetch_data(self):
		for n in range(self.numIm_):
			cSt = n * self.ch_
			cEn = cSt + self.ch_
			t1 = time.time()
			try:
				imRes      = self.jobs_[n].get()
			except:
				print 'Keyboard Interrupt received - terminating'
				self.pool_[n].terminate()
				#pdb.set_trace()
				raise Exception('Error/Interrupt Encountered')	
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
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()

	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass
	
	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass


def read_double_images(imName1, imName2, imPrms):
	imSz, cropSz = imPrms['imSz'], imPrms['cropSz']
	im     = []
	print (imName1, imName2)
	im.append(cv2.imread(imName1))
	im.append(cv2.imread(imName2))
	ims    = np.concatenate(im, axis=2)
	ims    = cv2.resize(ims,(imSz, imSz))
	ims    = ims.transpose((2,0,1))
	return ims

def get_rots(gp, imPrms, lbPrms):
	'''
		gp    : group
		lbPrms: parameter for computing the labels 
	'''
	lPerm  = np.random.permutation(gp.num)
	if len(lPerm) < 2:
		return (None, None)
	n1, n2 = lPerm[0], lPerm[1]
	lb     = slu.get_pose_delta(lbPrms, gp.data[n1].rots,
											gp.data[n2].rots,
							gp.data[n1].pts.camera, gp.data[n2].pts.camera)
	lb     = np.array(lb)
	imFolder = imPrms['imRootFolder'] % gp.folderId
	imName1  = osp.join(imFolder, gp.crpImNames[n1])
	imName2  = osp.join(imFolder, gp.crpImNames[n2])
	im     = read_double_images(imName1, imName2, imPrms)
	return im, lb			


def read_groups(args):
	grp, fetchPrms, lbPrms = args
	if lbPrms['type'] == 'pose':
		im, lb = get_rots(grp, fetchPrms, lbPrms)		
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
		parser.add_argument('--is_random_roll', dest='is_random_roll', 
                        action='store_true', default=False)
		parser.add_argument('--no-is_random_roll', dest='is_random_roll', action='store_false')
		parser.add_argument('--random_roll_max', default=0, type=float)
		parser.add_argument('--is_mirror',  dest='is_mirror', action='store_true', default=False)
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--jitter_pct', default=0, type=float)
		parser.add_argument('--jitter_amt', default=0, type=int)
		parser.add_argument('--nrmlz_file', default='None', type=str)
		parser.add_argument('--ncpu', default=2, type=int)
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
		#assert not self.param_.nrmlz_file == 'None'
		self.rotPrms_ = {}
		self.rotPrms_['randomRoll']   = self.param_.is_random_roll
		self.rotPrms_['randomRollMx'] = self.param_.random_roll_max
		if self.param_.nrmlz_file is not 'None':
			nrmlzDat = pickle.load(open(self.param_.nrmlz_file, 'r'))
			self.rotPrms_['nrmlzMu'] = nrmlzDat['nrmlzMu']
			self.rotPrms_['nrmlzSd'] = nrmlzDat['nrmlzSd'] 
			
		#Read the groups
		print ('Loading Group Data')
		grpNameDat = pickle.load(open(self.param_.grplist_file, 'r'))	
		grpFiles   = grpNameDat['grpFiles']
		self.grpDat_   = []
		self.grpCount_ = []
		numGrp         = 0 
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
	
		#Parameters that define how labels should be computed
		lbDat = pickle.load(open(self.param_.lbinfo_file))
		self.lbPrms_ = lbDat['lbInfo']
		self.lblSz_  = self.lbPrms_['lbSz']	
		
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
			rand   =  np.random.multinomial(1, self.grpSampleProb_)
			grpIdx =  np.where(rand==1)[0][0]
			ng     =  np.random.randint(low=0, high=self.grpCount_[grpIdx])
			self.argList.append([self.grpDat_[grpIdx][ng], self.fetchPrms_, self.lbPrms_])
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

def test_group_rots(isPlot=True):
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
				rots = tuple(pk[b].squeeze())[0:3]
				rots = [(r * 180.)/np.pi for r in rots]
				figTitle = 'yaw: %f,  pitch: %f, roll: %f' % (rots[0], rots[1], rots[2])
				ax = vu.plot_pairs(im[b,0:3], im[b,3:6], isBlobFormat=True,
             chSwap=(2,1,0), fig=fig, figTitle=figTitle)
				plt.draw()
				plt.show()
				ip = raw_input()
				if ip == 'q':
					return	


