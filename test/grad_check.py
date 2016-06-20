import matplotlib
matplotlib.use('agg')
import caffe
import scipy
import numpy as np

def forward_quat(pd_quat, *args):
	gt_quat, net = args
	loss = net.forward_all(['loss'], pd_quat=pd_quat.reshape((1,4,1,1)).astype(np.float32), gt_quat=gt_quat.reshape((1,5,1,1)).astype(np.float32))
	return loss['loss'].squeeze()

def grad_quat(pd_quat, *args):
	gt_quat, net = args
	loss, grad = net.forward_backward_all(['loss'], pd_quat=pd_quat.reshape((1,4,1,1)).astype(np.float32), 
				gt_quat=gt_quat.reshape((1,5,1,1)).astype(np.float32))
	print loss['loss'].squeeze(), pd_quat, gt_quat, grad['pd_quat'].squeeze()
	return grad['pd_quat'].reshape((4,))

def check_quat_loss():
	net = caffe.Net('quat_l2loss.prototxt', caffe.TEST)
	gt_quat = np.array([1,0,0,0,1]).reshape((5,))
	pd_quat = np.random.randn(4,).reshape((4,))
	pd_quat = pd_quat / np.sqrt(np.sum(pd_quat * pd_quat))
	eps = 1e-4
	err = scipy.optimize.check_grad(forward_quat, grad_quat, pd_quat, gt_quat, net, epsilon=eps)
	print scipy.optimize.approx_fprime(pd_quat, forward_quat, eps, gt_quat, net)
	print ('Error', err)
