import caffe
import numpy as np
import os
import sys
import argparse
from multiprocessing import Process, Queue


class MultiNFrame(caffe.Layer):
	def setup(self, bottom, top):
		layerPrms = 
