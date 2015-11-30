import caffe
import glog

def image_reader_list(args):
	imNum = args[0]
	#Perform processing to get data, label
	return data, label

class DummyPreFetch(caffe.Layer):
	def setup(self, bottom, top):
		#Set the layer
		#Create a pool
		self.num_threads_ = 2
		self.pool_ = Pool(processes=self.num_threads_)
		#Set the name of function for reading the data
		self.readfn_ = image_reader_list
		self.launch_prefetch_jobs()

	def launch_prefetch_jobs(self):
		#construct input arguments to the reader
		#For eg: indexes into a list of images - imNum
		jobArgs    = [imNum]
		try:
			self.jobs_ = self.pool_.map_async(self.readfn_, jobArgs)
		except KeyboardInterrupt
			glog.info('Keyboard Interrupt received - terminating in launch jobs')
			self.pool_.terminate()	

	def get_prefetch_data(self):
		try:
			data, label = self.jobs_.get()
		except:
			glog.info('Keyboard Interrupt received - terminating')
			self.pool_.terminate()	
		return data, label

	def forward(self, bottom, top):
		data, label = get_prefetch_data()
		#Set the top with data and labels
		launch_prefetch_jobs()
