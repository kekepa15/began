import tensorflow as tf
import numpy as np
import librosa
import scipy

from PIL import Image
from glob import glob
from os import walk, mkdir



#Get image information
image = Image.open("./sample_image.jpg")
image_w, image_h = image.size
image_c = 3


#Get spectrogram information
offset = 0
duration = 3
sampling_rate = 16000
fft_size = 1024

y,sr = librosa.load("./sample_audio.wav", offset=offset, duration=duration, sr=sampling_rate) # load audio
D = librosa.stft(y, n_fft=fft_size, hop_length=int(fft_size/2), win_length=fft_size, window='hann') # make spectrogram
spectrogram_h = D.shape[0] # height of spectrogram
spectrogram_w = D.shape[1] # width of spectrogram
spectrogram_c = 1 # channel of spectrogram


#Define constant
flags = tf.app.flags
FLAGS = flags.FLAGS

#training parameters
flags.DEFINE_float('lr_D', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_G', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_reverse', 0.005, 'Initial learning rate.')
flags.DEFINE_float('B1', 0.5, 'Beta1')
flags.DEFINE_float('B2', 0.99, 'Beta2')
flags.DEFINE_integer('z_threshold', 0.13, 'A criteria to save z')
flags.DEFINE_integer('epochs', 1000000, 'Maximum epochs to iterate.')
flags.DEFINE_integer('bn', 32, "Batch number")
flags.DEFINE_integer('bn_reverse', 1, "Batch number for reverse training")
flags.DEFINE_integer('reg_lambda', 0.05, "Regularizer parameter")

#data parameters
flags.DEFINE_integer('spec_h', spectrogram_h, "Height of spectrogram" )
flags.DEFINE_integer('spec_w', spectrogram_w, "Width of spectrogram" )
flags.DEFINE_integer('spec_c', spectrogram_c, "Channel of spectrogram" )
flags.DEFINE_integer('img_h', image_h, "Height of image" )
flags.DEFINE_integer('img_w', image_w, "Width of image" )
flags.DEFINE_integer('img_c', image_c, "Channel of image" )
flags.DEFINE_integer('scale_w', 64, "Width Scaling Factor" )
flags.DEFINE_integer('scale_h', 64, "Height Scaling Factor" )

#model parameters
flags.DEFINE_integer('z_size', 64, "Embedding vector size")
flags.DEFINE_integer('hidden_n', 64, "Hidden convolution depth")
flags.DEFINE_integer('output_channel', 3, "Output channel number")
flags.DEFINE_float("gamma", 0.5, "Gamma : Diversity ratio")
flags.DEFINE_float("lamb", 0.001, "Lambda : Learning rate of k_t")
flags.DEFINE_float("iteration", 10000000, "Maximum iteration number")

#gpu parameters
flags.DEFINE_float("gpu_portion", 0.4, "Limit the GPU portion")

#---------------------------------------------------------------------------#

#Functions
def generate_z_reverse(size=FLAGS.z_size):
    return tf.random_uniform(shape=(FLAGS.bn_reverse,size), minval=-1, maxval=1, dtype=tf.float32)


def generate_z(size=FLAGS.z_size):
	return tf.random_uniform(shape=(FLAGS.bn,size), minval=-1, maxval=1, dtype=tf.float32)

def get_loss(image, decoded_image):
	L1_norm = tf.reduce_mean(tf.abs(tf.subtract(image,decoded_image)))
	return L1_norm

def norm_img(image):
	image = image/127.5 - 1.
	return image

def denorm_img(norm):
	return tf.clip_by_value((norm + 1.)*127.5, 0, 255)

def denorm_img_np(norm):
    return np.clip((norm + 1.)*127.5, 0, 255)


def upsample(images, size):
	"""
	images : image having shape with [batch, height, width, channels],
	size : output_size with [new_height, new_width]
	"""
	return tf.image.resize_nearest_neighbor(images=images, size=size)


class conv_repeat(object):
	def __init__(self, name):
		"""
		Arguments
		name : tensorflow layer name
		"""
		self.name = name

	def __call__(self, prev, conv_info, reuse = False):
		"""
		Arguments
		conv_info : convolution layer information dictionary
		repeat : How many layers to repeat
		"""
		self.info = conv_info
        
		self.outdim_info = self.info['outdim']
		self.kernel_info = self.info['kernel']
		self.stride_info = self.info['stride']
		self.activation_info = self.info['activation']
		
		self.reuse = reuse
		self.repeat = len(self.outdim_info) # length of outdim_info list is same as the number of how many times conv layer will be defined
    
		with tf.variable_scope(self.name, reuse=self.reuse):
			for i in range(self.repeat):
				prev = tf.layers.conv2d(prev, self.outdim_info[i], self.kernel_info[i],	self.stride_info[i], padding = "same", activation = self.activation_info[i],reuse = self.reuse)
				print("Define layer: {}".format(prev.name), "Reuse : {}".format(self.reuse))
                
		return prev
