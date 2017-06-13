import tensorflow as tf
import numpy as np

from Utils import FLAGS, conv_repeat
from math import sqrt


class Encoder(object):

	def __init__(self, name, info):
		self.n = FLAGS.hidden_n # Hidden number of first dense ouput layer
		self.reshape_size_w = int(FLAGS.scale_w/8)
		self.reshape_size_h = int(FLAGS.scale_h/8)
		self.name = name
		self.info = info


		self.conv_repeat_layer_0 = conv_repeat('conv_repeat_layer_0') # Make conv_layer instance
		self.conv_repeat_layer_1 = conv_repeat('conv_repeat_layer_1') # Make conv_layer instance
		self.conv_repeat_layer_2 = conv_repeat('conv_repeat_layer_2') # Make conv_layer instance
		self.conv_repeat_layer_3 = conv_repeat('conv_repeat_layer_3') # Make conv_layer instance

	def encode(self, image, reuse=False):
		self.reuse = reuse
		self.image = image # Get image       
		 
		with tf.variable_scope(self.name, reuse=self.reuse):

			self.conv_repeat_info_0 = self.info['conv_repeat_0']
			self.conv_repeat_info_1 = self.info['conv_repeat_1']
			self.conv_repeat_info_2 = self.info['conv_repeat_2']
			self.conv_repeat_info_3 = self.info['conv_repeat_3']

			self.subsample_layer_info = self.info['subsample_layer']
			self.dense_layer_info = self.info['dense_layer']

			self.subsample_outdim = self.subsample_layer_info['outdim']
			self.subsample_kernel = self.subsample_layer_info['kernel']
			self.subsample_stride = self.subsample_layer_info['stride']
			self.subsample_activation = self.subsample_layer_info['activation']

			self.dense_outdim = self.dense_layer_info['outdim']

			

#           ________________________________________Layers_________________________________________________

			repeat_0 = self.conv_repeat_layer_0(image, conv_info = self.conv_repeat_info_0, reuse = self.reuse)  # Repeat CNN layers

			subsample_layer_0 = tf.layers.conv2d(repeat_0, self.subsample_outdim[0], self.subsample_kernel[0], self.subsample_stride[0], padding = "same", activation = self.subsample_activation[0], reuse = self.reuse, name = "subsample_layer_0")  

			repeat_1 = self.conv_repeat_layer_1(subsample_layer_0, conv_info = self.conv_repeat_info_1, reuse = self.reuse) # Repeat CNN layers

			subsample_layer_1 = tf.layers.conv2d(repeat_1, self.subsample_outdim[1], self.subsample_kernel[1], self.subsample_stride[1], padding = "same", activation = self.subsample_activation[1], reuse = self.reuse, name = "subsample_layer_1")  

			repeat_2 = self.conv_repeat_layer_2(subsample_layer_1, conv_info = self.conv_repeat_info_2, reuse = self.reuse) # Repeat CNN layers

			subsample_layer_2 = tf.layers.conv2d(repeat_2, self.subsample_outdim[1], self.subsample_kernel[1], self.subsample_stride[1], padding = "same", activation = self.subsample_activation[1], reuse = self.reuse, name = "subsample_layer_2") 
			
			repeat_3 = self.conv_repeat_layer_3(subsample_layer_2, conv_info = self.conv_repeat_info_3, reuse = self.reuse) # Repeat CNN layers

			reshaped_vector = tf.reshape(repeat_3, [FLAGS.bn, self.reshape_size_w*self.reshape_size_w*4*self.n])

			embedding_vector = tf.layers.dense(reshaped_vector, self.dense_outdim[0], reuse = self.reuse)

#			------------------------------------------------------------------------------------------------
			print("Encoded Embedding vector size : {}".format(embedding_vector))
			return embedding_vector



class Decoder(object):
    
	def __init__(self, name, info):
		self.n = FLAGS.hidden_n # Hidden number of first dense ouput layer
		self.reshape_size_w = int(FLAGS.scale_w/8)
		self.reshape_size_h = int(FLAGS.scale_h/8)
		self.name = name
		self.info = info

		self.conv_repeat_layer_0 = conv_repeat('conv_repeat_layer_0') # Make conv_layer instance
		self.conv_repeat_layer_1 = conv_repeat('conv_repeat_layer_1') # Make conv_layer instance
		self.conv_repeat_layer_2 = conv_repeat('conv_repeat_layer_2') # Make conv_layer instance
		self.conv_repeat_layer_3 = conv_repeat('conv_repeat_layer_3') # Make conv_layer instance
		self.conv_repeat_layer_4 = conv_repeat('conv_repeat_layer_4') # Make conv_layer instance


	def decode(self, h, reuse=False):
		self.reuse = reuse
		self.h = h # Get embedding vector "h"       
		 
		with tf.variable_scope(self.name, reuse=self.reuse):

			self.conv_repeat_info_0 = self.info['conv_repeat_0']
			self.conv_repeat_info_1 = self.info['conv_repeat_1']
			self.conv_repeat_info_2 = self.info['conv_repeat_2']
			self.conv_repeat_info_3 = self.info['conv_repeat_3']
			self.conv_repeat_info_4 = self.info['conv_repeat_4']

			self.dense_layer_info = self.info['dense_layer']

			self.dense_outdim = self.dense_layer_info['outdim']

			

#           ________________________________________Layers_________________________________________________
			dense_layer=tf.layers.dense(self.h, self.dense_outdim[0], reuse=self.reuse)
			h_0 = tf.reshape(dense_layer, [FLAGS.bn, self.reshape_size_w, self.reshape_size_h, self.n])
			
			repeat_0 = self.conv_repeat_layer_0(h_0, conv_info = self.conv_repeat_info_0, reuse = self.reuse)  # Repeat CNN layers

			h_1 = tf.image.resize_nearest_neighbor(h_0, [self.reshape_size_w*2, self.reshape_size_h*2]) # resize embedding vector
			resized_conv_layer_0 = tf.image.resize_nearest_neighbor(repeat_0, [self.reshape_size_w*2, self.reshape_size_w*2])  # resize previous output of conv_layer
			upsample_layer_1 = tf.concat([h_1, resized_conv_layer_0], axis=3) # concatenation

			repeat_1 = self.conv_repeat_layer_1(upsample_layer_1, conv_info = self.conv_repeat_info_1, reuse = self.reuse) # Repeat CNN layers

			h_2 = tf.image.resize_nearest_neighbor(h_0, [self.reshape_size_w*4, self.reshape_size_w*4]) # resize embedding vector
			resized_conv_layer_1 = tf.image.resize_nearest_neighbor(repeat_1, [self.reshape_size_w*4, self.reshape_size_w*4])  # resize previous output of conv_layer
			upsample_layer_2 = tf.concat([h_2, resized_conv_layer_1], axis=3) # concatenation

			repeat_2 = self.conv_repeat_layer_2(upsample_layer_2, conv_info = self.conv_repeat_info_2, reuse = self.reuse) # Repeat CNN layers

			h_3 = tf.image.resize_nearest_neighbor(h_0, [self.reshape_size_w*8, self.reshape_size_w*8]) # resize embedding vector
			resized_conv_layer_2 = tf.image.resize_nearest_neighbor(repeat_2, [self.reshape_size_w*8, self.reshape_size_w*8])  # resize previous output of conv_layer
			upsample_layer_3 = tf.concat([h_3, resized_conv_layer_2], axis=3) # concatenation

			repeat_3 = self.conv_repeat_layer_3(upsample_layer_3, conv_info = self.conv_repeat_info_3, reuse = self.reuse) # Repeat CNN layers


			output_img = self.conv_repeat_layer_4(repeat_3, conv_info = self.conv_repeat_info_4, reuse = self.reuse) # Repeat CNN layers

#			------------------------------------------------------------------------------------------------
			print("Decoded Output Image size : {}".format(output_img))
			return output_img
			









    