"""
Python2 & Python3 
Version Compatible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Utils import generate_z_reverse, generate_z, FLAGS, get_loss, norm_img, denorm_img, denorm_img_np
from Model_Reverse import Decoder, Encoder
from Loader import Image_Loader, save_image
from LayerConfiguration import Encoder_infos, Decoder_infos, Generator_infos
from PIL import Image

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import math

def main(_):

	"""
	Run main function
	"""

	#___________________________________________Layer info_____________________________________________________

	"""
	Prepare Image Loader
	"""
	# root = "/raid/kekepa15/Audio_montage/BEGAN/CelebA/images"
	root = "/raid/kekepa15/Audio_BEGAN/Real_Images"
	batch_size = FLAGS.bn_reverse
	scale_size = [FLAGS.scale_h,FLAGS.scale_w]
	data_format = "NHWC"
	loader = Image_Loader(root, batch_size, scale_size, data_format, file_type="jpg")



	"""
	Make Saving Directories
	"""
	os.makedirs("./Check_Point", exist_ok=True)
	os.makedirs("./Generator_Check_Point", exist_ok=True)
	os.makedirs("./logs", exist_ok=True) # make logs directories to save summaries
	os.makedirs("./Real_Images", exist_ok=True)
	os.makedirs("./Generated_Images", exist_ok=True)
	os.makedirs("./Generated_Images_by_Reversed_z", exist_ok=True)
	os.makedirs("./Reversed_Images", exist_ok=True)
	os.makedirs("./Decoded_Generated_Images", exist_ok=True)




	#----------------------------------------------------------------------------------------------------



	#____________________________________Model composition________________________________________
	lr_reverse = tf.Variable(FLAGS.lr_reverse, name='lr_reverse')
	lr_reverse_decrease = tf.assign(lr_reverse, lr_reverse*0.95, name='lr_reverse_decrease')
	lr_reverse_increase = tf.assign(lr_reverse, lr_reverse*3, name='lr_reverse_increase')
	lr_reverse_recovery = tf.assign(lr_reverse, FLAGS.lr_reverse, name='lr_reverse_recovery')

	k = tf.Variable(0.0, name = "k_t", trainable = False, dtype = tf.float32) #init value of k_t = 0
	z_G = tf.Variable(tf.random_uniform(shape=(FLAGS.bn_reverse,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable = True, name = "z_G") # Sample embedding vector batch from uniform distribution
	z_D = tf.Variable(tf.random_uniform(shape=(FLAGS.bn_reverse,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable = True, name = "z_D") # Sample embedding vector batch from uniform distribution	z_updated = tf.assign(z_G, generate_z_reverse())
	# z_scaling = tf.assign(z_G, (64/3)*(z_G/tf.norm(z_G,ord='euclidean')) )
	z_scaling = tf.multiply((tf.sqrt(64.0/3.0)),(z_G/tf.norm(z_G,ord='euclidean')))
	z_updated = tf.assign(z_G, tf.Variable(tf.random_uniform(shape=(FLAGS.bn_reverse,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable = True, name = "z_updated"))


	indice = tf.placeholder(tf.int32, shape=None)
	row = tf.gather(z_G, 0)
	new_row = tf.concat([row[:indice], tf.random_uniform(shape=[1], minval=-1, maxval=1, dtype=tf.float32), row[indice+1:]], axis=0)
	replace_z_element = tf.scatter_update(z_G, tf.constant(0), new_row)

	# batch = loader.queue # Get image batch tensor
	# image = norm_img(loader.get_image_from_loader)
	image_holder = tf.placeholder(tf.float32, shape=(FLAGS.bn_reverse, FLAGS.scale_h, FLAGS.scale_w, FLAGS.output_channel)) 

	E = Encoder("Encoder", Encoder_infos)
	D = Decoder("Decoder", Decoder_infos)
	G = Decoder("Generator", Generator_infos)

	#Generator
	generated_image = G.decode(z_G)
	generated_image_for_disc = G.decode(z_D, reuse = True)


	#Discriminator (Auto-Encoder)	

	#image <--AutoEncoder--> reconstructed_image_real
	embedding_vector_real = E.encode(image_holder)
	reconstructed_image_real = D.decode(embedding_vector_real)

	#generated_image_for_disc <--AutoEncoder--> reconstructed_image_fake
	embedding_vector_fake_for_disc = E.encode(generated_image_for_disc, reuse=True)
	reconstructed_image_fake_for_disc = D.decode(embedding_vector_fake_for_disc, reuse=True)

	#generated_image <--AutoEncoder--> reconstructed_image_fake
	embedding_vector_fake = E.encode(generated_image, reuse=True)
	reconstructed_image_fake = D.decode(embedding_vector_fake, reuse=True)


	#-----------------------------------------------------------------------------------------------



	#_________________________________Loss & Summary_______________________________________________


	"""
	Define Loss
	"""
	real_image_loss = get_loss(image_holder, reconstructed_image_real)
	generator_loss_for_disc = get_loss(generated_image_for_disc, reconstructed_image_fake_for_disc)
	discriminator_loss = real_image_loss - tf.multiply(k, generator_loss_for_disc)

	generator_loss = get_loss(generated_image, reconstructed_image_fake)
	global_measure = real_image_loss + tf.abs(tf.multiply(FLAGS.gamma,real_image_loss) - generator_loss)


	norm_regularizer =  tf.sqrt(tf.reduce_sum(tf.square(z_G-z_scaling)))
	z_loss = get_loss(image_holder, generated_image)
	z_loss_norm = FLAGS.reg_lambda*norm_regularizer
	# z_loss = get_loss(image_holder, generated_image)

	"""
	Summaries
	"""
	tf.summary.scalar('z loss', z_loss)
	z_list = []

	merged_summary = tf.summary.merge_all() # merege summaries, no more summaries under this line

	#-----------------------------------------------------------------------------------------------



	#_____________________________________________Train_______________________________________________

	optimizer_z = tf.train.AdamOptimizer(lr_reverse,beta1=0.5,beta2=0.9).minimize(z_loss, var_list=z_G)
	# optimizer_z_norm = tf.train.AdamOptimizer(lr_reverse,beta1=0.5,beta2=0.9).minimize(z_loss_norm, var_list=z_G)
	# optimizer_z = tf.train.GradientDescentOptimizer(0.1).minimize(z_loss, var_list=z_G)

	init = tf.global_variables_initializer()	


	NUM_THREADS=2
	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
						intra_op_parallelism_threads=NUM_THREADS,\
						allow_soft_placement=True,\
						device_count = {'CPU': 1},\
						)

	# config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_portion



	with tf.Session(config=config) as sess:

		sess.run(init) # Initialize Variables

		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
		writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		

#_______________________________Restore____________________________________

		print(" Reading checkpoints...")
		discriminator_parameters = []
		generator_parameters = []

		for v in tf.trainable_variables():
			if 'Encoder' in v.name:
				discriminator_parameters.append(v)
				print("Discriminator parameter : ", v.name)
			elif 'Decoder' in v.name:
				discriminator_parameters.append(v)
				print("Discriminator parameter : ", v.name)			
			elif 'Generator' in v.name:
				generator_parameters.append(v)
				print("Generator parameter : ", v.name)
			else:
				print("None of Generator and Discriminator parameter : ", v.name)

		print("Generator variables : {}".format(generator_parameters))
				
		restorer = tf.train.Saver(generator_parameters)
		checkpoint_path = tf.train.latest_checkpoint("./Check_Point")
		restorer.restore(sess, checkpoint_path)

#---------------------------------------------------------------------------
		j=0
		iteration = 0
		switch = 0
		image = norm_img(loader.get_image_from_loader(sess))
		Real_Images = denorm_img_np(image)
		# save_image(Real_Images, '{}.png'.format("./Real_Images/Real_Image"))

		print(sess.run(tf.norm(z_G, ord='euclidean')))
		print(sess.run(tf.norm(z_scaling, ord='euclidean')))

		for t in range(FLAGS.iteration): # Mini-Batch Iteration Loop
	
			iteration += 1

			if coord.should_stop():
				break
			
			start_time = time.time() # tic

			# _, _, current_z, loss, norm_loss = sess.run([optimizer_z, optimizer_z_norm, z_G, z_loss, z_loss_norm], feed_dict={image_holder : image})
			_, current_z, loss, norm_loss = sess.run([optimizer_z, z_G, z_loss, z_loss_norm], feed_dict={image_holder : image})
			for i in range(64):
				if current_z[0,i]>1:
					sess.run(replace_z_element, feed_dict={indice : i})
				elif current_z[0,i]<-1:
					sess.run(replace_z_element, feed_dict={indice : i})


			print(
				 " Iteration : {}".format(iteration),
				 " Loss(Real image - Generated image) : {}".format(loss),
				 " Loss(Scaled z - current_z) : {}".format(norm_loss),
				 )

			if iteration % 200 == 0:
				elapsed = time.time() - start_time #toc
				print("Elapsed time : {}".format(elapsed))
				# Generated_Images = sess.run(denorm_img(generated_image), feed_dict={z_G : current_z})				
				# # save_image(Generated_Images, '{}/{}{}{}{}.png'.format("./Generated_Images_by_Reversed_z",j,"th", "Generated", iteration))
				# # print("---------------------------------------Image saved----------------------------------------")

			# if iteration % 2000 == 0:
			# 	sess.run(z_scaling)
			# 	print("Scaling")


			# if iteration >= 5000:
			# 	if loss >= 0.15:
			# 		sess.run(z_updated)
			# 		sess.run(lr_reverse_recovery) # learning rate recovery
			# 		iteration = 1

			if (loss < FLAGS.z_threshold) or (iteration >= 10000) :
				
				j += 1				
				iteration = 1
				switch = 0
				# z_loss = get_loss(image_holder, generated_image) + FLAGS.reg_lambda*norm_regularizer # loss with regularizer
				# print("Loss with regularizer")
			
				z_list.append(current_z)
				Real_Images = denorm_img_np(image)
				Generated_Images = sess.run(denorm_img(generated_image), feed_dict={z_G : current_z})

				Concat_Images = np.vstack((Real_Images,Generated_Images))

				save_image(Concat_Images, '{}/{}_{}.png'.format("./Reversed_Images", "Real&Reversed_Images",j))

				# save_image(Real_Images, '{}/{}_{}.png'.format("./Reversed_Images", "Real_Image",j))
				# save_image(Generated_Images, '{}/{}_{}.png'.format("./Reversed_Images", "Generated",j))

			

				image = norm_img(loader.get_image_from_loader(sess)) # get new image!

				Adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
				print("Adam_initializers : {}".format(Adam_initializers))
				# GradientDescent_initializers = [var.initializer for var in tf.global_variables() if 'GradientDescent' in var.name]
				# print("GradientDescent_initializers : {}".format(Adam_initializers))				


				sess.run([Adam_initializers, z_updated, lr_reverse_recovery])  # init Adam 1st,2nd momentum; init z_G; learning rate recovery


	       #________________________________Save____________________________________

			if iteration % 1000 == 0:

				# if switch == 0:
					# print("Loss without regularizer")
					# z_loss = get_loss(image_holder, generated_image) # loss without regularizer
					# switch = 1
				
				sess.run(lr_reverse_decrease)

	       #--------------------------------------------------------------------
		
		writer.close()
		coord.request_stop()
		coord.join(threads)


#-----------------------------------Train Finish---------------------------------



if __name__ == "__main__" :
	tf.app.run()










# """
# Python2 & Python3 
# Version Compatible
# """
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from Utils import generate_z, FLAGS, get_loss, norm_img, denorm_img
# from Model import Decoder, Encoder
# from Loader import Image_Loader, save_image
# from LayerConfiguration import Encoder_infos, Decoder_infos, Generator_infos
# from PIL import Image

# import tensorflow as tf
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# def main(_):

# 	"""
# 	Run main function
# 	"""


# 	"""
# 	Prepare Image Loader
# 	"""
# 	root = "/home/choi/Documents/BEGAN/CelebA/images"
# 	batch_size = FLAGS.bn
# 	scale_size = [FLAGS.scale_h,FLAGS.scale_w]
# 	data_format = "NHWC"
# 	loader = Image_Loader(root, batch_size, scale_size, data_format, file_type="jpg")



# 	"""
# 	Make Saving Directories
# 	"""
# 	os.makedirs("./Generator_Check_Point", exist_ok=True)
# 	os.makedirs("./logs", exist_ok=True) # make logs directories to save summaries

# 	#----------------------------------------------------------------------------------------------------



# 	#____________________________________Model composition________________________________________


# 	batch = loader.queue # Get image batch tensor
# 	image = norm_img(batch) # Normalize Imgae

# 	z_G = tf.Variable(tf.random_uniform(shape=(FLAGS.bn,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable=True, name = "z_G") # Sample embedding vector batch from uniform distribution
# 	G = Decoder("Generator", Generator_infos)

# 	#Generator
# 	generated_image = G.decode(z_G)


# 	#-----------------------------------------------------------------------------------------------



# 	#_________________________________Loss & Summary_______________________________________________


# 	"""
# 	Define Loss
# 	"""
# 	z_loss = get_loss(image, generated_image)


# 	"""
# 	Summaries
# 	"""
# 	tf.summary.scalar('z loss', z_loss)
# 	merged_summary = tf.summary.merge_all() # merege summaries, no more summaries under this line
# 	z_list = []
# 	#-----------------------------------------------------------------------------------------------



# 	#_____________________________________________Train_______________________________________________
# 	optimizer_z = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(z_loss,var_list=z_G)

# 	init = tf.global_variables_initializer()	


# 	NUM_THREADS=2
# 	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
# 						intra_op_parallelism_threads=NUM_THREADS,\
# 						allow_soft_placement=True,\
# 						device_count = {'CPU': 1},\
# 						)

# 	# config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_portion



# 	with tf.Session(config=config) as sess:

# 		sess.run(init) # Initialize Variables

# 		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
# 		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
# 		writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		



# 		# generator_saver = tf.train.Saver(max_to_keep=1000)
# 		generator_saver = tf.train.import_meta_graph("./Generator_Check_Point/model.ckpt-3500.meta")
# 		generator_saver.restore(sess,tf.train.latest_checkpoint("./Generator_Check_Point"))


# 		# generator_ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Generator_Check_Point")


# 		# try :	
# 		# 	if generator_ckpt and generator_ckpt.model_checkpoint_path:
# 		# 		print("check point path : ", generator_ckpt.model_checkpoint_path)
# 		# 		generator_saver.restore(sess, generator_ckpt.model_checkpoint_path)	
# 		# 		print('Restored!')
# 		# except AttributeError:
# 		# 		print("No checkpoint")	

# 		generator_parameter = []

# 		for v in tf.trainable_variables():		
# 			if 'Generator' in v.name:
# 				generator_parameters.append(v)
# 				print("Generator parameter : ", v.name)
# 			else:
# 				print("Others : ", v.name)


# 		for t in range(FLAGS.iteration): # Mini-Batch Iteration Loop

# 			if coord.should_stop():
# 				break
			
# 			_, current_z, loss = sess.run(optimizer_Z,z,loss)

		
# 			if loss < FLAGS.z_threshold:
# 				z_list.append(z)



# if __name__ == "__main__" :
# 	tf.app.run()


