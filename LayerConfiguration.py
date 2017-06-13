import tensorflow as tf
import numpy as np

from Utils import FLAGS


n = FLAGS.hidden_n
z_size = FLAGS.z_size

#-------------------------------------------Encoder informations------------------------------------------------

Encoder_conv_repeat_infos_0 = {
                                "outdim":[n, n, n, n, 2*n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }


Encoder_conv_repeat_infos_1 = {
                                "outdim":[2*n, 2*n, 2*n, 3*n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Encoder_conv_repeat_infos_2 = {
                                "outdim":[3*n, 3*n, 3*n, 4*n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Encoder_conv_repeat_infos_3 = {
                                "outdim":[4*n, 4*n, 4*n, 4*n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }                              

Encoder_subsample_layer_infos = {
                                 "outdim":[2*n, 3*n, 4*n], \
                                 "kernel":[ [3, 3], [3, 3], [3, 3] ], \
                                 "stride":[ [2, 2], [2, 2], [2, 2] ], \
                                 "activation":[ tf.nn.elu, tf.nn.elu, tf.nn.elu ], \
                                }

Encoder_dense_layer_infos = {
                                 "outdim":[FLAGS.z_size], \
                            }


#------------------------------------------------Encoder dictionary------------------------------------------------


Encoder_infos = {
                    "conv_repeat_0":Encoder_conv_repeat_infos_0,\
                    "conv_repeat_1":Encoder_conv_repeat_infos_1,\
                    "conv_repeat_2":Encoder_conv_repeat_infos_2,\
                    "conv_repeat_3":Encoder_conv_repeat_infos_3,\
                    "subsample_layer":Encoder_subsample_layer_infos,\
                    "dense_layer":Encoder_dense_layer_infos, \
                } 

#______________________________________________________________________________________________________________







#-------------------------------------------Decoder informations------------------------------------------------

Decoder_dense_layer_infos = {
                                 "outdim":[int(FLAGS.scale_w/8)*int(FLAGS.scale_w/8)*n], \
                            }

Decoder_conv_repeat_infos_0 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }


Decoder_conv_repeat_infos_1 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Decoder_conv_repeat_infos_2 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }


Decoder_conv_repeat_infos_3 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }                              

Decoder_conv_repeat_infos_4 = {
                                "outdim":[3], \
                                "kernel":[ [3, 3] ], \
                                "stride":[ [1, 1] ], \
                                "activation" : [tf.nn.elu]
                              }



#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------Decoder dictionary------------------------------------------------


Decoder_infos = {
                    "dense_layer":Decoder_dense_layer_infos, \
                    "conv_repeat_0":Decoder_conv_repeat_infos_0,\
                    "conv_repeat_1":Decoder_conv_repeat_infos_1,\
                    "conv_repeat_2":Decoder_conv_repeat_infos_2,\
                    "conv_repeat_3":Decoder_conv_repeat_infos_3,\
                    "conv_repeat_4":Decoder_conv_repeat_infos_4,\
                } 

#------------------------------------------------------------------------------------------------------------







#_______________________________________Generator informations_________________________________________________

Generator_dense_layer_infos = {
                                 "outdim":[int(FLAGS.scale_w/8)*int(FLAGS.scale_w/8)*n], \
                            }

Generator_conv_repeat_infos_0 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }


Generator_conv_repeat_infos_1 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Generator_conv_repeat_infos_2 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Generator_conv_repeat_infos_3 = {
                                "outdim":[n, n, n, n], \
                                "kernel":[ [3, 3], [3, 3], [3, 3], [3, 3] ], \
                                "stride":[ [1, 1], [1, 1], [1, 1], [1, 1] ], \
                                "activation" : [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu]
                              }

Generator_conv_repeat_infos_4 = {
                                "outdim":[3], \
                                "kernel":[ [3, 3] ], \
                                "stride":[ [1, 1] ], \
                                "activation" : [ tf.nn.elu ]
                              }



#-------------------------------------------------------------------------------------------------------------

#_________________________________________Generator dictionary_________________________________________________


Generator_infos = {
                    "dense_layer":Generator_dense_layer_infos, \
                    "conv_repeat_0":Generator_conv_repeat_infos_0,\
                    "conv_repeat_1":Generator_conv_repeat_infos_1,\
                    "conv_repeat_2":Generator_conv_repeat_infos_2,\
                    "conv_repeat_3":Generator_conv_repeat_infos_3,\
                    "conv_repeat_4":Generator_conv_repeat_infos_4,\
                } 

#------------------------------------------------------------------------------------------------------------