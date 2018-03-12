import os
import pdb
import sys
import numpy as np
import tensorflow as tf
from multivariate_mmd_gan import build_model, load_checkpoint 
from weighting import get_estimation_points 


def load_model_and_get_points():
    # Set FIXED model parameters.
    model = 'test'
    ckpt = '48000'
    z_dim = 10  # Fixed, once model is trained.
    log_dir = '/home/maurice/mmd/simulation_studies/gp/logs_{}'.format(model)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'test-{}.meta'.format(ckpt))

    # Create an empty graph for the session.
    loaded_graph = tf.Graph()

    # Set up session. 
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))

    with tf.Session(graph=loaded_graph, config=sess_config) as sess:
        saver = tf.train.import_meta_graph(checkpoint_file)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # Fetch input and output tensors.
        g_out_tensor = loaded_graph.get_tensor_by_name(
            'generator_out_1/dense/BiasAdd:0')
        z_tensor = loaded_graph.get_tensor_by_name('z_read:0')

        ###############################
        # RUN THE MODEL AND GET POINTS.
        gen_num = 800
        g_out = sess.run(g_out_tensor, 
            feed_dict={ 
                z_tensor: np.random.normal(size=[gen_num, z_dim])})

        # Make estimation points.
        results = get_estimation_points(
            log_dir=log_dir, mode='coreset', support_points=g_out)
        support_points, coreset, weights_estimation_pts, weights_data = results
        ###############################


load_model_and_get_points()
