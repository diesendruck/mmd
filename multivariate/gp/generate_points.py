import pdb
import numpy as np
import tensorflow as tf
from multivariate_mmd_gan import build_model, load_checkpoint 
from weighting import get_estimation_points 


# Within scope of code, need to set up data and TF session.
def set_up_tf_model_once():
    data_file = '/home/maurice/mmd/multivariate/gp/gp_data.txt'
    checkpoint_dir = '/home/maurice/mmd/multivariate/gp/logs_test/checkpoints'
    batch_size = 200
    gen_num = 200
    z_dim = 3
    # Load data.
    data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
    data_num = data.shape[0]
    out_dim = data.shape[1]
    # Build model.
    x, z, _, g_read_only, _, _, _, _, _ = build_model(
        batch_size, gen_num, out_dim, z_dim)
    # Initialize TF session.
    init_op = tf.global_variables_initializer() 
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)
    sess.run(init_op)
    # Load existing checkpoint.
    could_load, checkpoint_counter = load_checkpoint(
        saver, sess, checkpoint_dir)
    if could_load: 
        load_step = checkpoint_counter
        print(' [*] Load SUCCESS, checkpoint {}'.format(load_step))
    else:
        print(' [!] Load failed...') 
    # Package output for reading graph in generate_points().
    tf_setup = [data, batch_size, gen_num, z_dim, sess, x, z, g_read_only]
    return tf_setup 


# Store values from one-time setup.
tf_setup = set_up_tf_model_once()


# After one-time setup, this function calls samples.
def generate_points(tf_setup=tf_setup, mode='coreset'):
    data = tf_setup[0] 
    batch_size = tf_setup[1] 
    gen_num = tf_setup[2] 
    z_dim = tf_setup[3] 
    sess = tf_setup[4] 
    x = tf_setup[5] 
    z = tf_setup[6] 
    g_read_only = tf_setup[7] 

    random_batch_data = np.array(
            [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = np.random.uniform(size=[gen_num, z_dim],low=-1.0, high=1.0) 
    #random_batch_z = np.random.normal(size=[gen_num, z_dim]) 
    g_out = sess.run(g_read_only, 
            feed_dict={ 
                x: random_batch_data,
                z: random_batch_z})

    support_points, coreset, weights_estimation_pts, weights_data = \
        get_estimation_points(mode=mode, support_points=g_out, email=False)
    return support_points, coreset, weights_estimation_pts, weights_data

test = 1
if test:
    results = generate_points() 
    support_points, coreset, weights_estimation_pts, weights_data = results
    for r in results:
        print r.shape

    results = generate_points() 
    support_points, coreset, weights_estimation_pts, weights_data = results
    for r in results:
        print r.shape



