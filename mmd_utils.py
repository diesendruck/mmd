import pdb
import numpy as np
import tensorflow as tf


def compute_mmd(arr1, arr2, sigma_list=None, use_tf=False):
    """Computes mmd between two numpy arrays of same size."""
    if sigma_list is None:
        sigma_list = [1.0]

    n1 = len(arr1)
    n2 = len(arr2)

    if use_tf:
        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        num_combos_x = tf.to_float(n1 * (n1 - 1) / 2)
        num_combos_y = tf.to_float(n2 * (n2 - 1) / 2)
        num_combos_xy = tf.to_float(n1 * n2)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / num_combos_xy)
        return mmd, exp_object
    else:
        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])
        v = np.concatenate((arr1, arr2), 0)
        VVT = np.matmul(v, np.transpose(v))
        sqs = np.reshape(np.diag(VVT), [-1, 1])
        sqs_tiled_horiz = np.tile(sqs, np.transpose(sqs).shape)
        exp_object = sqs_tiled_horiz - 2 * VVT + np.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += np.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        num_combos_x = n1 * (n1 - 1) / 2
        num_combos_y = n2 * (n2 - 1) / 2
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (n1 * n2))
        return mmd, exp_object


def compute_kmmd(arr1, arr2, sigma_list=None, use_tf=False):
    """Computes 2-mmd between two numpy arrays of same size.
    
    The projection of a distribution into the RKHS defined by the Gaussian
    kernel, is equivalent to the inner product of an infinite-dimension Hermite
    polynomial basis. Truncating this basis involves using only the first k
    Hermite polynomials in the inner product. For the Gaussian kernel, the
    first bases are: H_0 = 1, H_1 = x, H_2 = x^2 - 1, such that the kernel
    is as follows:
      k(x, y) = (1 * 1) + (x * y) + ((x^2 - 1) * (y^2 - 1))
              = 2 + xy + x^2 * y^2 - x^2 - y^2.
    The MMD^2 estimator then sums these kernel values over all unique pairs
    in the given arrays.
    """
    if sigma_list is None:
        sigma_list = [1.0]

    n1 = len(arr1)
    n2 = len(arr2)

    if use_tf:
        pass
        # TODO: Update this with kMMD.
        """
        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
        #exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        num_combos_x = tf.to_float(n1 * (n1 - 1) / 2)
        num_combos_y = tf.to_float(n2 * (n2 - 1) / 2)
        num_combos_xy = tf.to_float(n1 * n2)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / num_combos_xy)
        return mmd, exp_object
        """
    else:
        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])

        v = np.concatenate((arr1, arr2), 0)
        VVT = np.matmul(v, np.transpose(v))  # Outer product.
        v_sq = v ** 2 
        VVT_sq = np.matmul(v_sq, np.transpose(v_sq))  # Outer product.

        v_sq_tiled = np.tile(v_sq, [1, v_sq.shape[0]])

        # Construct polynomial kernel as inner product of up-to-k=2 Hermite 
        # polynomial bases.
        polynomial_object = 2. + VVT + VVT_sq - v_sq_tiled - \
            np.transpose(v_sq_tiled)  
        K = polynomial_object
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]

        # More explicit version.
        """
        x = arr1
        y = arr2
        K_xx = (2. + 
            x * np.transpose(x) +
            x**2 * np.transpose(x**2) -
            np.tile(x**2, [1, n1]) -
            np.transpose(np.tile(x**2, [1, n1])))
        K_yy = (2. + 
            y * np.transpose(y) +
            y**2 * np.transpose(y**2) -
            np.tile(y**2, [1, n2]) -
            np.transpose(np.tile(y**2, [1, n2])))
        K_xy = (2. +
            x * np.transpose(y) +
            x**2 * np.transpose(y**2) -
            np.tile(x**2, [1, n2]) -
            np.transpose(np.tile(y**2, [1, n1])))
        xx_test = (K_xx_ == K_xx).all()
        yy_test = (K_yy_ == K_yy).all()
        xy_test = (K_xy_ == K_xy).all()
        if not np.array([xx_test, yy_test, xy_test]).all():
            pdb.set_trace()
        """

        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        num_combos_x = n1 * (n1 - 1) / 2
        num_combos_y = n2 * (n2 - 1) / 2
        kmmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (n1 * n2))
        return kmmd, polynomial_object
