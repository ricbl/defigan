# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Modified by Ricardo Bigolin Lanfredi 
# Last modified: 2020-07-01

import os.path
import tensorflow as tf

from . import system as sys_config
from . import tf_utils

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

class vagan:

    """
    This class contains all the methods for defining training and evaluating the VA-GAN method
    """

    def __init__(self, exp_config, fixed_batch_size=None):

        """
        Initialise the VA-GAN model with the two required networks, loss functions, etc... 
        :param exp_config: An experiment config file
        :param data: A handle to the data object that should be used
        :param fixed_batch_size: Optionally, a fixed batch size can be selected. If None, the batch_size will stay
                                 flexible. 
        """

        self.exp_config = exp_config

        self.mode3D = True if len(exp_config.image_size) == 3 else False

        self.critic_net = exp_config.critic_net
        self.generator_net = exp_config.generator_net
        
        self.img_tensor_shape = [fixed_batch_size] + list(exp_config.image_size) + [1]
        self.batch_size = exp_config.batch_size

        # Generate placeholders for the images and labels.
        self.training_pl_cri = tf.placeholder(tf.bool, name='training_phase_critic')
        self.training_pl_gen = tf.placeholder(tf.bool, name='training_phase_generator')

        self.lr_pl = tf.placeholder(tf.float32, name='learning_rate')

        self.x_c0 = tf.placeholder(tf.float32, self.img_tensor_shape, name='c0_img')
        self.x_c1 = tf.placeholder(tf.float32, self.img_tensor_shape, name='c1_img')

        # network outputs
        self.M = self.generator_net(self.x_c1, self.training_pl_gen)
        self.y_c0_ = self.x_c1 + self.M

        if exp_config.use_tanh:
            self.y_c0_ = tf.tanh(self.y_c0_)

        self.D = self.critic_net(self.x_c0, self.training_pl_cri, scope_reuse=False)
        self.D_ = self.critic_net(self.y_c0_, self.training_pl_cri, scope_reuse=True)

        # Make optimizers
        train_vars = tf.trainable_variables()
        self.gen_vars = [v for v in train_vars if v.name.startswith("generator")]
        self.cri_vars = [v for v in train_vars if v.name.startswith("critic")]
        
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.saver_best = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)

    def predict_mask(self, input_image):

        """
        Get the estimated mask for an input_image
        """

        pred_mask = self.sess.run(self.M, feed_dict={self.x_c1: input_image, self.training_pl_gen: False})
        return pred_mask


    def load_weights(self, log_dir=None, type='latest', **kwargs):

        """
        Load weights into the model
        :param log_dir: experiment directory into which all the checkpoints have been written
        :param type: can be 'latest', 'best_ncc' (highest validation Wasserstein distance), or 'iter' (specific
                     iteration, requires passing the iteration argument with a valid step number from the checkpoint 
                     files)
        """

        if not log_dir:
            log_dir = self.log_dir

        if type=='latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type=='best_ncc':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_ncc.ckpt')
        elif type=='iter':
            assert 'iteration' in kwargs, "argument 'iteration' must be provided for type='iter'"
            iteration = kwargs['iteration']
            init_checkpoint_path = os.path.join(log_dir, 'model.ckpt-%d' % iteration)
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_ncc/iter.' % type)

        self.saver.restore(self.sess, init_checkpoint_path)

    def _get_optimizer(self, lr_pl):

        """
        Helper function for getting the right optimizer
        """

        if self.exp_config.optimizer_handle == tf.train.AdamOptimizer:
            return self.exp_config.optimizer_handle(lr_pl, beta1=self.exp_config.beta1, beta2=self.exp_config.beta2)
        else:
            return self.exp_config.optimizer_handle(lr_pl)