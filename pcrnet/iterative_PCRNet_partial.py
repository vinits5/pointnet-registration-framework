import argparse
import math
import h5py
import numpy as np
from numpy import matlib as npm
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import helper
from helper import print_
import transforms3d.euler as t3d

parser = argparse.ArgumentParser()
parser.add_argument('-mode','--mode', required=True, type=str, default='no_mode', help='mode: train or test')
parser.add_argument('-log','--log_dir', required=True, default='log_itrPCRNet', help='Log dir [default: log]')
parser.add_argument('-results','--results', required=True, type=str, default='best_model', help='Store the best model')
parser.add_argument('-noise','--use_noise_data', type=str, required=True, default='False', help='Use of Noise in source data in training')
parser.add_argument('-partial','--use_partial_data', type=str, default='True', help='Use of Partial Data for Registration')
parser.add_argument('-sparse','--use_sparse_data', type=str, default='False', help='Use of Vertices Data for Registration')

parser.add_argument('--iterations', type=int, default=8, help='No of Iterations for pose estimation')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ipcr_model_pcn', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=3000000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--model_path', type=str, default='log_multi_catg_noise/model300.ckpt', help='Path of the weights (.ckpt file) to be used for test')
parser.add_argument('--centroid_sub', type=bool, default=True, help='Centroid Subtraction from Source and Template before Pose Prediction.')
parser.add_argument('--use_pretrained_model', type=bool, default=False, help='Use a pretrained model of airplane to initialize the training.')
parser.add_argument('--use_random_poses', type=bool, default=False, help='Use of random poses to train the model in each batch')
parser.add_argument('--data_dict', type=str, default='train_data',help='Templates data used for training network')
parser.add_argument('--train_poses', type=str, default='itr_net_train_data45.csv', help='Poses for training')
parser.add_argument('--eval_poses', type=str, default='itr_net_eval_data45.csv', help='Poses for evaluation')
FLAGS = parser.parse_args()

TRAIN_POSES = FLAGS.train_poses
EVAL_POSES = FLAGS.eval_poses

# Change batch size during test mode.
if FLAGS.mode == 'test':
	BATCH_SIZE = 1
else:
	BATCH_SIZE = FLAGS.batch_size

# Do/Don't Use Noise
if FLAGS.use_noise_data == 'True': ADD_NOISE = 0.5
else: ADD_NOISE = 0.0

if FLAGS.use_sparse_data == 'True': ADD_SPARSE = 0.5
else: ADD_SPARSE = 0.0

if FLAGS.use_partial_data == 'True': ADD_PARTIAL = 0.5
else: ADD_PARTIAL = 0.0

print_('Dataset Settings:',color='r', style='bold')
print_('Noise Added: {}'.format(FLAGS.use_noise_data), color='bl', style='bold')
print_('Sparse Data Added: {}'.format(FLAGS.use_sparse_data), color='bl', style='bold')
print_('Partial Data Added: {}'.format(FLAGS.use_partial_data), color='bl', style='bold')

# Parameters for data
NUM_POINT = FLAGS.num_point
MAX_NUM_POINT = 2048
NUM_CLASSES = 40
centroid_subtraction_switch = FLAGS.centroid_sub

# Network hyperparameters
MAX_EPOCH = FLAGS.max_epoch
MAX_LOOPS = FLAGS.iterations
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Model Import
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

# Take backup of all files used to train the network with all the parameters.
if FLAGS.mode == 'train':
	if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)			# Create Log_dir to store the log.
	os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 				# bkp of model def
	os.system('cp iterative_PCRNet.py %s' % (LOG_DIR)) 			# bkp of train procedure
	os.system('cp train_itrPCRNet.sh %s' % (LOG_DIR)) 			# bkp of train procedure
	os.system('cp -a utils/ %s/'%(LOG_DIR))						# Store the utils code.
	os.system('cp helper.py %s'%(LOG_DIR))
	LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')# Create a text file to store the loss function data.
	LOG_FOUT.write(str(FLAGS)+'\n')

# Write all the data of loss function during training.
def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)
 
# Calculate Learning Rate during training.
def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
						BASE_LEARNING_RATE,  # Base learning rate.
						batch * BATCH_SIZE,  # Current index into the dataset.
						DECAY_STEP,          # Decay step.
						DECAY_RATE,          # Decay rate.
						staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def train():
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			batch = tf.Variable(0)										# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
			
		with tf.device('/gpu:'+str(GPU_INDEX)):
			is_training_pl = tf.placeholder(tf.bool, shape=())			# Flag for dropouts.
			learning_rate = get_learning_rate(batch)					# Calculate Learning Rate at each step.

			# Define a network to backpropagate the using final pose prediction.
			with tf.variable_scope('Network') as _:
				# Get the placeholders.
				#source_pointclouds_pl, template_pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
				source_pointclouds_pl, template_pointclouds_pl, full_source_pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
				source_pcn_feature = MODEL.encoder(source_pointclouds_pl, is_training_pl, bn_decay=None)
				source_pcn_pointclouds = MODEL.decoder(source_pcn_feature, is_training_pl, bn_decay=None)

				# Extract Features.
				source_global_feature, template_global_feature = MODEL.get_model(source_pcn_pointclouds, template_pointclouds_pl, is_training_pl, bn_decay=None)
				# Find the predicted transformation.
				predicted_transformation = MODEL.get_pose(source_global_feature, template_global_feature,is_training_pl, bn_decay=None)
				# Find the loss using source and transformed template point cloud.
				loss = MODEL.get_loss(predicted_transformation, BATCH_SIZE, template_pointclouds_pl, source_pointclouds_pl, full_source_pointclouds_pl, source_pcn_pointclouds)
				# Add the loss in tensorboard.

			# Get training optimization algorithm.
			if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
			elif OPTIMIZER == 'adam':
				optimizer = tf.train.AdamOptimizer(learning_rate)

			train_op = optimizer.minimize(loss, global_step=batch)

		with tf.device('/cpu:0'):
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()
			tf.summary.scalar('loss', loss)
			tf.summary.scalar('learning_rate', learning_rate)

			
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# Add summary writers
		merged = tf.summary.merge_all()
		if FLAGS.mode == 'train':			# Create summary writers only for train mode.
			train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
									  sess.graph)
			eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'))

		# Init variables
		init = tf.global_variables_initializer()
		sess.run(init, {is_training_pl: True})

		# Just to initialize weights with pretrained model.
		if FLAGS.use_pretrained_model:
			saver.restore(sess,os.path.join('log_512pts_1024feat_6itr_180deg_random_poses','model250.ckpt'))

		# Create a dictionary to pass the tensors and placeholders in train and eval function for Network.
		ops = {'source_pointclouds_pl': source_pointclouds_pl,
			   'template_pointclouds_pl': template_pointclouds_pl,
			   'full_source_pointclouds_pl': full_source_pointclouds_pl,
			   'is_training_pl': is_training_pl,
			   'predicted_transformation': predicted_transformation,
			   'loss': loss,
			   'train_op': train_op,
			   'merged': merged,
			   'step': batch}

		poses = helper.read_poses(FLAGS.data_dict, TRAIN_POSES)				# Read all the poses data for training.
		eval_poses = helper.read_poses(FLAGS.data_dict, EVAL_POSES)

		if FLAGS.mode == 'train':
			print(tf.__version__)
			print_('Training Begins!',color='r',style='bold')
			# For actual training.
			for epoch in range(MAX_EPOCH):
				log_string('**** EPOCH %03d ****' % (epoch))
				sys.stdout.flush()
				# Train for all triaining poses.
				train_one_epoch(sess, ops, train_writer, poses)
				save_path = saver.save(sess, os.path.join(LOG_DIR, FLAGS.results+".ckpt"))
				if epoch % 10 == 0:
					# Evaluate the trained network after 50 epochs.
					eval_one_epoch(sess, ops, eval_writer, eval_poses)
				# Save the variables to disk.
				if epoch % 50 == 0:
					# Store the Trained weights in log directory.
					save_path = saver.save(sess, os.path.join(LOG_DIR, "models", "model"+str(epoch)+".ckpt"))
					log_string("Model saved in file: %s" % save_path)

		
# Train the Network and copy weights from Network to Network19 to find the poses between source and template.
def train_one_epoch(sess, ops, train_writer, poses):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:			Dictionary for tensors of Network
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.

	is_training = True
	display_ptClouds = False
	display_poses = False
	display_poses_in_itr = False
	display_ptClouds_in_itr = False

	templates, sources, poses = helper.read_partial_data('train_data','partial_data.h5')
	loss_sum = 0											# Total Loss in each batch.
	num_batches = int(templates.shape[0]/BATCH_SIZE)		# Number of batches in an epoch.

	# Training for each batch.
	for fn in range(num_batches):
		start_idx = fn*BATCH_SIZE 			# Start index of poses.
		end_idx = (fn+1)*BATCH_SIZE 		# End index of poses.
		
		template_data = np.copy(templates[start_idx:end_idx])
		batch_euler_poses = np.copy(poses[start_idx:end_idx])

		# Check for partial source.
		if np.random.sample() < ADD_PARTIAL: 
			source_data = np.copy(sources[start_idx:end_idx])
			source_full_data = helper.apply_transformation(template_data, batch_euler_poses)
		else:
			source_data = np.copy(templates[start_idx:end_idx, 0:512])
			source_data = helper.apply_transformation(source_data, batch_euler_poses)
			source_full_data = helper.apply_transformation(template_data, batch_euler_poses)

		# Check for noise in source.
		if np.random.sample() < ADD_NOISE: source_data = helper.add_noise(source_data)

		if centroid_subtraction_switch:
			source_data = source_data - np.mean(source_data, axis=1, keepdims=True)
			template_data = template_data - np.mean(template_data, axis=1, keepdims=True)
			source_full_data = source_full_data - np.mean(source_full_data, axis=1, keepdims=True)

		# Only chose limited number of points from the source and template data.
		template_data = template_data[:,0:NUM_POINT,:]
		source_full_data = source_full_data[:,0:NUM_POINT,:]

		# To visualize the source and point clouds:
		if display_ptClouds:
			helper.display_clouds_data(source_data[0])
			helper.display_clouds_data(template_data[0])

		TRANSFORMATIONS = np.identity(4)	# Initialize identity transformation matrix.
		TRANSFORMATIONS = npm.repmat(TRANSFORMATIONS,BATCH_SIZE,1).reshape(BATCH_SIZE,4,4)	# Intialize identity matrices of size equal to batch_size

		# Iterations for pose refinement.
		for loop_idx in range(MAX_LOOPS-1):
			# 4a
			# Feed the placeholders of Network with template data and source data.
			feed_dict = {ops['source_pointclouds_pl']: source_data,
						 ops['template_pointclouds_pl']: template_data,
						 ops['is_training_pl']: is_training}
			predicted_transformation = sess.run([ops['predicted_transformation']], feed_dict=feed_dict)	# Ask the network to predict the pose.

			# 4b,4c
			TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

			# Display Results after each iteration.
			if display_poses_in_itr:
				print(predicted_transformation[0,0:3])
				print(predicted_transformation[0,3:7]*(180/np.pi))
			if display_ptClouds_in_itr:
				helper.display_clouds_data(source_data[0])

		# Feed the placeholders of Network with source data and template data obtained from N-Iterations.
		feed_dict = {ops['source_pointclouds_pl']: source_data,
					 ops['template_pointclouds_pl']: template_data,
					 ops['full_source_pointclouds_pl']: source_full_data,
					 ops['is_training_pl']: is_training}

		# Ask the network to predict transformation, calculate loss using distance between actual points, calculate & apply gradients for Network and copy the weights to Network19.
		summary, step, _, loss_val, predicted_transformation = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['predicted_transformation']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)		# Add all the summary to the tensorboard.

		TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

		# Display the ground truth pose and predicted pose for first Point Cloud in batch 
		if display_poses:
			print('Ground Truth Position: {}'.format(batch_euler_poses[0,0:3].tolist()))
			print('Predicted Position: {}'.format(final_pose[0,0:3].tolist()))
			print('Ground Truth Orientation: {}'.format((batch_euler_poses[0,3:6]*(180/np.pi)).tolist()))
			print('Predicted Orientation: {}'.format((final_pose[0,3:6]*(180/np.pi)).tolist()))

		# Display Loss Value.
		print("Batch: {} & Loss: {}\r".format(fn,loss_val),end='')

		# Add loss for each batch.
		loss_sum += loss_val
	print('\n')
	log_string('Train Mean loss: %f\n' % (loss_sum/num_batches))		# Store and display mean loss of epoch.

def eval_one_epoch(sess, ops, eval_writer, poses):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:			Dictionary for tensors of Network
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.

	is_training = False
	display_ptClouds = False
	display_poses = False
	display_poses_in_itr = False
	display_ptClouds_in_itr = False

	loss_sum = 0											# Total Loss in each batch.
	templates, sources, poses = helper.read_partial_data('train_data','partial_data_eval.h5')
	num_batches = int(templates.shape[0]/BATCH_SIZE) 				# Number of batches in an epoch.
	
	for fn in range(num_batches):
		start_idx = fn*BATCH_SIZE 			# Start index of poses.
		end_idx = (fn+1)*BATCH_SIZE 		# End index of poses.

		template_data = np.copy(templates[start_idx:end_idx])
		batch_euler_poses = np.copy(poses[start_idx:end_idx])

		# Check for partial source.
		if np.random.sample() < ADD_PARTIAL:
			source_data = np.copy(sources[start_idx:end_idx])
			source_full_data = helper.apply_transformation(template_data, batch_euler_poses)
		else:
			source_data = np.copy(templates[start_idx:end_idx,0:512])
			source_data = helper.apply_transformation(source_data, batch_euler_poses)
			source_full_data = helper.apply_transformation(template_data, batch_euler_poses)

		# Check for noise in source.
		if np.random.sample() < ADD_NOISE: source_data = helper.add_noise(source_data)


		if centroid_subtraction_switch:
			source_data = source_data - np.mean(source_data, axis=1, keepdims=True)
			template_data = template_data - np.mean(template_data, axis=1, keepdims=True)
			source_full_data = source_full_data - np.mean(source_full_data, axis=1, keepdims=True)

		# Only chose limited number of points from the source and template data.
		template_data = template_data[:,0:NUM_POINT,:]
		source_full_data = source_full_data[:,0:NUM_POINT,:]

		# To visualize the source and point clouds:
		if display_ptClouds:
			helper.display_clouds_data(source_data[0])
			helper.display_clouds_data(template_data[0])

		TRANSFORMATIONS = np.identity(4)				# Initialize identity transformation matrix.
		TRANSFORMATIONS = npm.repmat(TRANSFORMATIONS,BATCH_SIZE,1).reshape(BATCH_SIZE,4,4)		# Intialize identity matrices of size equal to batch_size

		# Iterations for pose refinement.
		for loop_idx in range(MAX_LOOPS-1):
			# 4a
			# Feed the placeholders of Network with template data and source data.
			feed_dict = {ops['source_pointclouds_pl']: source_data,
						 ops['template_pointclouds_pl']: template_data,
						 ops['is_training_pl']: is_training}
			predicted_transformation = sess.run([ops['predicted_transformation']], feed_dict=feed_dict)		# Ask the network to predict the pose.

			TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

			# Display Results after each iteration.
			if display_poses_in_itr:
				print(predicted_transformation[0,0:3])
				print(predicted_transformation[0,3:7]*(180/np.pi))
			if display_ptClouds_in_itr:
				helper.display_clouds_data(source_data[0])

		# Feed the placeholders of Network with source data and template data obtained from N-Iterations.
		feed_dict = {ops['source_pointclouds_pl']: source_data,
					 ops['template_pointclouds_pl']: template_data,
					 ops['full_source_pointclouds_pl']: source_full_data,
					 ops['is_training_pl']: is_training}

		# Ask the network to predict transformation, calculate loss using distance between actual points.
		summary, step, loss_val, predicted_transformation = sess.run([ops['merged'], ops['step'], ops['loss'], ops['predicted_transformation']], feed_dict=feed_dict)

		eval_writer.add_summary(summary, step)			# Add all the summary to the tensorboard.

		TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

		final_pose = helper.find_final_pose_inv(TRANSFORMATIONS)		# Find the final pose (translation, orientation (euler angles in degrees)) from transformation matrix.

		# Display the ground truth pose and predicted pose for first Point Cloud in batch 
		if display_poses:
			print('Ground Truth Position: {}'.format(batch_euler_poses[0,0:3].tolist()))
			print('Predicted Position: {}'.format(final_pose[0,0:3].tolist()))
			print('Ground Truth Orientation: {}'.format((batch_euler_poses[0,3:6]*(180/np.pi)).tolist()))
			print('Predicted Orientation: {}'.format((final_pose[0,3:6]*(180/np.pi)).tolist()))

		# Display Loss Value.
		print("Batch: {}, Loss: {}\r".format(fn, loss_val),end='')

		# Add loss for each batch.
		loss_sum += loss_val
	print('\n')
	log_string('Eval Mean loss: %f' % (loss_sum/num_batches))		# Store and display mean loss of epoch.

if __name__ == "__main__":
	if FLAGS.mode == 'no_mode':
		print('Specity a mode argument: train')
	elif FLAGS.mode == 'train':
		helper.download_data(FLAGS.data_dict)
		train()
		LOG_FOUT.close()