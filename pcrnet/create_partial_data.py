import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import helper

templates = helper.loadData('train_data')

# Display given Point Cloud Data in blue color (default).
def display_clouds_data(data, color='C0'):
	# Arguments:
		# data: 		array of point clouds (num_points x 3)

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	try:
		data = data.tolist()
	except:
		pass
	X,Y,Z = [],[],[]
	for row in data:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	ax.scatter(X,Y,Z, color=color)
	plt.show()

# PyTorch based Code
def sphere_torch(p):
	# args: p, batch of pointclouds [B x N x 3]
	# returns: mask of visible points in p [B x N]
	p_trans = p + 2
	p_mag = torch.norm(p_trans, 2, 2)
	p_COM = torch.mean(p_trans, 1, keepdim=True)
	p_COM_mag = torch.norm(p_COM, 2, 2)
	mask = p_mag < p_COM_mag
	return mask

# Numpy based Code
def sphere(data):
	# args: data (batch of pointclouds [B x N x 3])
	# returns: mask of visible points in data [B x N] and partial data
	data_trans = data + 2
	data_mag = np.linalg.norm(data_trans, 2, 2)
	data_COM = np.mean(data_trans, 1, keepdims=True)
	data_COM_mag = np.linalg.norm(data_COM, 2, 2)
	mask = data_mag < data_COM_mag
	data = [d[mask[idx]] for idx, d in enumerate(data)]
	data = [d[0:512] for d in data]
	return mask, np.array(data)
	
if __name__ == '__main__':
	data = np.array(templates[0:2])
	#poses = np.array([[0,0,0,np.pi/4,np.pi/4,np.pi/4]]) 
	#data = helper.apply_transformation(data, poses)
	mask, data = sphere(data)
	#helper.print_('Number of Points: {}'.format(data[0].shape[0]),color='r',style='bold')
	display_clouds_data(data[1,0:512])
	# display_clouds_data(data_trans[0])
	# display_clouds_data(templates[0,0:128])

