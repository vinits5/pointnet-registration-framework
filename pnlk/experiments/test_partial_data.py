import torch
import torchvision
import torch.utils.data
import numpy as np
import transforms3d.euler as t3d
import helper
import argparse
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

class Action:
	def __init__(self, model_path):
		# PointNet
		self.dim_k = 1024
		self.sym_fn = ptlk.pointnet.symfn_max

		# LK
		self.delta = 1.0e-2
		self.max_iter = 20
		self.xtol = 1.0e-7
		self.p0_zero_mean = True
		self.p1_zero_mean = True
		self.model_path = model_path
		self.device = 'cpu'

	def create_model(self):
		ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
		return ptlk.pointlk.PointLK(ptnet, self.delta)

	def eval_1(self, model, source, template):
		model.eval()
		with torch.no_grad():
			p0, p1 = torch.from_numpy(template).to(self.device), torch.from_numpy(source).to(self.device)
			r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean)
			est_g = model.g # (1, 4, 4)
			g_hat = est_g.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
			# f_source, f_template = f_source.cpu().numpy(), f_template.cpu().numpy()
		return g_hat

	def run(self, source, template):
		if not torch.cuda.is_available():
			self.device = 'cpu'
		self.device = torch.device(self.device)
		model = self.create_model()

		if self.model_path:
			assert os.path.isfile(self.model_path)
			model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
		model.to(self.device)

		# testing
		return (self.eval_1(model, source, template)).cpu().numpy()


if __name__ == '__main__':
	act = Action(os.path.join('results_train_data','ex1_pointlk_0915_model_best.pth'))
	templates, sources, poses = helper.read_partial_data('train_data', 'partial_data.h5')
	template_data = templates[0].reshape(1,-1,3)
	source_data = sources[0].reshape(1,-1,3)

	#source_data = helper.apply_transformation(template_data, poses[0].reshape(1,-1))
	transformation = act.run(source_data, template_data)

	pred_data = np.matmul(transformation[0,0:3,0:3],source_data[0].T).T
	
	pred_data = pred_data - np.mean(pred_data, axis=0, keepdims=True)
	helper.display_three_clouds(template_data[0], source_data[0], pred_data, "PointNetLK Partial Source Result")

