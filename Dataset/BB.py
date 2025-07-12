import os.path
import pdb
from typing import Dict, Optional
from .base import Piece
import os
import numpy as np
import torch.utils.data
import torch
try:
	import open3d as o3d
except:
	print('Can not import open3d. No key points selection and no visualization.')

from .functional import (
	random_jitter_points,
	random_sample_points,
	random_crop_point_cloud_with_plane,
	rotate,
	random_sample_rotation
)
import pickle
from Utils import so3_utils, cpath

from model.module_util import get_graph_scale

class MaDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			dataset_root='Data/BB_preload',
			subset: str = 'train',
			num_points: int = 500,
			noise_magnitude: Optional[float] = None,
			keep_ratio: Optional[float] = None,
			use_normal=0,
			L=1,
			t_sampler=None,
			center_noise=None,
			noise_sigma=None,
			distance_weight=None,
			r_type=None,
			path_type=None,
			n_scales=2,
			knn=5,
			pool_ratio = 0.25,
			class_indices=0,
			**kwargs
	):

		super(MaDataset, self).__init__()

		self.dataset_root = dataset_root

		assert subset in ['train', 'test']
		self.subset = subset
		self.scale  = 1.0

		# Load data. 0 everyday, 1 artifact
		if subset == 'test':
			if class_indices == 0:
				name_list_f = os.path.join(dataset_root, 'everyday.val.txt.pkl')
			elif class_indices == 1:
				name_list_f = os.path.join(dataset_root, 'artifact.val.txt.pkl')
			else:
				raise NotImplementedError
		elif subset == 'train':
			if class_indices == 0:
				name_list_f = os.path.join(dataset_root, 'everyday.train.txt.pkl')
			elif class_indices == 1:
				name_list_f = os.path.join(dataset_root, 'artifact.train.txt.pkl')
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

		with open(name_list_f, 'rb') as f:
			self.All_data_Dic = pickle.load(f)

		self.max_num_points = num_points
		if noise_magnitude is None:
			self.noise_magnitude = 0.
		else:
			self.noise_magnitude = noise_magnitude

		self.rotation_magnitude = 180.

		if keep_ratio is None or subset == 'test':
			self.keep_ratio = 1.0
		else:
			self.keep_ratio = keep_ratio

		self.use_normal = use_normal

		self.L = L
		self.Channel = np.array([0, 0])

		if self.use_normal:
			self.Channel[0] += 1

		if self.Channel[1] == 0:
			self.Channel[1] += 1

		self.t_sampler = t_sampler

		self.center_noise = center_noise
		self.noise_sigma = noise_sigma
		self.distance_weight = distance_weight
		self.r_type = r_type
		self.path_type = path_type

		self.n_scales = n_scales
		self.pool_ratio = pool_ratio
		self.knn = knn


	def __getitem__(self, index):
		random_seed = None
		label = 0

		rng = np.random.RandomState(random_seed)
		current_data_list = self.All_data_Dic[index]

		data_list = [i[0] for i in current_data_list]
		normal_list = [i[1] for i in current_data_list]


		sample_piece_num = len(data_list)
		data_list = [i * self.scale for i in data_list]

		r = random_sample_rotation(batch=sample_piece_num, rotation_deg=self.rotation_magnitude, rng=rng)

		PC_list = []
		Normal_list = []
		transform = np.zeros((sample_piece_num, 4, 4), dtype=np.float32)
		transform[:, 3, 3] = 1
		transform[:, :3, :3] = np.transpose(r, (0, 2, 1))

		for i in range(sample_piece_num):
			random_seed_i = None
			pc, normals = data_list[i], normal_list[i]
			pc, normals = random_sample_points(pc, self.max_num_points, normals=normals, seed=random_seed_i)

			keep_ratio = np.random.rand() * (1 - self.keep_ratio) + self.keep_ratio
			pc, normals = random_crop_point_cloud_with_plane(pc, keep_ratio=keep_ratio, normals=normals, seed=random_seed_i)
			pc = random_jitter_points(pc, scale=0.01, noise_magnitude=self.noise_magnitude, seed=random_seed_i)

			t = np.mean(pc, 0)
			pc = pc - np.expand_dims(t, 0)

			if pc.ndim != 2:
				# data error
				pdb.set_trace()

			transform[i, :3, 3] = t
			pc, normals = rotate(pc, r[i], normals=normals)
			PC_list.append(pc)
			if normals is not None:
				Normal_list.append(normals)

		mean_t = np.mean(transform[:, :3, 3], 0, keepdims=True)
		transform[:, :3, 3] = transform[:, :3, 3] - mean_t

		PC_all = np.concatenate(PC_list, 0)
		batch = np.concatenate([np.ones(p.shape[0], dtype=np.int64) * n for n, p in enumerate(PC_list)])

		feature_0_list = []
		feature_0_list.append(torch.ones(PC_all.shape[0], 1))
		Feature_0 = torch.cat(feature_0_list, -1)


		Feature = torch.zeros((Feature_0.shape[0], self.Channel[0], self.L, self.L * 2 + 1))
		if self.use_normal == 1:
			Feature[:, 0:1, 0:1, self.L - 1: self.L + 2] = torch.tensor(np.concatenate(Normal_list, 0)).reshape(-1, 1, 1, 3)

		PC = Piece(x_cor=torch.tensor(PC_all.astype(np.float32)),
				  piece_index=torch.tensor(batch),
				  transform=torch.tensor(transform.astype(np.float32)),
				  label=torch.tensor(int(label)),
				  index=torch.tensor(int(index)),
				  x_f=Feature,
				  x_f_0=Feature_0,
		          piece_num=torch.tensor(len(PC_list))
				  )


		if self.subset == 'test':
			# create graph
			graph_list, bi_graph_list = get_graph_scale(PC, n_scales=self.n_scales,
			                                                      pool_ratio=self.pool_ratio, knn=self.knn,
			                                                      L=self.L, only_distance=True)
			return PC, graph_list, bi_graph_list

		time = self.t_sampler.get_t(batch_size=1, training=True)


		g0 = so3_utils.random_SE3(piece_num=sample_piece_num, zero_mean=self.center_noise, sigma=self.noise_sigma,
		                          seed=None)

		global_r = cpath.get_global_r(g0=g0, g1=torch.tensor(transform), distance_weight=self.distance_weight,
		                              seed_noise=None, r_type=self.r_type,
		                              batch_index=torch.zeros((sample_piece_num), dtype=torch.long))

		g_t, gt_deg2, gt_deg1 = so3_utils.corrupt(g0, global_r @ transform, time, path_type=self.path_type)


		PC_t = so3_utils.SE3_action(g_t, PC)
		PC_t.gt_deg2 = gt_deg2
		PC_t.gt_deg1 = gt_deg1
		PC_t.time = time
		graph_list, bi_graph_list = get_graph_scale(PC_t, n_scales=self.n_scales,
		                                            pool_ratio=self.pool_ratio, knn=self.knn,
		                                            L=self.L, only_distance=False)
		return PC_t, graph_list, bi_graph_list

	def __len__(self):
		return len(self.All_data_Dic)




if __name__ == '__main__':
	import torch.distributed as dist
	from torch.utils.data.distributed import DistributedSampler as DistributedSampler

	from torch_geometric.loader import DataLoader
	import Utils.vis as vis
	import time
	# get_file('/mimer/NOBACKUP/groups/naiss2023-22-572/BiEq/Data/match/')
	import yaml
	import argparse
	from tqdm import tqdm
	##########
	with open('config.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	parser = argparse.ArgumentParser()
	for k,v in cfg.items():
		parser.add_argument(f'--{k}', type=type(v), default=v)
	args_ = parser.parse_args()
	args_dict = {}
	for k, v in vars(args_).items():
		if type(v) == str and v.lower() == 'false':
			v = False
		elif type(v) == str and v.lower() == 'true':
			v = True
		if type(v) == str and ',' in v:
			v = [int(i) for i in v.split(',')]
		args_dict.update({k:v})
	args = argparse.Namespace(**args_dict)
	##########

	# t_sampler = cpath.t_Sampler(args)
	t_sampler = None
	train_dataset = MaDataset(
			dataset_root = 'Vis',
			subset = 'test',
			num_points = args.num_points,
			noise_magnitude = args.noise_magnitude,
			keep_ratio = args.keep_ratio,
			asymmetric = args.asymmetric,
			use_normal = args.return_normals,
			pieces_num=args.pieces_num,
			overfit=args.overfit,
			fix_index=args.fix_index,
			use_xcor=args.use_xcor,
			rotation_magnitude=args.rotation_magnitude,
			class_indices=1,
			L=args.L,
			center_noise=args.center_noise,
			voxel_size=args.voxel_size,
			frame_skip=args.frame_skip,
			fix_piece_num=args.fix_piece_num,
			preload=args.preload,
			t_sampler=t_sampler,
			# corrput=Corrupt,
			noise_sigma=args.noise_sigma,
			distance_weight=args.distance_weight,
			r_type=args.r_type,
			path_type=args.path_type,
			n_scales=args.n_scales,
			pool_ratio=args.pool_ratio,
			knn=args.knn
			)



	train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, follow_batch=['x_cor', 'transform'], num_workers=0)
	# pdb.set_trace()

	# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, follow_batch=['x_cor', 'transform'], pin_memory=True, num_workers=8)

	# data1 = bunny[0]
	t1 = time.time()
	M = 0
	m = 100000000000

	print(f'Len {len(train_dataset)}')

	# for epoch in range(1):
	# for n, data in tqdm(enumerate(train_dataloader)):
	for Data in tqdm(train_dataloader):

		# time.sleep(1.)
	# 	pc0 = data.x_cor[data.piece_index == 0]
	# 	pc1 = data.x_cor[data.piece_index == 1]
	#
	# 	for size in [pc0.shape[0], pc1.shape[0]]:
	# 		if size > M:
	# 			M = size
	#
	# 		if size < m:
	# 			m = size
	#
	# 	# pdb.set_trace()
	# 	# print(n)
	# 	# pass
	# # pdb.set_trace()
	# # t2 = time.time()
	#
	# # print(f'Time:{t2 - t1:.2f}')
	# print(f'M: {M} m {m}')

		data = Data[0]
		pc_list = []
		transformed_pc_list = []
		for i in range(data['transform'].shape[0]):
			pc = data.x_cor[data.piece_index==i]

			# radius of pc
			radius = torch.max(torch.norm(pc, dim=1))
			print('Radius:', radius)
			print('Number of points:', pc.shape[0])
			# print('Center:', torch.mean(pc, 0))

			pc_list.append(pc)
			transformed_pc = pc @ data.transform[i][:3, :3].T + np.expand_dims(data.transform[i][:3, 3], 0)

			transformed_pc_list.append(transformed_pc)

		All_pc = torch.cat(pc_list, 0)
		print('Target center:', torch.mean(All_pc, 0))

		vis.visualize_3d(transformed_pc_list)