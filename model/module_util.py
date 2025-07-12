import pdb

import torch
from torch_scatter import scatter_mean, scatter_sum, segment_coo
import torch.nn as nn
import numpy as np
from model.Module.Connectivity import FpsPool

class SinusoidalPositionEmbeddings(nn.Module):
	def __init__(self, dim: int):
		super(SinusoidalPositionEmbeddings, self).__init__()
		if dim % 2 != 0:
			raise ValueError(f'Sinusoidal positional encoding with odd d_model: {dim}')
		self.d_model = dim
		div_indices = torch.arange(0, dim, 2).float()
		div_term = torch.exp(div_indices * (-np.log(10000.0) / dim))
		# self.max_val = float(max_val)
		self.register_buffer('div_term', div_term)

	def forward(self, x: torch.Tensor, max_val: float=1.0) -> torch.Tensor:
		r"""Sinusoidal Positional Embedding.

		Args:
			emb_indices: torch.Tensor (*)

		Returns:
			embeddings: torch.Tensor (*, D)
		"""
		input_shape = x.shape

		emb_indices = x

		omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
		sin_embeddings = torch.sin(omegas)
		cos_embeddings = torch.cos(omegas)
		embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
		embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
		embeddings = embeddings.detach()
		return embeddings


class TimestepEmbedder(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size=256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t, dim, max_period=10000):
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
						  These may be fractional.
		:param dim: the dimension of the output. dim = frequency_embedding_size
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t):
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb


def scatter_mean_keepsize(X, index, dim, keepsize=False, sum=False, type=0):
	"""Compute mean of `X` along `dim` using `index`"""
	if type == 0:
		# faster scatter. Index needs to be sorted
		if sum:
			mean = segment_coo(X, index, reduce='sum')
		else:
			mean = segment_coo(X, index, reduce='mean')

		if not keepsize:
			return mean
		else:
			return torch.index_select(mean, dim, index)
	elif type == 1:
		# slower scatter. No requirement for index
		if sum:
			mean = scatter_sum(X, index, dim=dim)
		else:
			mean = scatter_mean(X, index, dim=dim)
		if not keepsize:
			return mean
		else:
			return torch.index_select(mean, dim, index)
	else:
		raise NotImplementedError


def get_graph_scale(g, n_scales: int, pool_ratio: float, knn: int, L: int, Gfeature=False, only_distance=False):
	'''
	Downsample point clouds using Farthest Point Sampling.
	Build knn graph at each scale
	'''
	with torch.no_grad():
		pooling = FpsPool(ratio=pool_ratio, k=knn,  Gfeature=Gfeature, only_distance=only_distance)

		graph_list = []

		bi_graph_list = []
		for i in range(n_scales):

			bi_graph, g = pooling(g, L=L)
			bi_graph_list.append(bi_graph)
			graph_list.append(g)

		return graph_list, bi_graph_list


def scatter_softmax_coo(src: torch.Tensor, index: torch.Tensor, dim: int = -1) -> torch.Tensor:
	"""A slightly faster version of `scatter_softmax` assuming sorted indices."""
	if not torch.is_floating_point(src):
		raise ValueError('`scatter_softmax` can only be computed over tensors '
						 'with floating point data types.')

	assert dim == 0
	max_value_per_index = segment_coo(src, index, reduce='max')
	max_per_src_element = torch.index_select(max_value_per_index, dim, index)

	recentered_scores = src - max_per_src_element
	recentered_scores_exp = recentered_scores.exp_()

	sum_per_index = segment_coo(recentered_scores_exp, index, reduce='sum')
	normalizing_constants = torch.index_select(sum_per_index, dim, index)

	return recentered_scores_exp.div(normalizing_constants)


# if __name__ == '__main__':

	# A = torch.randn((1, 10, 3))
	# get_embedding_indices(A)



	# from torch_scatter import scatter_softmax
	# Sizes = [20, 30]
	# index = torch.cat([torch.ones(s) * i for i, s in enumerate(Sizes)]).type(torch.int64)
	#
	# X = torch.randn((50, 3))
	# V = torch.randn((50, 3))
	#
	# weight = torch.randn((50, 1))
	# weight = scatter_softmax(weight, index, 0)
	#
	# pdb.set_trace()
	# w, t  = regress_se3(X, V, index, weight=weight)
	#
	# w.requires_grad_(True)
	# t.requires_grad_(True)
	#
	#
	# Loss =  (((torch.linalg.cross(torch.index_select(w, 0, index), X) + torch.index_select(t, 0, index) - V) ** 2) * weight).sum()
	#
	# Loss.backward()
	# print(w.grad)
	# print(t.grad)
	#
	# ####
	# # embedder = SinusoidalPositionEmbeddings(dim=8, max_val=1, n=10000.)
	# # a = torch.linspace(0, 1, 100)
	# # b = embedder(a)
	# # pdb.set_trace()
