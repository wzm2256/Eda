import pdb
from typing import List, Tuple, Any, Union, Optional

import torch
import torch.nn as nn
from ..module_util import scatter_mean_keepsize

def get_size(channel: List[int], L: int, type: str = '5601') -> (int, int):
	"""Compute the size of the parameters of the modulate function.
	return the parameters for equivariant feature, and that of invariant feature
	1: (1,      1)
	2: (L,      1)
	3: (C1,     C2)
	4: (L * C1, C2)
	"""
	if type[3] == '1':
		return 1, 1
	elif type[3] == '2':
		return L, 1
	elif type[3] == '3':
		return channel[0], channel[1]
	elif type[3] == '4':
		return channel[0] * L, channel[1]
	else:
		raise NotImplementedError


def modulate(x: torch.Tensor, x_0: torch.Tensor, scale: torch.Tensor, shift_0: torch.Tensor,
             scale_0: torch.Tensor, type: str = '5601') -> (torch.Tensor, torch.Tensor):
	'''
	Applying scale and shift to x and x_0.

	:param x:     B C1 L m
		   x_0:   B C2
		   The shape of scale, shift_0, scale_0 are given by get_size

		To keep the equivariance of x, there is no shift of x, and the scale of x takes the form of  1, (1 or C), (1 or L), 1

		1: scale of shape (1, 1, 1, 1), scale_0 and shift_0 (1, 1)
		2: scale of shape (1, 1, L, 1), scale_0 and shift_0 (1, 1)
		3: scale of shape (1, C, 1, 1), scale_0 and shift_0 (1, C)
		4: scale of shape (1, C, L, 1), scale_0 and shift_0 (1, C)
	:return:
	'''

	if type[3] in ['1', '2']:
		y = x * (1 + scale.reshape([x.shape[0], 1, -1, 1]))
	else:
		y = x * (1 + scale.reshape([x.shape[0], x.shape[1], -1, 1]))

	y_0 = x_0 * (1 + scale_0) + shift_0

	return y, y_0


def gate(x: torch.Tensor, x_0: torch.Tensor, scale: torch.Tensor, scale_0: torch.Tensor, type: str = '5601') \
		-> (torch.Tensor, torch.Tensor):
	'''
	Applying scale to x and x_0 after attention.

		:param x:     B C1 L m
    	       x_0:   B C2

		1: scale of shape (1, 1, 1, 1), scale_0 (1, 1)
		2: scale of shape (1, 1, L, 1), scale_0 (1, 1)
		3: scale of shape (1, C, 1, 1), scale_0 (1, C)
		4: scale of shape (1, C, L, 1), scale_0 (1, C)
	'''
	if type[3] in ['1', '2']:
		y = x * scale.reshape([x.shape[0], 1, -1, 1])
	else:
		y = x * scale.reshape([x.shape[0], x.shape[1], -1, 1])
	y_0 = x_0 * scale_0
	return y, y_0


def square_root(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
	return torch.sqrt(x + eps)

class LayerNorm(nn.Module):
	def __init__(self, Channel: List[int], L: int, eps: float = 1e-5, type: str = '0101', use_affine: bool=True):
		'''

		Shape of the input (P, C, L, M) B/G/P: batch/graph/point numbers. C: channel, L: degree, D: feature dim
		:param type:
		4 digits abcd, where a: compute mean
							b: compute std
							c: using graph norm type parameters
							d: scale and shift
		a:
			n: No normalization
			0: No mean    (defaut for RMS-normalization)
		b:
			1: Shape (G, 1, 1, 1) (default)
			2: Shape (G, 1, L, 1)
			3: Shape (G, C, 1, 1)
			4: Shape (G, C, L, 1)
			5: Shape (B, 1, 1, 1)
			6: Shape (B, 1, L, 1)
			7: Shape (B, C, 1, 1)
			8: Shape (B, C, L, 1)
			A: Shape (P, 1, 1, 1)
			B: Shape (P, 1, L, 1)
			C: Shape (P, C, 1, 1)
			D: Shape (P, C, L, 1)

            # B menas the batch normalization style normalization. It does not performs well.
            # P means normalizating each point independently. It does not performs well.
            # G means the graph normalization style normalization. (default)
            # 1 or C: 1 performs better than C. This is observed in Equiformerv2. Choose 1 as default
            # 1 or L:  For L=2 in our experiments, no big difference is observed. Choose 1 as default.


		d:
			1: Use scale of shape (1, 1, 1, 1) (default)
			2: Use scale of shape (1, 1, L, 1)
			3: Use scale of shape (1, C, 1, 1)
			4: Use scale of shape (1, C, L, 1)

			# We keep the shape consistent with b and choose 1 as default.
			# Equiformerv2 choose 3 as default.

		for x_0,
		The shape of std  is Shape[0:2]
		The shape of scale is Shape[0:2]
		The shape of weight is the same as scale

		# Equiformer2 uses layer norm, while we use rms norm.


		:param normalization:
		'''

		super().__init__()
		self.Channel = Channel
		self.eps = eps


		self.type = type
		self.L = L
		self.a = self.type[0]
		self.b = self.type[1]
		self.c = self.type[2]
		self.d = self.type[3]

		# only use b and d.
		assert self.a =='n' or self.a == '0'
		assert self.c == '0'

		if self.a == 'n':
			return

		self.use_affine = use_affine

		if use_affine:
			self.scale_0 = nn.Parameter(torch.ones((1, Channel[1])))
			if self.d == '1':
				self.scale = nn.Parameter(torch.ones(1, 1, 1, 1))
			elif self.d == '2':
				self.scale = nn.Parameter(torch.ones(1, 1, L, 1))
			elif self.d == '3':
				self.scale = nn.Parameter(torch.ones(1, Channel[0], 1, 1))
			elif self.d == '4':
				self.scale = nn.Parameter(torch.ones(1, Channel[0], L, 1))
			else:
				raise NotImplementedError


	def __repr__(self):
		return f"{self.__class__.__name__}({self.Channel}, L={self.L}, type={self.type}, eps={self.eps})"

	def forward(self, x: torch.Tensor, x_0: torch.Tensor, batch: torch.Tensor, piece: torch.Tensor) -> (torch.Tensor, torch.Tensor):

		# x    P C L M
		# x_0  P C
		if self.a == 'n':
			return x, x_0

		ratio_adjust = (((torch.arange(self.L, device=x.device, dtype=torch.float) + 1) * 2 + 1) / (2 * self.L + 1)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

		if self.b == '5':
			std = square_root(scatter_mean_keepsize(x ** 2, batch, 0, keepsize=True).mean([1, 2, 3], keepdim=True)) # (B, 1, 1, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True).mean(1, keepdim=True))
		elif self.b == '6':
			std = square_root(scatter_mean_keepsize(x ** 2, batch, 0, keepsize=True).mean([1, 3], keepdim=True) / ratio_adjust) # (B, 1, L, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True).mean(1, keepdim=True))
		elif self.b == '7':
			std =  square_root(scatter_mean_keepsize(x ** 2, batch, 0, keepsize=True).mean([2, 3], keepdim=True)) #    (B, C, 1, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True))
		elif self.b == '8':
			std = square_root(scatter_mean_keepsize(x ** 2, batch, 0, keepsize=True).mean([3], keepdim=True) / ratio_adjust) # (B, C, L, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True))
		elif self.b == '1':
			std = square_root(scatter_mean_keepsize(x ** 2, piece, 0, keepsize=True).mean([1, 2, 3], keepdim=True)) # (G, 1, 1, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True).mean(1, keepdim=True))
		elif self.b == '2':
			std = square_root(scatter_mean_keepsize(x ** 2, piece, 0, keepsize=True).mean([1, 3], keepdim=True) / ratio_adjust) # (G, 1, L, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True).mean(1, keepdim=True))
		elif self.b == '3':
			std =  square_root(scatter_mean_keepsize(x ** 2, piece, 0, keepsize=True).mean([2, 3], keepdim=True)) #    (G, C, 1, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True))
		elif self.b == '4':
			std = square_root(scatter_mean_keepsize(x ** 2, piece, 0, keepsize=True).mean([3], keepdim=True) / ratio_adjust) # (G, C, L, 1)
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True))
		elif self.b == 'A':
			std = square_root((x ** 2).mean([1, 2, 3], keepdim=True)) # (P, 1, 1, 1)
			std_0 = square_root((x_0 ** 2).mean(1, keepdim=True))
		elif self.b == 'B':
			std = square_root((x ** 2).mean([1, 3], keepdim=True) / ratio_adjust) # (P, 1, L, 1)
			std_0 = square_root((x_0 ** 2).mean(1, keepdim=True))
		elif self.b == 'C':
			std =  square_root((x ** 2).mean([2, 3], keepdim=True)) #    (P, C, 1, 1)
			std_0 = square_root((x_0 ** 2).mean(1, keepdim=True))
		elif self.b == 'D':
			std = square_root((x ** 2).mean([3], keepdim=True) / ratio_adjust) # (P, C, L, 1)
			std_0 = square_root((x_0 ** 2).mean(1, keepdim=True))
		else:
			raise NotImplementedError

		normalized_x = x / (std + self.eps)
		normalized_x_0 = x_0 / (std_0 + self.eps)
		if self.use_affine:
			y = normalized_x * self.scale
			y_0 = normalized_x_0 * self.scale_0
			return y, y_0
		else:
			return normalized_x, normalized_x_0


class LayerNorm_0(nn.Module):
	'''
	Normalize degree 0 features
	0: LayerNorm (default)
	'''

	def __init__(self, Channel: int,  eps: float = 1e-5, type: str = '0', use_affine: bool=True):
		super().__init__()
		self.Channel = Channel
		self.eps = eps


		self.type = type
		self.e = self.type[0]

		if self.e == '0':
			self.normal = nn.LayerNorm(Channel)
		else:
			self.use_affine = use_affine
			if use_affine:
				self.scale_0 = nn.Parameter(torch.ones((1, Channel)))


	def __repr__(self):
		return f"{self.__class__.__name__}({self.Channel}, L={self.L}, type={self.type}, eps={self.eps})"

	def forward(self, x_0: torch.Tensor, batch: torch.Tensor, piece: torch.Tensor) -> torch.Tensor:
		# x_0  P C
		if self.e == 'n':
			return x_0

		if self.e == '0':
			return self.normal(x_0)

		if self.e == '1':
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True).mean(1, keepdim=True))
		elif self.e == '2':
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, piece, 0, keepsize=True))
		elif self.e == '3':
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True).mean(1, keepdim=True))
		elif self.e == '4':
			std_0 = square_root(scatter_mean_keepsize(x_0 ** 2, batch, 0, keepsize=True))
		else:
			raise NotImplementedError

		normalized_x_0 = x_0 / (std_0 + self.eps)
		if self.use_affine:
			y_0 = normalized_x_0 * self.scale_0
			return y_0
		else:
			return normalized_x_0


if __name__ == '__main__':
	output_channel_list = [2, 1]
	L = 1
	P = 50
	Sizes = [20, 30]
	index = torch.cat([torch.ones(s) * i for i, s in enumerate(Sizes)]).type(torch.int64)

	l = LayerNorm(output_channel_list, L=L, type='0800')
	x = torch.randn((P, output_channel_list[0], L, L * 2 + 1))
	x_0 = torch.randn((P, output_channel_list[1]))
	y, y_0= l(x, x_0, index)
	# pdb.set_trace()


# @ torch.compile(mode='max-autotune', fullgraph=True)
# def Reshape2to3(X: torch.Tensor, Sizes: List):
#     '''
#     Reshaping a matrix of shape (B, c*d1+ c*d2) to (B, C, d1 + d2)
#     :param X: Matrix of shape (B, c*d1+ c*d2)
#     :param Sizes: The Sizes type of X
#     :return: a matrix of shape (B, C, d1 + d2)
#     '''
#
#     X_reshape = []
#     i_start = 0
#     for (d, mul) in Sizes:
#         length = d * mul
#         X_reshape.append(X[:, i_start: i_start + length].reshape((-1, mul, d)))
#         i_start = i_start + length
#     # pdb.set_trace()
#     Y = torch.cat(X_reshape, -1)
#     return Y
#
# # @ torch.compile(mode='max-autotune', fullgraph=True)
# def Reshape3to2(X: torch.Tensor, Sizes: List):
#     '''
#     Reshaping a matrix of shape (B, C, d1 + d2) to (B, c*d1+ c*d2)
#     :param X: Matrix of shape (B, C, d1 + d2)
#     :param Sizes: The Sizes type of X
#     :return: a matrix of shape (B, c*d1+ c*d2)
#     '''
#
#     X_reshape = []
#     i_start = 0
#     for (d, mul) in Sizes:
#         X_reshape.append(X[:, :, i_start: i_start + d].reshape((-1, mul * d)))
#         i_start = i_start + d
#     Y = torch.cat(X_reshape, -1)
#     return Y
