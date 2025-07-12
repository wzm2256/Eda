from torch_geometric.data import Data
import torch
class Piece(Data):
	'''
	The base class for piece data.
	Specify how to increase index and concatenate for different attributes.
	'''

	def __inc__(self, key, value, *args, **kwargs):

		if key in ('piece_index', 'u_piece', 'v_piece'):
			return self.piece_num
		if 'batch' in key and isinstance(value, torch.Tensor):
			return int(value.max()) + 1
		if key == 'e' and 'v_cor' not in self.keys():
			return self.x_cor.shape[0]
		if key == 'e_cross_random' and 'v_cor' not in self.keys():
			return self.x_cor.shape[0]

		if key == 'e' and 'v_cor' in self.keys():
			return torch.tensor([[self.u_cor.shape[0]], [self.v_cor.shape[0]]])

		if key == 'v_idx':
			return self.u_cor.shape[0]
		return 0

	def __cat_dim__(self, key, value, *args, **kwargs):
		if key == 'e' or key =='e_cross_random':
			return 1
		else:
			return 0
