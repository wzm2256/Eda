import pdb
import torch.utils.data
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
import torch
from typing import Optional, Union
from torch import Tensor
from torch_geometric.utils import cumsum, scatter, coalesce



def get_linear_schedule_with_warmup(
		optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LambdaLR:
	"""
	Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
	a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

	Args:
		optimizer ([`~torch.optim.Optimizer`]):
			The optimizer for which to schedule the learning rate.
		num_warmup_steps (`int`):
			The number of steps for the warmup phase.
		num_training_steps (`int`):
			The total number of training steps.
		last_epoch (`int`, *optional*, defaults to -1):
			The index of the last epoch when resuming training.

	Return:
		`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""

	def lr_lambda(current_step: int):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		return max(
			0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
		optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer ([`~torch.optim.Optimizer`]):
			The optimizer for which to schedule the learning rate.
		num_warmup_steps (`int`):
			The number of steps for the warmup phase.
		num_training_steps (`int`):
			The total number of training steps.
		num_periods (`float`, *optional*, defaults to 0.5):
			The number of periods of the cosine function in a schedule (the default is to just decrease from the max
			value to 0 following a half-cosine).
		last_epoch (`int`, *optional*, defaults to -1):
			The index of the last epoch when resuming training.

	Return:
		`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""

	def lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

	return LambdaLR(optimizer, lr_lambda, last_epoch)



def get_data_std(dataloader: torch.utils.data.DataLoader):

	square_sum = torch.tensor(0.)
	N = 0
	for data in tqdm(dataloader):
		square_sum += (data[0]['transform'][:, :3, 3] ** 2).sum()
		N += data[0]['transform'][:, :3, 3].shape[0]

	std = (square_sum / (N * 3)) ** 0.5
	print(f'Estimated std: {std}')

	return std.item()


def topk(
	x_input: Tensor,
	ratio: Optional[Union[float, int]],
	batch: Tensor,
) -> Tensor:

	if ratio is not None:
		num_nodes = scatter(batch.new_ones(x_input.size(0)), batch, reduce='sum')

		if ratio >= 1:
			k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
		else:
			raise ValueError('k must be greater than 1.')
			# k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

		x, x_perm = torch.sort(x_input.view(-1), descending=True)
		batch = batch[x_perm]
		batch, batch_perm = torch.sort(batch, descending=False, stable=True)

		arange = torch.arange(x_input.size(0), dtype=torch.long, device=x.device)
		ptr = cumsum(num_nodes)
		batched_arange = arange - ptr[batch]
		mask = batched_arange < k[batch]

		return x_perm[batch_perm[mask]]

	raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
					 "must be specified")

def get_piece_wise_link(piece_index, batch_index, degree) -> Tensor:
	arange = torch.arange(batch_index.shape[0], dtype=torch.long, device=batch_index.device)
	Edge = []
	for d in range(degree):
		x_perm = torch.randperm(batch_index.shape[0]).to(batch_index.device)
		batch = batch_index[x_perm]
		batch, batch_perm = torch.sort(batch, descending=False, stable=True)
		permutated = x_perm[batch_perm]

		permutated_piece_index = piece_index[x_perm[batch_perm]]
		Mask = (permutated_piece_index != piece_index)

		Edge.append(torch.stack([arange[Mask], permutated[Mask]], 0))
	Edge_tensor = torch.cat(Edge, 1)
	Edge_tensor_reverse = torch.stack([Edge_tensor[1], Edge_tensor[0]], 0)
	Edge_tensor_bi = torch.cat([Edge_tensor, Edge_tensor_reverse], 1)
	E = coalesce(Edge_tensor_bi)
	return E


def get_piece_wise_link_single_batch(piece_index, degree) -> Tensor:
	arange = torch.arange(piece_index.shape[0], dtype=torch.long, device=piece_index.device)
	Edge = []
	for d in range(degree):
		permutated = torch.randperm(piece_index.shape[0]).to(piece_index.device)
		permutated_piece_index = piece_index[permutated]
		Mask = (permutated_piece_index != piece_index)
		Edge.append(torch.stack([arange[Mask], permutated[Mask]], 0))

	Edge_tensor = torch.cat(Edge, 1)
	Edge_tensor_reverse = torch.stack([Edge_tensor[1], Edge_tensor[0]], 0)
	Edge_tensor_bi = torch.cat([Edge_tensor, Edge_tensor_reverse], 1)
	E = coalesce(Edge_tensor_bi, sort_by_row=False)
	return E

