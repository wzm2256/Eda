import argparse
import yaml
import json
import pdb
import os
import logging
import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from torch import optim

from torch_geometric.loader import DataLoader
import util
from Utils import so3_utils, cpath
import pickle
from torch.utils.tensorboard import SummaryWriter
from model.FastAssemblyNet import Eda as Eda

from model.loss import regress_loss, opt_loss3 #opt_loss1 #, opt_loss, opt_loss2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch_ema import ExponentialMovingAverage
import time
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def train_epoch(args, model, ema, train_dataloader, t_sampler, optimizer, scaler, epoch=0, sampler=None):
	model.train()
	rank = args.rank

	if sampler is not None:
		sampler.set_epoch(epoch)

	# with autograd.detect_anomaly():
	if True:
		for step, data in enumerate(train_dataloader):
			PC_t, graph_list_test, bi_graph_list_test = data[0].to(rank), [i.to(rank) for i in data[1]], [i.to(rank) for i in data[2]]
			if 'time' not in PC_t.keys():
				raise NotImplementedError

			with torch.autocast(dtype=torch.float16, enabled=args.use_amp, device_type='cuda'):
				deg1, deg2 = model(PC_t, PC_t.time, graph_list=graph_list_test, bi_graph_list=bi_graph_list_test)

				loss, r_loss, t_loss =  regress_loss(deg2=deg2, deg1=deg1, gt_deg2=PC_t.gt_deg2,
				                                     gt_deg1=PC_t.gt_deg1,
				                                     distance_weight=args.distance_weight)

			optimizer.zero_grad()

			scaler.scale(loss).backward()

			if args.use_gradient_clip:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

			scaler.step(optimizer)
			scaler.update()

			if step % args.ema_step == 0:
				ema.update()

			if args.ddp == 1:
				dist.all_reduce(loss, dist.ReduceOp.AVG, async_op=False)
				dist.all_reduce(r_loss, dist.ReduceOp.AVG, async_op=False)
				dist.all_reduce(t_loss, dist.ReduceOp.AVG, async_op=False)

			if rank == 0:
				args.writer.add_scalar('train/loss',   loss, step+ epoch * len(train_dataloader))
				args.writer.add_scalar('train/r_loss', r_loss, step+ epoch * len(train_dataloader))
				args.writer.add_scalar('train/t_loss', t_loss, step+ epoch * len(train_dataloader))

				args.train_logger.info(f"Epoch: {epoch}\t Step: {step}\t loss: {loss:2f}")



def test_epoch(args, model, test_dataloader, t_sampler, epoch=0, total_step=100,  model_type='model'):
	model.eval()
	rank = args.rank

	with torch.no_grad(), torch.autocast(dtype=torch.float16, enabled=args.use_amp, device_type='cuda'):
		rot_err_list = []
		trans_err_list = []
		rot_err_list_all = []
		trans_err_list_all = []

		Save_data = {}
		Save_data['piece'] = []
		Save_data['batch'] = []
		Save_data['PC'] = []
		Save_data['Error'] = []
		Save_data['True'] = []

		for step, data in enumerate(test_dataloader):

			PC, graph_list_test, bi_graph_list_test = data[0].to(rank), [i.to(rank) for i in data[1]], [i.to(rank) for i in data[2]]

			num_pieces = PC.transform.shape[0]

			Record_r_error = []
			Record_t_error = []
			current_data = []

			g_t = so3_utils.random_SE3(num_pieces, zero_mean=args.center_noise, sigma=args.noise_sigma).to(rank)

			bs = max(PC.transform_batch) + 1
			for repeat_i in range(args.repeat):
				for i in range(total_step):
					t, t_step = t_sampler.get_t(training=False, step=i)
					t = t.repeat(bs).to(g_t.device)

					if args.use_amp:
						tol_step = 1e-5
					else:
						tol_step = 1e-7

					g_t = cpath.Step(t, g_t, t_step, model, PC, order=args.sample_order, alpha=args.sample_alpha,
					                 path_type=args.test_path_type, path_coe=args.path_coe,
					                 graph_list=graph_list_test,
					                 bi_graph_list=bi_graph_list_test, tol=tol_step)


					######################## Record assembly trajectory ########################
					if args.save > 0:
						assert bs == 1, 'To save the assembly trajectory, use batch size 1.'
						PC_t = so3_utils.SE3_action(g_t, PC)
						current_data.append(PC_t.x_cor)

						error = opt_loss3(PC.transform, g_t, PC.transform_batch)
						Record_r_error.append(error[0].item())
						Record_t_error.append(error[1].item())

						if i % 10 == 0:
							args.test_logger.debug(f"{model_type} Epoch: {epoch}\t sample: {step}\t Step: {i} "
							                       f"r_err: {error[0].item():2f}\t t_err: {error[1].item():.2f}")


			error = opt_loss3(PC.transform, g_t, PC.transform_batch)

			if args.Collect_all:
				error_all = opt_loss3(PC.transform, g_t, PC.transform_batch, return_all=args.Collect_all)
				rot_err_list_all.extend(list(error_all[0].cpu().numpy()))
				trans_err_list_all.extend(list(error_all[1].cpu().numpy()))

			if args.ddp == 1:
				error = torch.stack([error[0], error[1]])
				dist.all_reduce(error, dist.ReduceOp.AVG, async_op=False)

			if rank == 0:
				args.test_logger.debug(f"{model_type} Epoch: {epoch}\t sample: {step}\t final_r_err: {error[0].item():2f}  "
			                       f"final_t_err: {error[1].item():.2f}")

				if args.only_test == 0:
					# Record batch error during training.
					args.writer.add_scalar(f'Test_batch/{model_type}/r_err', error[0], step + epoch * len(test_dataloader))
					args.writer.add_scalar(f'Test_batch/{model_type}/t_err', error[1], step + epoch * len(test_dataloader))

			rot_err_list.append(error[0])
			trans_err_list.append(error[1])

			if args.save > 0:
				# Save assembly trajectory. Move the last piece to the ground truth position for better comparison.
				N_piece = PC.transform.shape[0]
				g_true = g_t[N_piece - 1:] @ so3_utils.inv_SE3(PC.transform[N_piece - 1:]) @ PC.transform
				PC_true = so3_utils.SE3_action(g_true, PC)

				Save_data['True'].append(PC_true.x_cor.cpu().numpy())
				Save_data['PC'].append(torch.stack(current_data, 0).cpu().numpy())
				Save_data['piece'].append(PC.piece_index.cpu().numpy())
				Save_data['batch'].append(PC.x_cor_batch.cpu().numpy())
				Save_data['Error'].append(list(zip(Record_r_error, Record_t_error)))

				with open('Vis/data.pk', 'wb') as f:
					pickle.dump(Save_data, f)


		if args.Collect_all:
			# Record all errors
			with open(f'Vis/{args.run_id}_r_error.pk', 'wb') as f:
				pickle.dump(rot_err_list_all, f)

			with open(f'Vis/{args.run_id}_t_error.pk', 'wb') as f:
				pickle.dump(trans_err_list_all, f)
			print('Collect all done')

		rot_err_mean = torch.tensor(rot_err_list).mean()
		trans_err_mean = torch.tensor(trans_err_list).mean()
		if rank == 0:
			args.test_logger.info(f"{model_type} Epoch: {epoch}\t r_err: {rot_err_mean.item():2f} t_err: {trans_err_mean.item():.2f}")

			if args.only_test == 0:
				# Record epoch error during training
				args.writer.add_scalar(f'Test/{model_type}/r_err', rot_err_mean, epoch)
				args.writer.add_scalar(f'Test/{model_type}/t_err', trans_err_mean, epoch)

		return rot_err_mean, trans_err_mean

def train(args):

	if args.ddp == 1:
		torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
		dist.init_process_group("nccl")
		rank = dist.get_rank()
		args.rank = rank
	else:
		rank = 0
		args.rank = rank

	t_sampler = cpath.t_Sampler(args)

	if rank == 0:
		args.train_logger.info(args)

	train_dataset = dataset(
			subset = 'train',
			num_points = args.num_points,
			noise_magnitude = args.noise_magnitude,
			keep_ratio = args.keep_ratio,
			use_normal = args.return_normals,
			pieces_num=args.pieces_num,
			class_indices=args.class_indices,
			L=args.L,
			center_noise=args.center_noise,
			frame_skip=args.frame_skip,
			fix_piece_num=args.fix_piece_num,
			t_sampler=t_sampler,
			noise_sigma=args.noise_sigma,
			distance_weight=args.distance_weight,
			r_type=args.r_type,
			path_type=args.path_type,
			n_scales=args.n_scales,
			pool_ratio=args.pool_ratio,
			knn=args.knn,
			)

	test_dataset = dataset(
			subset = 'test',
			num_points = args.num_points,
			noise_magnitude = args.noise_magnitude,
			keep_ratio = args.keep_ratio,
			use_normal = args.return_normals,
			pieces_num=args.pieces_num,
			class_indices=args.class_indices,
			L=args.L,
			center_noise = args.center_noise,
			frame_skip=args.frame_skip,
			fix_piece_num=args.fix_piece_num,
			t_sampler=t_sampler,
			noise_sigma=args.noise_sigma,
			distance_weight=args.distance_weight,
			r_type=args.r_type,
			path_type=args.path_type,
			n_scales=args.n_scales,
			pool_ratio=args.pool_ratio,
			knn=args.knn,
	)
	args.input_channel = train_dataset.Channel

	train_bs = args.bs
	test_bs = args.bs

	if args.data_name == 'kitti':
		test_bs = min(test_bs, 5)


	if args.ddp == 1:
		sampler_train = DistributedSampler(train_dataset)
		sampler_test = DistributedSampler(test_dataset, shuffle=False)
	else:
		sampler_train = None
		sampler_test = None

	train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=(sampler_train is None),
	                              sampler=sampler_train, follow_batch=['x_cor', 'transform'], num_workers=8, drop_last=True)
	test_dataloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, sampler=sampler_test,
	                             follow_batch=['x_cor', 'transform'], num_workers=8)

	if args.only_test == 0 and args.starting_epoch == 0:
		# starting a new training
		if args.noise_sigma < -0.5:
			if args.ddp == 1:
				raise NotImplementedError

			print('Estimate noise sigma for the whole dataset.')
			args.noise_sigma = util.get_data_std(train_dataloader)
			train_dataset.noise_sigma = args.noise_sigma
			test_dataset.noise_sigma = args.noise_sigma


	args.Max_len = (np.ones(args.n_scales + 1), np.ones(args.n_scales + 1))

	if rank == 0:
		print(f'Max len graph: {args.Max_len[0]}')
		print(f'Max len bi-graph: {args.Max_len[1]}')

	model = Eda(args)
	model.to(rank)
	if args.ddp == 1:
		model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

	if args.optim == 1:
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = optim.AdamW(model.parameters(), lr=args.lr)

	scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

	if args.only_test == 1:
		if args.starting_epoch == -1:
			statedict_uri = os.path.join('LOG', args.run_id, 'best.pt')
		elif args.starting_epoch == -2:
			statedict_uri = os.path.join('LOG', args.run_id, 'last.pt')
		elif args.starting_epoch == -3:
			# Just use random weights without weight loading
			pass
		else:
			statedict_uri = os.path.join('LOG', args.run_id, f'{args.starting_epoch}.pt')

		tar_loaded = torch.load(statedict_uri, map_location={'cuda:0':f'cuda:{rank}'})
		use_ema = True
		if args.starting_epoch != -3:
			if 'model_state_dict' in tar_loaded:
				model.load_state_dict(tar_loaded['model_state_dict'], strict=False)  #
				try:
					ema.load_state_dict(tar_loaded['ema_state_dict'])
				except:
					print('EMA state not loaded.')
					use_ema = False

				scaler.load_state_dict(tar_loaded["scaler"])
				noise_sigma = tar_loaded["noise_sigma"]
			else:
				raise ValueError('No model_state_dict found in the tar file.')

		if args.noise_sigma < -0.5:
			print(f'Load noise sigma {noise_sigma:.2f} for testing.')
			test_dataset.noise_sigma = noise_sigma
			args.noise_sigma = noise_sigma


		rot_err_list = []
		trans_err_list = []
		rot_err_ema_list = []
		trans_err_ema_list = []


		for i in range(args.repeat_test):
			rot_err, trans_err = test_epoch(args, model, test_dataloader, t_sampler, epoch=0, total_step=args.test_step, model_type='model')
			rot_err_list.append(rot_err)
			trans_err_list.append(trans_err)

			# If ema has been loaded, then test ema.
			if use_ema:
				with ema.average_parameters():
					rot_err_ema, trans_err_ema = test_epoch(args, model, test_dataloader, t_sampler, epoch=0, total_step=args.test_step, model_type='ema')
			else:
				rot_err_ema, trans_err_ema = 0., 0.
			rot_err_ema_list.append(rot_err_ema)
			trans_err_ema_list.append(trans_err_ema)

		rot_err_mean = np.mean(rot_err_list)
		trans_err_mean = np.mean(trans_err_list)
		rot_err_ema_mean = np.mean(rot_err_ema_list)
		trans_err_ema_mean = np.mean(trans_err_ema_list)

		if rank == 0:
			save_record_name = f'Result_record/result_{args.run_id}_{args.class_indices}_step_{args.test_step}_order_{args.sample_order}_noise_{args.noise_sigma}.txt'
			with open(save_record_name, 'w') as f:
				f.write(f'rot_err: {rot_err_mean:.2f}, trans_err: {trans_err_mean:.2f}, rot_err_ema: {rot_err_ema_mean:.2f}, trans_err_ema: {trans_err_ema_mean:.2f}\n')

		return

	if args.starting_epoch > 0 or args.starting_epoch == -2:
		# initialize with saved weights
		if args.starting_epoch > 0:
			statedict_uri = os.path.join('LOG', args.run_id, f'{args.starting_epoch}.pt')
		else:
			statedict_uri = os.path.join('LOG', args.run_id, 'last.pt')
		tar_loaded = torch.load(statedict_uri, map_location={'cuda:0': f'cuda:{rank}'})
		model.load_state_dict(tar_loaded['model_state_dict'])
		ema.load_state_dict(tar_loaded['ema_state_dict'])
		optimizer.load_state_dict(tar_loaded['optimizer_state_dict'])
		scaler.load_state_dict(tar_loaded["scaler"])
		noise_sigma = tar_loaded["noise_sigma"]
		test_dataset.noise_sigma = noise_sigma
		train_dataset.noise_sigma = noise_sigma
		args.noise_sigma = noise_sigma

		if args.starting_epoch == -2:
			starting_epoch = tar_loaded['epoch']
		else:
			starting_epoch = args.starting_epoch

		print('---------------')
		print(f'Training starts from: {args.run_id} Step: {starting_epoch}')
		print(f'Load noise sigma {noise_sigma:.2f} to continue training.')
		print('---------------')
	else:
		starting_epoch = 0

	if rank == 0:
		args.writer.add_text('Config', '  \n'.join([f'{k}: \t{v}' for k,v in vars(args).items()]))

	best_score = -np.inf
	for epoch in range(starting_epoch, args.epochs):
		if rank == 0:
			print(f"Epoch {epoch+1}\n-------------------------------")
		train_epoch(args, model, ema, train_dataloader, t_sampler, optimizer, scaler=scaler, epoch=epoch, sampler=sampler_train)

		if args.save_model and epoch % args.save_freq == 0 and rank == 0:
			torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
			            "ema_state_dict": ema.state_dict(),
			            'optimizer_state_dict': optimizer.state_dict(),
			            "scaler": scaler.state_dict(),
			            "noise_sigma": args.noise_sigma},
			           os.path.join('LOG', args.run_id, f'{epoch}.pt'))


			args.train_logger.info(f"Saved: {epoch}.pt")

		if epoch % args.test_feq == 0:
			score, _ = test_epoch(args, model, test_dataloader, t_sampler, epoch=epoch, total_step=args.test_step, model_type='model')
			with ema.average_parameters():
				score_ema, _ = test_epoch(args, model, test_dataloader, t_sampler, epoch=epoch, total_step=args.test_step, model_type='ema')


			if score_ema > best_score and rank == 0:
				best_score = score_ema
				torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
				            "ema_state_dict": ema.state_dict(),
				            'optimizer_state_dict': optimizer.state_dict(),
				            "noise_sigma": args.noise_sigma},
				           os.path.join('LOG', args.run_id, 'best.pt'))

				args.train_logger.info(f"Best model saved")

	if rank == 0:
		torch.save({'epoch': epoch,
	            'model_state_dict': model.state_dict(),
	            "ema_state_dict": ema.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            "scaler": scaler.state_dict(),
	            "noise_sigma": args.noise_sigma
	            }, os.path.join('LOG', args.run_id, 'last.pt'))

		args.train_logger.info(f"Last model saved")


if __name__ == '__main__':
	with open('config.yaml', 'r') as f:
		cfg = yaml.safe_load(f)


	parser = argparse.ArgumentParser()
	for k,v in cfg.items():
		parser.add_argument(f'--{k}', type=type(v), default=v)
	args_ = parser.parse_args()

	## convert string to bool
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


	if args.only_test != 1 and args.starting_epoch == 0:
		# create a new run_id for a new training
		args.run_id = args.data_name + '_' + args.Experiment + '_' + str(np.random.randint(100000)).zfill(8)

	if args.only_test == 1:
		print('---------------')
		print(f'Test run: {args.run_id} Step: {args.starting_epoch}')
		print('---------------')

	if args.ddp == 0 or int(os.environ["LOCAL_RANK"]) == 0:
		tf_log = os.path.join('LOG', args.run_id)
		args.writer = SummaryWriter(tf_log)
		log_file = os.path.join(tf_log, 'log.txt')

		##########
		handlers = [logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
		logging.basicConfig(level=logging.DEBUG, handlers=handlers)
		##########
		args.train_logger = logging.getLogger("training")
		args.test_logger = logging.getLogger("test")

	if args.data_name == 'Match2':
		from Dataset.DMatch2 import MaDataset as dataset
		if args.ddp == 0 or int(os.environ["LOCAL_RANK"]) == 0:
			args.train_logger.info('Using Match-2 dataset.....')
	elif args.data_name == 'BB':
		from Dataset.BB import MaDataset as dataset
		if args.ddp == 0 or int(os.environ["LOCAL_RANK"]) == 0:
			args.train_logger.info('Using BB dataset.....')
	elif args.data_name == 'kitti':
		from Dataset.Kitti import MaDataset as dataset
		if args.ddp == 0 or int(os.environ["LOCAL_RANK"]) == 0:
			args.train_logger.info('Using kitti dataset.....')
	else:
		raise NotImplementedError
	train(args)