import pdb

try:
	import open3d as o3d
except:
	print('Warnining open3d is not loaded. No visualization.')
import matplotlib


def view_with_direction(LandScape, point_size=8., parameters=None, savename=None):
	vis = o3d.visualization.Visualizer()
	vis.create_window()

	if isinstance(LandScape, list):
		for i in LandScape:
			vis.add_geometry(i)
	else:
		vis.add_geometry(LandScape)

	if parameters is not None:
		ctr = vis.get_view_control()
		ctr.convert_from_pinhole_camera_parameters(parameters)

	render = vis.get_render_option()
	render.point_size = point_size

	vis.run()
	if savename is not None:
		vis.capture_screen_image(savename)
	# vis.destroy_window()



def visualize_3d(A, A_color=None, viewpoint=None, point_size=1., savename=None):
	cmap = matplotlib.cm.get_cmap('gist_rainbow')
	color_2_list = [[1, 0, 0],
	                [0, 0, 1.]]
	color_3_list = [[1, 0., 0.],
	                [0, 0.651, 0.929],
	                [1, 0.706, 0]]

	pc_list = []
	for i in range(len(A)):
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(A[i])
		if A_color is not None:
			pc.colors = o3d.utility.Vector3dVector(A_color[i])
		else:
			if len(A) == 2:
				pc.paint_uniform_color(color_2_list[i])
			elif len(A) == 3:
				pc.paint_uniform_color(color_3_list[i])
			else:
				rgba = cmap(i / (len(A) - 1))
				pc.paint_uniform_color(rgba[:3])
		pc_list.append(pc)

	if viewpoint != None:
		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
	else:
		parameters = None

	if savename is not None:
		current_savename = savename + '_vis3.png'
	else:
		current_savename = None

	view_with_direction(pc_list, parameters=parameters, point_size=point_size, savename=current_savename)



#
# def get_continuous_se3(N):
# 	alpha = np.linspace(0, np.pi * 2, N)
# 	beta = np.linspace(0, np.pi, N)
# 	theta = np.linspace(0, np.pi * 2, N)
#
# 	Rot_list = []
# 	trans_list = []
# 	t_old = torch.randn(3).to('cuda')
# 	t_old = t_old / t_old.norm()
#
# 	for i in range(N):
# 		r = R.from_quat([np.sin(theta[i] / 2) * np.sin(alpha[i]) * np.cos(beta[i]),
# 						 np.sin(theta[i] / 2) * np.sin(alpha[i]) * np.sin(beta[i]),
# 						 np.sin(theta[i] / 2) * np.cos(alpha[i]), np.cos(theta[i] / 2)])
# 		# pdb.set_trace()
# 		Rot_list.append(torch.tensor(r.as_matrix(), dtype=torch.float).to('cuda'))
# 		trans_list.append(t_old * (3 * i / N))
#
# 	return Rot_list, trans_list
#
# def isotropic_R_error(r1, r2):
# 	'''
# 	Calculate isotropic rotation degree error between r1 and r2.
# 	:param r1: shape=(B, 3, 3), pred
# 	:param r2: shape=(B, 3, 3), gt
# 	:return:
# 	'''
# 	r2_inv = r2.permute(0, 2, 1).contiguous()
# 	r1r2 = torch.matmul(r2_inv, r1)
# 	############
# 	# tr_old = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
# 	############
# 	tr = torch.vmap(torch.trace)(r1r2)
# 	############
# 	rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
# 	degrees = rads / math.pi * 180
# 	return degrees
#
#
#
# def matrix2euler(mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
#     """Converts rotation matrix to euler angles
#
#     Args:
#         mats: (B, 3, 3) containing the B rotation matricecs
#         seq: Sequence of euler rotations (default: 'zyx')
#         degrees (bool): If true (default), will return in degrees instead of radians
#
#     Returns:
#
#     """
#
#     eulers = []
#     for i in range(mats.shape[0]):
#         r = R.from_matrix(mats[i])
#         eulers.append(r.as_euler(seq, degrees=degrees))
#     return np.stack(eulers)
#
# def matrix2vec(mats: np.ndarray, degrees: bool = True):
#     """Converts rotation matrix to euler angles
#
#     Args:
#         mats: (B, 3, 3) containing the B rotation matricecs
#         seq: Sequence of euler rotations (default: 'zyx')
#         degrees (bool): If true (default), will return in degrees instead of radians
#
#     Returns:
#
#     """
#
#     eulers = []
#     for i in range(mats.shape[0]):
#         r = R.from_matrix(mats[i])
#         eulers.append(r.as_rotvec(degrees=degrees))
#     return np.stack(eulers)
#
#
#
# def compute_metrics(R, t, gtR, gtt, euler=False):
# 	# cur_r_mse, cur_r_mae = anisotropic_R_error(R, gtR)
# 	# cur_t_mse, cur_t_mae = anisotropic_t_error(t, gtt)
# 	cur_r_isotropic = isotropic_R_error(R, gtR)
# 	# cur_t_isotropic = isotropic_t_error(t, gtt, gtR)
# 	cur_t_isotropic = (t - gtt).norm(dim=-1)
#
# 	r_gt_euler_deg = matrix2euler(gtR[:, :3, :3].detach().cpu().numpy(), seq='zxy')
# 	r_pred_euler_deg = matrix2euler(R[:, :3, :3].detach().cpu().numpy(), seq='zxy')
# 	# pdb.set_trace()
# 	# print(r_pred_euler_deg)
# 	euler_error =  np.abs(r_gt_euler_deg - r_pred_euler_deg)
# 	# print(euler_error)
#
# 	# z_error = torch.arccos(R[:, 2, 2]) / np.pi * 180
# 	# print(z_error )
# 	# rotvec = matrix2vec(R[:, :3, :3].detach().cpu().numpy())
# 	# rotvec = rotvec / (np.linalg.norm(rotvec) + 1e-5)
# 	# z_degree = np.arccos(rotvec[0, 2]) / np.pi * 180
# 	# print(z_degree)
# 	# pdb.set_trace()
# 	# print(rotvec)
# 	if euler:
# 		return cur_r_isotropic, cur_t_isotropic, euler_error
# 		# return cur_r_isotropic, cur_t_isotropic, z_error.unsqueeze(0)
#
#
#
# 	return cur_r_isotropic, cur_t_isotropic, 0.0



# def vis_PC(A, savename=None, viewpoint=None , point_size=1.):
# 	if len(A) == 2:
# 		visualize_3d_2(A, savename=savename, viewpoint=viewpoint, point_size=point_size)
# 	else:
# 		visualize_3d_3(A, savename=savename, viewpoint=viewpoint, point_size=point_size)


#
#
#
# def visualize_3d_5(A, viewpoint=None, point_size=1., savename=None):
# 	template = o3d.geometry.PointCloud()
# 	template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample = o3d.geometry.PointCloud()
# 	sample.points = o3d.utility.Vector3dVector(A[1])
#
# 	sample2 = o3d.geometry.PointCloud()
# 	sample2.points = o3d.utility.Vector3dVector(A[2])
#
# 	sample3 = o3d.geometry.PointCloud()
# 	sample3.points = o3d.utility.Vector3dVector(A[3])
#
# 	sample4 = o3d.geometry.PointCloud()
# 	sample4.points = o3d.utility.Vector3dVector(A[4])
#
#
# 	sample.paint_uniform_color([1, 0.655, 0.655])
# 	template.paint_uniform_color([1, 0, 0.196])
# 	sample2.paint_uniform_color([1, 0.78, 0])
# 	sample3.paint_uniform_color([0.106, 0, 1])
# 	sample4.paint_uniform_color([0.325, 1, 0])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		current_savename = savename + '_vis3.png'
# 	else:
# 		current_savename = None
#
# 	view_with_direction([template, sample, sample2, sample3, sample4],
# 						parameters=parameters, point_size=point_size, savename=current_savename)
#
# def visualize_3d_4(A, viewpoint=None, point_size=1., savename=None):
# 	template = o3d.geometry.PointCloud()
# 	template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample = o3d.geometry.PointCloud()
# 	sample.points = o3d.utility.Vector3dVector(A[1])
#
# 	# sample2 = o3d.geometry.PointCloud()
# 	# sample2.points = o3d.utility.Vector3dVector(A[2])
#
# 	# sample3 = o3d.geometry.PointCloud()
# 	# sample3.points = o3d.utility.Vector3dVector(A[3])
#
# 	template.paint_uniform_color([1, 0.718, 0.882])
# 	sample.paint_uniform_color([0.3, 0.86, 0.92])
#
# 	# template.paint_uniform_color([0, 1, 0])
# 	# sample.paint_uniform_color([1, 1, 0])
#
# 	mesh_spheres = []
# 	for kp in range(A[2].shape[0]):
# 		mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
# 		mesh_sphere.translate(A[2][kp])
# 		mesh_sphere.paint_uniform_color([1, 0, 0])
# 		mesh_spheres.append(mesh_sphere)
#
# 	for kp in range(A[3].shape[0]):
# 		mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
# 		mesh_sphere.translate(A[3][kp])
# 		mesh_sphere.paint_uniform_color([0, 0, 1])
# 		mesh_spheres.append(mesh_sphere)
#
#
# 	# sample2.paint_uniform_color([1, 0, 0])
# 	# sample3.paint_uniform_color([0, 0, 1.])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		current_savename = savename + '_vis3.png'
# 	else:
# 		current_savename = None
#
# 	view_with_direction([template, sample] + mesh_spheres,
# 						parameters=parameters, point_size=point_size, savename=current_savename)
#
#
# def visualize_3d_single_color(A, viewpoint=None, point_size=1., savename=None):
# 	# template = o3d.geometry.PointCloud()
# 	# template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample = o3d.geometry.PointCloud()
# 	sample.points = o3d.utility.Vector3dVector(A[0])
#
# 	# sample2 = o3d.geometry.PointCloud()
# 	# sample2.points = o3d.utility.Vector3dVector(A[2])
# 	#
# 	sample3 = o3d.geometry.PointCloud()
# 	sample3.points = o3d.utility.Vector3dVector(A[1])
#
# 	# template.paint_uniform_color([1, 0.706, 0])
# 	sample.paint_uniform_color([0.3, 0.86, 0.92])
#
# 	# sample2.paint_uniform_color([1, 0, 0])
# 	sample3.paint_uniform_color([0, 0, 1.])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		current_savename = savename + '_vis3.png'
# 	else:
# 		current_savename = None
#
# 	view_with_direction([sample, sample3],
# 						parameters=parameters, point_size=point_size, savename=current_savename)
#
#
# def visualize_3d_3(A, viewpoint=None, point_size=1., savename=None):
# 	template = o3d.geometry.PointCloud()
# 	template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample = o3d.geometry.PointCloud()
# 	sample.points = o3d.utility.Vector3dVector(A[1])
#
# 	sample2 = o3d.geometry.PointCloud()
# 	sample2.points = o3d.utility.Vector3dVector(A[2])
#
# 	template.paint_uniform_color([1, 0., 0.])
# 	sample.paint_uniform_color([0, 0.651, 0.929])
# 	sample2.paint_uniform_color([1, 0.706, 0])
#
# 	# template.paint_uniform_color([0, 1., 0.])
# 	# sample.paint_uniform_color([0, 1., 0.])
# 	# sample2.paint_uniform_color([0, 1., 0.])
#
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		current_savename = savename + '_vis3.png'
# 	else:
# 		current_savename = None
#
# 	view_with_direction([template, sample, sample2], parameters=parameters, point_size=point_size, savename=current_savename)
#
#
# def visualize_3d_3_color(A, A_color, viewpoint=None, point_size=1., savename=None):
#
# 	PC_list = []
# 	for i in range(len(A)):
# 		template = o3d.geometry.PointCloud()
# 		template.points = o3d.utility.Vector3dVector(A[i])
# 		template.colors = o3d.utility.Vector3dVector(A_color[i])
# 		PC_list.append(template)
#
# 	# sample = o3d.geometry.PointCloud()
# 	# sample.points = o3d.utility.Vector3dVector(A[1])
# 	# sample.colors = o3d.utility.Vector3dVector(A_color[1])
# 	#
# 	# sample2 = o3d.geometry.PointCloud()
# 	# sample2.points = o3d.utility.Vector3dVector(A[2])
# 	# sample2.colors = o3d.utility.Vector3dVector(A_color[2])
#
# 	# template.paint_uniform_color([1, 0.706, 0])
# 	# sample.paint_uniform_color([0, 0.651, 0.929])
# 	# sample2.paint_uniform_color([0, 1., 0.])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		current_savename = savename + '_vis3.png'
# 	else:
# 		current_savename = None
#
# 	view_with_direction(PC_list, parameters=parameters, point_size=point_size, savename=current_savename)
#
#
#
# def visualize_3d_2(A, viewpoint=None, point_size=1., savename=None):
# 	template = o3d.geometry.PointCloud()
# 	template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample2 = o3d.geometry.PointCloud()
# 	sample2.points = o3d.utility.Vector3dVector(A[1])
# 	sample2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0., 0., 1.]]), [A[1].shape[0], 1]))
#
# 	template.paint_uniform_color([1, 0, 0])
# 	sample2.paint_uniform_color([0, 0, 1.])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	if savename is not None:
# 		my_savename = savename + '_vis2.png'
# 	else:
# 		my_savename = None
# 	view_with_direction([template, sample2], parameters=parameters, point_size=point_size, savename=my_savename)
#
#
#
#
#
# def vis_gradnorm(All_points, d_loss, point_mass):
# 	gradients = grad(outputs=d_loss, inputs=All_points, grad_outputs=torch.ones(d_loss.size()).to('cuda'),
# 					 create_graph=False, retain_graph=False)[0].contiguous()
# 	grad_norm = (gradients / point_mass).norm(2, dim=1, keepdim=True)
# 	return grad_norm
#
#
# def visualize_3_2_step(A, viewpoint=None, point_size=1., savename=None):
# 	template = o3d.geometry.PointCloud()
# 	template.points = o3d.utility.Vector3dVector(A[0])
#
# 	sample2 = o3d.geometry.PointCloud()
# 	sample2.points = o3d.utility.Vector3dVector(A[1])
# 	sample2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0., 0., 1.]]), [A[1].shape[0], 1]))
#
# 	template.paint_uniform_color([1, 0, 0])
# 	sample2.paint_uniform_color([0, 0, 1.])
#
# 	if viewpoint != None:
# 		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
# 	else:
# 		parameters = None
#
# 	view_with_direction([template], parameters=parameters, point_size=point_size, savename=savename + '_vis3_step1.png')
# 	view_with_direction([sample2], parameters=parameters, point_size=point_size, savename=savename + '_vis3_step2.png')
# 	view_with_direction([template, sample2], parameters=parameters, point_size=point_size,  savename=savename + '_vis3.png')
#
