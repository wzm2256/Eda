import pdb
from typing import Tuple, List, Dict, Optional, Union

import torch
from Dataset.base import Piece
from torch_geometric.nn import fps

from torch_geometric.nn import knn, knn_graph
from torch_geometric.utils.mask import index_to_mask

from ..FastSo3 import so3
import einops
from torch_scatter import scatter_mean, scatter_sum, segment_coo


def knn_bi_graph(
    x: torch.Tensor,
    node_dst_idx: torch.Tensor,
    k: int,
    piece_x: Optional[torch.Tensor] = None,
    loop: bool = False,
    cosine: bool = False,
    num_workers: int = 1,
    batch_x: Optional[torch.Tensor] = None,
    L:int = 1,
    eps:float = 1e-3,
    g_fine=None,
    only_distance=False
):

    """Create a bi-graph from a point cloud using kNN"""
    y = x.index_select(index=node_dst_idx, dim=0)
    piece_y = piece_x.index_select(index=node_dst_idx, dim=0)
    batch_y = batch_x.index_select(index=node_dst_idx, dim=0)

    num_node_x = segment_coo(torch.ones(piece_x.shape[0]), piece_x, reduce='sum')
    contain_small_graph = (num_node_x < k + 1).any()

    if 'e' not in g_fine.keys() or contain_small_graph:
    # Compute the bi-graph.
        edge_index = knn(x, y, k if loop else k + 1, piece_x, piece_y, cosine, num_workers)
        edge_ind = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_vec = x[edge_ind[0]] - y[edge_ind[1]]
        d = torch.norm(edge_vec, dim=1, p=2)
        Nonzero = edge_ind[:, d > eps]
        d_Nonzero = d[d > eps]
        edge_vec_Nonzero = edge_vec[d > eps]
    else:
    # if the fine graph has been created, just extract the edge
        node_dst_mask = index_to_mask(node_dst_idx, size=x.shape[0])
        edge_mask = node_dst_mask[g_fine.e[1, :]]
        edge_x = g_fine.e[0, edge_mask]
        bi_edge_renew = einops.repeat(torch.arange(node_dst_idx.shape[0], device=g_fine.e.device), 'b -> (b r)', r=k)
        Nonzero = torch.stack([edge_x, bi_edge_renew])
        edge_vec_Nonzero = g_fine.edge_vec[edge_mask]
        d_Nonzero = g_fine.edge_length[edge_mask]

    bi_graph = Piece(u_cor=x, v_cor=y, u_piece=piece_x, v_piece=piece_y, e=Nonzero, u_batch=batch_x, v_batch=batch_y,
                    type='bi', v_idx=node_dst_idx, edge_length=d_Nonzero, edge_vec=edge_vec_Nonzero)

    if only_distance:
        return bi_graph

    bi_graph.uv_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(bi_graph.edge_vec, norm=bi_graph.edge_length), L)

    return bi_graph


def knn_graph_g(x_cor, k=32, L: int = 1, piece_index=None, x_cor_batch=None, loop=False, eps=1e-3, Gfeature=False,
                only_distance=False):
    """Create a graph from a point cloud using kNN"""

    e_ = knn_graph(x_cor, k=k, batch=piece_index, loop=loop)
    d1 = torch.norm(x_cor[e_[0]] - x_cor[e_[1]], dim=1, p=2)
    e = e_[:, d1 > eps]
    edge_vec = x_cor.index_select(0, e[0]) - x_cor.index_select(0, e[1])
    edge_length = edge_vec.norm(dim=1, p=2)

    g = Piece(x_cor=x_cor, e=e, x_cor_batch=x_cor_batch, piece_index=piece_index, type=None, edge_vec=edge_vec,
              edge_length=edge_length)

    if only_distance:
        return g

    g.e_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(edge_vec, norm=edge_length), L)

    return g

    # compute target based feature
    # target_group_edge_vec = torch.index_select(scatter_group(edge_vec, g.e[1], 0, size=x_cor.shape[0])[0], 0, g.e[1]) # e k 3
    # source_group_edge_vec = torch.index_select(scatter_group(edge_vec, g.e[1], 0, size=x_cor.shape[0])[0], 0, g.e[0]) # e k 3
    #
    # edge_vec_expand = edge_vec.unsqueeze(1) # e 1 3
    # angle_target = compute_angle(edge_vec_expand, target_group_edge_vec)
    # angle_source = compute_angle(edge_vec_expand, -source_group_edge_vec)
    #
    # g.v_angle = angle_target
    # g.u_angle = angle_source
    #
    # return g


class FpsPool(torch.nn.Module):
    '''
    Farthest Point Sampling
    '''
    def __init__(self, ratio: float = 0.5, k: int = 32, Gfeature=False, only_distance=False):
        super().__init__()
        self.ratio: float = ratio
        self.k: int = k
        self.Gfeature = Gfeature
        self.only_distance = only_distance

    def forward(self, g, L: int):
        node_coord_src = g.x_cor
        piece_src = g.piece_index
        if 'x_cor_batch' not in g.keys():
            batch_src = torch.zeros((g.x_cor.shape[0]), dtype=torch.long)
        else:
            batch_src = g.x_cor_batch

        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3

        # FPS downsampling
        node_dst_idx = fps(node_coord_src, batch=piece_src, ratio=self.ratio, random_start=False)

        # if a piece is too small, then keep all points in that piece.
        num_node = segment_coo(torch.ones(node_coord_src.shape[0]), piece_src, reduce='sum')
        Mask = (num_node < self.k)
        Mask_node = torch.index_select(Mask, 0, piece_src)
        Index_node = torch.arange(node_coord_src.shape[0])[Mask_node]
        node_dst_idx = torch.unique(torch.cat([Index_node, node_dst_idx]))

        if 'e' in g.keys():
            node_dst_idx = torch.sort(node_dst_idx)[0]

        # compute bi-graph
        bi_graph = knn_bi_graph(node_coord_src, node_dst_idx=node_dst_idx, k=self.k, piece_x=piece_src, loop=False,
                                batch_x=batch_src, L=L, g_fine=g, only_distance=self.only_distance)

        # compute coarse g
        coarse_g = knn_graph_g(bi_graph.v_cor, L=L, k=self.k, piece_index=bi_graph.v_piece, x_cor_batch=bi_graph.v_batch,
                               loop=False, Gfeature=self.Gfeature, only_distance=self.only_distance)
        bi_graph.piece_num = g.piece_num
        coarse_g.piece_num = g.piece_num

        return bi_graph, coarse_g

        # if not hasattr(g, 'edge_vec'):
        #     fine_g = knn_graph_g(bi_graph.u_cor, L=L, k=self.k, piece_index=bi_graph.u_piece, x_cor_batch=bi_graph.u_batch, loop=False, Gfeature=self.Gfeature)
        #     fine_edge_vec = fine_g.edge_vec
        #     fine_edge = fine_g.e
        #     g.edge_length = fine_g.edge_length
        #     g.edge_vec = fine_edge_vec
        #     g.e = fine_g.e
        # else:
        #     fine_edge_vec = g.edge_vec
        #     fine_edge = g.e
        #
        #
        # # geo distance
        # target_group_edge_vec = torch.index_select(scatter_group(coarse_g.edge_vec, coarse_g.e[1], 0, size=coarse_g.x_cor.shape[0])[0], 0, bi_graph.e[1])  # e k 3
        # source_group_edge_vec = torch.index_select(scatter_group(fine_edge_vec,      fine_edge[1], 0, size=g.x_cor.shape[0])[0], 0, bi_graph.e[0])  # e k 3
        # edge_vec_expand = bi_graph.edge_vec.unsqueeze(1)  # e 1 3
        # bi_graph.v_angle = compute_angle(edge_vec_expand, target_group_edge_vec)
        # bi_graph.u_angle = compute_angle(edge_vec_expand, -source_group_edge_vec)
        #
        # return bi_graph, coarse_g

# def scatter_group(x: torch.Tensor, index: torch.Tensor, dim: int=0, size: Optional[int]=None):
#     assert dim == 0, 'Only support dim 0!'
#     sorted_batch, indices = torch.sort(index, 0)
#     sorted_x = x[indices]
#     grouped_x, mask = to_dense_batch(sorted_x, sorted_batch, batch_size=size)
#     return grouped_x, mask
#
#
# def compute_angle(edge_vec_expand: torch.Tensor, group_edge_vec: torch.Tensor):
#     sin_values = torch.linalg.norm(torch.cross(edge_vec_expand, group_edge_vec, dim=-1), dim=-1)  # (e k)
#     cos_values = torch.sum(edge_vec_expand * group_edge_vec, dim=-1)  # (e k)
#     angles = torch.atan2(sin_values, cos_values)  # (e k)
#     return angles


# def knn_bi_graph_general(
#     x: torch.Tensor,
#     y:torch.Tensor,
#     k: int,
#     piece_x: Optional[torch.Tensor] = None,
#     piece_y: Optional[torch.Tensor] = None,
#     batch_x: Optional[torch.Tensor] = None,
#     batch_y: Optional[torch.Tensor] = None,
#     loop: bool = False,
#     # flow: str = 'source_to_target',
#     cosine: bool = False,
#     num_workers: int = 1,
#     L:int = 1,
#     eps:float = 1e-3,
#     g_fine=None,
#     only_distance=False
# ):
#
#
#     edge_index = knn(x, y, k if loop else k + 1, piece_x, piece_y, cosine, num_workers)
#     edge_ind = torch.stack([edge_index[1], edge_index[0]], dim=0)
#     edge_vec = x[edge_ind[0]] - y[edge_ind[1]]
#     d = torch.norm(edge_vec, dim=1, p=2)
#     Nonzero = edge_ind[:, d > eps]
#     d_Nonzero = d[d > eps]
#     edge_vec_Nonzero = edge_vec[d > eps]
#
#     bi_graph = Piece(u_cor=x, v_cor=y, u_piece=piece_x, v_piece=piece_y, e=Nonzero, u_batch=batch_x, v_batch=batch_y,
#                     type='bi', v_idx=torch.zeros(dtype=torch.long), edge_length=d_Nonzero, edge_vec=edge_vec_Nonzero)
#
#     if only_distance:
#         return bi_graph
#
#     bi_graph.uv_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(bi_graph.edge_vec, norm=bi_graph.edge_length), L)
#     # bi_graph.vu_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(-bi_graph.edge_vec, norm=bi_graph.edge_length), L)
#
#     return bi_graph

