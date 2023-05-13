import math
import os

import open3d as o3d
from time import time
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


# 归一化点云，使用以centroid为中心的坐标，球半径为1
def pc_normalize(pc):
    """
        pc:numpy ndarray
    """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# square_distance函数用来在ball query过程中确定每一个点距离采样点的距离
# 函数输入是两组点，N为第一组点src的个数，M为第二组点dst的个数，C为输入点的通道数（如果是xyz时C=3）
# 函数返回的是两组点两两之间的欧几里德距离，即N×M的矩阵
# 在训练中数据以Mini-Batch的形式输入，所以一个Batch数量的维度为B
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  # 注意，这里的距离没有开根号，也就是说不是欧氏距离，而是欧氏距离的平方


# 按照输入的点云数据和索引返回索引的点云数据
# 例如points为B×2048×3点云，idx为[5,666,1000,2000]，
# 则返回Batch中第5,666,1000,2000个点组成的B×4×3的点云集
# 如果idx为一个[B,D1,...DN]，则它会按照idx中的维度结构将其提取成[B,D1,...DN,C]
def index_points(points, idx):
    """
        Input:
        points: input points data, [B, N, C]
        idx: sample data, [B, S] or index [B,S,K]
    Return:
        new_points:, indexed points data, [B, S, C] or [B,S,K,C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# farthest_point_sample函数完成最远点采样: 
# 从一个输入点云中按照所需要的点的个数npoint采样出足够多的点，
# 并且点与点之间的距离要足够远
# 返回结果是npoint个采样点在原始点云中的索引
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: 输入的点云坐标（或特征）矩阵, [B, N, 3]
        npoint: 要从输入xyz中采样多少个点
    Return:
        centroids: 返回的是在输入矩阵xyz中的中心点坐标（或特征）的索引, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
    # 其中B为BatchSize的个数
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices初始化为0~(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 直到采样点达到npoint，否则进行如下迭代: 
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest
        # 取出该中心点centroid的坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小，
        # 其相当于记录着某个Batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return centroids


# query_ball_point函数用于寻找球形邻域中的点
# 输入中radius为球形邻域的半径，nsample为每个邻域中要采样的点，
# new_xyz为centroids点的数据，xyz为所有的点云数据
# 输出为每个样本的每个球形邻域的nsample个采样点集的索引[B,S,nsample]
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    经过测试发现，这个算法是很鲁棒的，即便是给定的半径内没有nsample那么多个点，返回的仍然是nsample个邻域点
    例如，输入的是nsample=24，xyz=[B, N, 3]即要在这个点集矩阵中查询，new_xyz=[B, S, 3]即采样出来的关键点集
    那么返回值就是关键点集在输入xyz里面的索引: [B, S, nsample]
    Input:
        radius: local region radius
        nsample: max sample number in local region,即K值
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] 记录S个中心点（new_xyz）与所有点(xyz)之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距离大于radius^2的点，其group_idx直接置为N；其余的保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），
    # 这种点需要舍弃，直接用第一个点来代替即可
    # group_first: 实际就是把group_idx中的第一个点的值复制；为[B, S, K]的维度，便于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx  # S个group


def knn(origin_xyz, center_xyz, k):
    """Dense knn serach，近邻搜索，如果k=1的情况下，那么相当于没有近邻搜索，直接返回了center_xyz本身
    Arguments:
        origin_xyz - [B,N,3] support points
        center_xyz - [B,S,3] centre of queries
        k - number of neighboors, needs to be > N

    Returns:
        idx - [B,S,k]
        dist2 - [B,S,k] squared distances
    """
    dist2 = square_distance(center_xyz, origin_xyz)  # [B, S, N]
    dist2, idx = torch.sort(dist2, dim=-1, descending=False)  # 在最后一个维度升序排序
    dist2, idx = dist2[:, :, :k], idx[:, :, :k]
    return idx, dist2


# Sampling + Grouping主要用于将整个点云分散成局部的group，
# 对每一个group都可以用PointNet单独地提取局部的全局特征
# Sampling + Grouping分成了sample_and_group和sample_and_group_all两个函数，
# 其区别在于sample_and_group_all直接将所有点作为一个group
# 例如: 
# 512 = npoint: points sampled in farthest point sampling
# 0.2 = radius: search radius in local region
# 32 = nsample: how many points in each local region
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: points sampled in farthest point sampling，即采样多少个中心点（关键点）
        radius: 邻域半径
        nsample: how many points in each local region
        xyz: input points position data, [B, N, 3]，每个原始点的坐标
        points: input points data, [B, N, D]，每个原始点的特征
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]，sample_and_group后的点坐标
        new_points: sampled points data, [B, npoint, nsample, 3+D]，sample_and_group后的点特征
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原点云通过最远点采样挑出的采样点作为new_xyz: 
    # 先用farthest_point_sample函数实现最远点采样得到采样点的索引，
    # 再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    # torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)  # 中心点
    # torch.cuda.empty_cache()
    # idx:[B, npoint, nsample]，代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # torch.cuda.empty_cache()
    # grouped_xyz减去采样点即中心值
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # torch.cuda.empty_cache()

    # 如果每个点上有新的特征的维度，则拼接新的特征与旧的特征，否则直接返回旧的特征
    # 注: 用于拼接点特征数据和点坐标数据
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


# sample_and_group_all直接将所有点作为一个group; npoint=1
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4  # 表示整数除法，返回一个整数；如果用/，那么即便能整除，返回的也是浮点数

        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = self.k_conv.weight  # Q和K的权重初始化为一样的
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input
            x: [B, de, N]

        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)  # [B, da, N]
        x_v = self.v_conv(x)  # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))  # [B, N, N]，至于为什么除以根号da，现在还没看懂
        attention = self.softmax(energy)  # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))

        # residual
        x = x + x_s

        return x


class Neighbor_SA_kaiming(nn.Module):
    """组内做自注意力"""

    def __init__(self, in_channel):
        super(Neighbor_SA_kaiming, self).__init__()
        self.da = in_channel // 4

        self.w_Q = nn.Parameter(
            nn.init.kaiming_uniform_(torch.Tensor(1, 1, in_channel, in_channel // 4), a=0, mode='fan_in'))
        self.w_K = nn.Parameter(
            nn.init.kaiming_uniform_(torch.Tensor(1, 1, in_channel, in_channel // 4), a=0, mode='fan_in'))
        self.w_V = nn.Parameter(
            nn.init.kaiming_uniform_(torch.Tensor(1, 1, in_channel, in_channel), a=0, mode='fan_in'))

        self.trans_conv = nn.Conv2d(in_channel, in_channel, 1)
        self.after_norm = nn.BatchNorm2d(in_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Input
            x: [B, N, K, D]

        Output
            x: [B, N, K, D]
        """
        x_q = torch.matmul(x, self.w_Q)  # [B,N,K,D/4]
        x_k = torch.matmul(x, self.w_K).transpose(-1, -2)  # [B,N,D/4,K]
        x_v = torch.matmul(x, self.w_V).transpose(-1, -2)  # [B,N,D,K]

        # compute attention map and scale, the sorfmax
        energy = torch.matmul(x_q, x_k) / (math.sqrt(self.da))  # [B ,N, K, K]，至于为什么除以根号da，现在还没看懂
        attention = F.softmax(energy, dim=-1)  # [B ,N, K, K]

        # weighted sum
        x_s = torch.matmul(x_v, attention)  # matmul([B,N,D,K],[B ,N, K, K])=[B,N,D,K]
        x_s = x_s.permute(0, 2, 3, 1)  # [B,D,K,N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s))).permute(0, 3, 2, 1)  # [B,N,K,D]

        # residual
        x = x + x_s

        return x


class Neighbor_SA_xavier(nn.Module):
    """组内做自注意力"""

    def __init__(self, in_channel):
        super(Neighbor_SA_xavier, self).__init__()
        self.da = in_channel // 4

        self.w_Q = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(1, 1, in_channel, in_channel // 4)))
        self.w_K = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(1, 1, in_channel, in_channel // 4)))
        self.w_V = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(1, 1, in_channel, in_channel)))

        self.trans_conv = nn.Conv2d(in_channel, in_channel, 1)
        self.after_norm = nn.BatchNorm2d(in_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Input
            x: [B, N, K, D]

        Output
            x: [B, N, K, D]
        """
        x_q = torch.matmul(x, self.w_Q)  # [B,N,K,D/4]
        x_k = torch.matmul(x, self.w_K).transpose(-1, -2)  # [B,N,D/4,K]
        x_v = torch.matmul(x, self.w_V).transpose(-1, -2)  # [B,N,D,K]

        # compute attention map and scale, the sorfmax
        energy = torch.matmul(x_q, x_k) / (math.sqrt(self.da))  # [B ,N, K, K]，至于为什么除以根号da，现在还没看懂
        attention = F.softmax(energy, dim=-1)  # [B ,N, K, K]

        # weighted sum
        x_s = torch.matmul(x_v, attention)  # matmul([B,N,D,K],[B ,N, K, K])=[B,N,D,K]
        x_s = x_s.permute(0, 2, 3, 1)  # [B,D,K,N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s))).permute(0, 3, 2, 1)  # [B,N,K,D]

        # residual
        x = x + x_s

        return x


class Neighbor_SA_New(nn.Module):
    """组内做自注意力"""

    def __init__(self, in_channel):
        super(Neighbor_SA_New, self).__init__()
        self.q_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
                x : [B, S, K, D]
            returns :
                out : attention value + input feature [B, S, K, D]
        """
        proj_query = self.q_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # [B, S, K, D//8]
        proj_key = self.k_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 1, 2)  # [B, S, D//8, K]
        proj_value = self.v_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 1, 2)  # [B, S, D, K]

        energy = torch.matmul(proj_query, proj_key)  # [B, S, K, K]
        attention = self.softmax(energy)  # [B, S, K, K]

        out = torch.matmul(proj_value, attention.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # [B, S, K, D]
        out = self.gamma * out + x  # [B, S, K, D]

        return out


class PAM_Module(nn.Module):
    """ point attention module
        是对所有的点去做注意力
        代码来自DAPnet
        (操，内存消耗和时间消耗就高的离谱)
    """

    def __init__(self, in_channel):
        super(PAM_Module, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : [B, S, K, D]
            returns :
                out : attention value + input feature [B, S, K, D]
        """

        B, S, K, D = x.size()
        x = x.permute(0, 3, 2, 1)  # [B, D, K, S]
        proj_query = self.query_conv(x).view(B, -1, K * S).permute(0, 2, 1)
        # [B, D//8, K, S] -> [B, D//8, K*S] -> [B, K*S, D//8]
        proj_key = self.key_conv(x).view(B, -1, K * S)  # [B, D//8, K, S] -> [B, D//8, K*S]
        energy = torch.bmm(proj_query, proj_key)  # [B, K*S, D//8] * [B, D//8, K*S] = [B, K*S, K*S]
        attention = self.softmax(energy)  # [B, K*S, K*S]
        proj_value = self.value_conv(x).view(B, -1, K * S)  # [B, D, K, S] -> [B, D, K*S]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, D, K*S] * [B, K*S, K*S] = [B, D, K*S]
        out = out.view(B, D, K, S)  # [B, D, K, S]
        out = self.gamma * out + x  # 残差连接,这里由于乘的是一个标量，所以即便是[B, D, K, S]形状，也没关系，最后改变的都是D这一维度的数据，和整形为[B, S, K, D]在乘或加都是一样的结果

        out = out.permute(0, 3, 2, 1)
        return out


class GAM_Module(nn.Module):
    """ Group attention module
        S之间，也就是组之间做注意力
        这个的内存和计算消耗要小很多了
    """

    def __init__(self):
        super(GAM_Module, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps [B, S, K, D]
            returns :
                out : attention value + input feature [B, S, K, D]
        """
        B, S, K, D = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, D, K, S]

        proj_query = x.view(B, D, -1)  # [B, D, K+S]
        proj_key = x.view(B, D, -1).permute(0, 2, 1)  # [B, K+S, D]
        energy = torch.bmm(proj_query, proj_key)  # [B, D, D]
        energy_new = torch.max(energy, dim=-1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(B, D, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(B, D, K, S)

        out = self.gamma * out + x
        out = out.permute(0, 3, 2, 1)
        return out


class get_loss(nn.Module):
    def __init__(self):
        """PointMLP中使用的标签平滑loss，使用这个损失函数的时候，最后一层直接是全连接层，而不是log_softmax"""
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        super(get_loss, self).__init__()

    def forward(self, pred, gold, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss


def get_parameters_scale(model: nn.Module):
    scale = sum(x.numel() for x in model.parameters())
    print("model have {}M paramerters in total".format(scale / (1000 ** 2)))
    return scale


class GAM_Module_revise_by_xmilu(nn.Module):
    """ Group attention module
        S之间，也就是组之间做注意力
        这个的内存和计算消耗要小很多了
    """

    def __init__(self, in_channel):
        """
        in_channel:传进来的参数是 K*D 的值
        """
        super(GAM_Module_revise_by_xmilu, self).__init__()

        # self.query_conv = nn.Linear(in_features=in_channel, out_features=in_channel // 8)
        # self.key_conv = nn.Linear(in_features=in_channel, out_features=in_channel // 8)
        # self.value_conv = nn.Linear(in_features=in_channel, out_features=in_channel)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps [B, S, K, D]
            returns :
                out : attention value + input feature [B, S, K, D]
        """
        B, S, K, D = x.size()

        proj_query = x.view(B, S, -1)  # [B, S, K*D]
        proj_key = x.view(B, S, -1).permute(0, 2, 1)  # [B, S, K*D//8] -> [B, K*D, S]
        energy = torch.bmm(proj_query, proj_key)  # [B, S, S]
        attention = self.softmax(energy)  # [B, S, S]
        proj_value = x.view(B, S, -1)  # [B, S, K*D]

        out = torch.bmm(attention, proj_value)  # [B, S, K*D]
        out = out.view(B, S, K, D)  # [B, S, K, D]

        out = self.gamma * out + x

        return out


class InputEmbeding(nn.Module):
    """
    最开始的那个坐标嵌入模块
    """

    def __init__(self, D_in, D_out):
        super(InputEmbeding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=D_in, out_channels=D_out // 2, kernel_size=1),
            nn.BatchNorm1d(num_features=D_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=D_out // 2, out_channels=D_out, kernel_size=1),
            nn.BatchNorm1d(num_features=D_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz_norm):
        """
            xyz: [B, N, D_in]
            norm:[B, N, D_in]
        return:
            output: [B, N, D_out]
        """
        xyz = xyz_norm[:, :, :3]

        points = xyz_norm.transpose(-1, -2)  # [B, D_in, N]
        points = self.mlp(points)
        points = points.transpose(-1, -2)  # [B, N, D_out]

        return xyz, points


def sample_and_group(xyz, points, npoint, K_num=32):
    """
    input:
        xyz: 坐标
        points: 特征
        npoint: 采样多少个中心点
        K_num: 邻域点个数
    returen:
        center_xyz, center_points, grouped_xyz, grouped_points, fps_idx, knn_idx, knn_dist
    """
    # B, N, D = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)  # 最远点采样的索引 [B, npoint]
    center_xyz = index_points(xyz, fps_idx)  # 中心点坐标 [B, npoint, 3]
    center_points = index_points(points, fps_idx)  # 中心点特征 [B, npoint, D]

    knn_idx, knn_dist = knn(xyz.contiguous(), center_xyz.contiguous(), K_num)  # [B, npoint, k_num] , [B, npoint, k_num]
    grouped_xyz = index_points(xyz, knn_idx)  # [B, npoint, k_num, 3]
    grouped_points = index_points(points, knn_idx)  # [B, npoint, k_num, D]

    return center_xyz, center_points, grouped_xyz, grouped_points, fps_idx, knn_idx, knn_dist


class PosEmbeding(nn.Module):
    def __init__(self, D_pos):
        """
        D_pos:中心坐标、邻居坐标、相对坐标、距离 拼接后的MLP输出的维度
        """
        super(PosEmbeding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=D_pos // 2, kernel_size=1),
            nn.BatchNorm2d(num_features=D_pos // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=D_pos // 2, out_channels=D_pos, kernel_size=1),
            nn.BatchNorm2d(num_features=D_pos),
            nn.ReLU(inplace=True),
        )

    def forward(self, grouped_xyz, center_xyz, knn_dist):
        """
        input:
            grouped_xyz:[B, npoint, k_num, 3]
            center_xyz:[B, npoint, 3]
            knn_dist:[B, npoint, k_num]
        return:
            x:[B, npoint, k_num, D_out]
        """
        B, npoint, k_num, d = grouped_xyz.size()
        # 中心坐标、邻居坐标、相对坐标、距离
        center = center_xyz.unsqueeze(-2).expand(B, npoint, k_num, 3)  # [B, npoint, k_num, 3]
        neibor = grouped_xyz  # [B, npoint, k_num, 3]
        relative = grouped_xyz - center  # [B, npoint, k_num, 3] = [B, npoint, k_num, 3] - [B, npoint, 1, 3]
        dist = knn_dist.unsqueeze(-1)  # [B, npoint, k_num, 1]

        x = torch.cat((center, neibor, relative, dist), dim=-1)  # [B, npoint, k_num, 10]

        x = x.permute(0, 3, 2, 1)
        x = self.mlp(x)
        x = x.permute((0, 3, 2, 1))

        return x


class Double_SA(nn.Module):
    """
    对特征points分组，组内按距离权重调整，然后再与中心点坐标做concat，concat之后来一个LBR
    """

    def __init__(self, K, D_in):
        super(Double_SA, self).__init__()
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels=2 * D_in, out_channels=D_out, kernel_size=1),
        #     nn.BatchNorm2d(num_features=D_out),
        #     nn.ReLU()
        # )
        self.neiSA_new = Neighbor_SA_New(D_in)
        # self.gropSA = GAM_Module()
        self.gropSA = GAM_Module_revise_by_xmilu(in_channel=K * D_in)

    def forward(self, grouped_points):
        """
        input:
            grouped_points:
            center_points:
            knn_dist:
        return:
            x:[B, npoint, k_num, 2D],输出是两倍的 D_in
        """
        # B, n, k, D = grouped_points.size()

        nei_sa_out = self.neiSA_new(grouped_points)  # [B, npoint, k_num, D]
        grop_sa_out = self.gropSA(grouped_points)  # [B, npoint, k_num, D]

        x = torch.cat((nei_sa_out, grop_sa_out), dim=-1)  # [B, npoint, k_num, 2D]

        # x = x.permute(0, 3, 2, 1)
        # x = self.mlp(x)
        # x = x.permute(0, 3, 2, 1)

        return x


class StageBlock(nn.Module):
    def __init__(self, D_in, D_out, D_pos, npoint, K_num):
        super(StageBlock, self).__init__()
        self.npoint = npoint
        self.K_num = K_num

        self.pos_emb = PosEmbeding(D_pos=D_pos)
        self.double_sa = Double_SA(K=K_num, D_in=D_in + D_pos)

        # self.pre_pos_emb_mlp = nn.Sequential(
        #     nn.Conv2d(in_channels=D_in + D_pos, out_channels=D_in + D_pos, kernel_size=1),
        #     nn.BatchNorm2d(num_features=D_in + D_pos),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=D_in + D_pos, out_channels=D_in, kernel_size=1),
        #     nn.BatchNorm2d(num_features=D_in),
        #     nn.ReLU(inplace=True),
        # )

        self.cated_mlp = nn.Sequential(
            nn.Conv2d(in_channels=2 * (D_in + D_pos), out_channels=2 * (D_in + D_pos), kernel_size=1),
            nn.BatchNorm2d(num_features=2 * (D_in + D_pos)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * (D_in + D_pos), out_channels=D_out, kernel_size=1),
            nn.BatchNorm2d(num_features=D_out),
        )

        self.shortcut_mlp = nn.Sequential(
            nn.Conv2d(in_channels=D_in, out_channels=D_out, kernel_size=1),
            nn.BatchNorm2d(num_features=D_out),
            nn.ReLU()
        )

    def forward(self, xyz, points):
        center_xyz, center_points, grouped_xyz, grouped_points, fps_idx, knn_idx, knn_dist = \
            sample_and_group(xyz, points, self.npoint, self.K_num)

        pos_enco_out = self.pos_emb(grouped_xyz, center_xyz, knn_dist)

        grouped_points_pos_embed = torch.cat((pos_enco_out, grouped_points), dim=-1)

        # grouped_points_pos_embed = grouped_points_pos_embed.permute(0, 3, 2, 1)
        # grouped_points_pos_embed = self.pre_pos_emb_mlp(grouped_points_pos_embed)
        # grouped_points_pos_embed = grouped_points_pos_embed.permute(0, 3, 2, 1)

        double_sad = self.double_sa(grouped_points_pos_embed)

        double_sad = double_sad.permute(0, 3, 2, 1)
        double_sad = self.cated_mlp(double_sad)
        double_sad = double_sad.permute(0, 3, 2, 1)

        res = grouped_points.permute(0, 3, 2, 1)
        res = self.shortcut_mlp(res)
        res = res.permute(0, 3, 2, 1)

        new_points = F.relu(double_sad + res, inplace=True)
        new_points, _ = torch.max(new_points, dim=2)  # [B, npoint, 2D]

        # dis_weig_out = self.dis_weig(grouped_points, center_points, knn_dist)  # [B, npoint, k_num, 2D]
        # concat = torch.cat((geo_enco_out, dis_weig_out), dim=-1)  # [B, npoint, k_num, 32+2D]
        # concat = concat.permute(0, 3, 2, 1)
        # concat = self.cated_mlp(concat)
        # concat = concat.permute(0, 3, 2, 1)  # [B, npoint, k_num, 2D]
        #
        # res = grouped_points.permute(0, 3, 2, 1)
        # res = self.shortcut(res)
        # res = res.permute(0, 3, 2, 1)  # [B, npoint, k_num, 2D]
        #
        # new_points = F.relu(concat + res)
        # new_points, _ = torch.max(new_points, dim=2)  # [B, npoint, 2D]

        return center_xyz, new_points  # 即采样的后的中心点坐标和特征 nwe_xyz , new_points


class Classifier(nn.Module):
    def __init__(self, D_in, cls_num):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, cls_num),
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, points):
        """
        """
        x, _ = torch.max(points, dim=-2)
        x = self.mlp(x)  # [B, cls_num]
        return x


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.embeding = InputEmbeding(D_in=3, D_out=64)
        self.stage1 = StageBlock(D_in=64, D_out=128, D_pos=32, npoint=512, K_num=24)
        self.stage2 = StageBlock(D_in=128, D_out=256, D_pos=32, npoint=256, K_num=24)
        self.stage3 = StageBlock(D_in=256, D_out=512, D_pos=32, npoint=128, K_num=24)
        self.stage4 = StageBlock(D_in=512, D_out=1024, D_pos=32, npoint=64, K_num=24)
        self.classifier = Classifier(D_in=1024, cls_num=40)

    def forward(self, xyz_norm):
        xyz_l0, points = self.embeding(xyz_norm)
        xyz_l1, points = self.stage1(xyz_l0, points)
        xyz_l2, points = self.stage2(xyz_l1, points)
        xyz_l3, points = self.stage3(xyz_l2, points)
        xyz_l4, points = self.stage4(xyz_l3, points)
        out = self.classifier(points)

        return out


##############################################################################################################


# class_dict = {
#     1: 'airplane',
#     2: 'bathtub',
#     3: 'bed',
#     4: 'bench',
#     5: 'bookshelf',
#     6: 'bottle',
#     7: 'bowl',
#     8: 'car',
#     9: 'chair',
#     10: 'cone',
#     11: "cup",
#     12: 'curtain',
#     13: 'desk',
#     14: 'door',
#     15: 'dresser',
#     16: 'flower_pot',
#     17: 'glass_box',
#     18: 'guitar',
#     19: 'keyboard',
#     20: 'lamp',
#     21: 'laptop',
#     22: 'vmantel',
#     23: 'monitor',
#     24: 'night_stand',
#     25: 'person',
#     26: 'piano',
#     27: 'plant',
#     28: 'radio',
#     29: 'range_hood',
#     30: 'sink',
#     31: 'sofa',
#     32: 'vstairs',
#     33: 'stool',
#     34: 'table',
#     35: 'tent',
#     36: 'toilet',
#     37: 'tv_stand',
#     38: 'vase',
#     29: 'wardrobe',
#     40: 'xbox',
# }
class_dict = {
    1: '飞机',
    2: '浴缸',
    3: '床',
    4: '长椅',
    5: '书架',
    6: '瓶子',
    7: '碗',
    8: '车',
    9: '椅子',
    10: '锥子',
    11: "杯",
    12: '帘子',
    13: '桌子',
    14: '门',
    15: '梳妆台',
    16: '花盆',
    17: '玻璃盒',
    18: '吉他',
    19: '键盘',
    20: '灯',
    21: '笔记本电脑',
    22: '壁炉架',
    23: '显示器',
    24: '床头柜',
    25: '人',
    26: '钢琴',
    27: '植物',
    28: '收音机',
    29: '油烟机',
    30: '水槽',
    31: '沙发',
    32: '楼梯',
    33: '凳子',
    34: '桌子',
    35: '帐篷',
    36: '厕所',
    37: '电视架',
    38: '花瓶',
    29: '衣柜',
    40: 'XBOX游戏机',
}


def ply2classname(ply_path, pth_name="in_used.pth"):
    classifier = get_model()
    checkpoint = torch.load(f"applications/view/pc/network2/cls/pth/{pth_name}", map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'], )
    classifier = classifier.eval()
    pc = o3d.io.read_point_cloud(ply_path)
    pc = np.asarray(pc.points)
    pc = pc_normalize(pc)
    pc = torch.Tensor(pc).unsqueeze(0)  # [1,N,3]
    pred = classifier(pc)
    pred_choice = int(pred.data.max(1)[1]) + 1
    class_name = class_dict[pred_choice]
    print(f"{ply_path}预测为: {class_name}")
    return class_name
