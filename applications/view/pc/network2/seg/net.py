import torch
from torch import nn
import torch.nn.functional as F


# farthest_point_sample函数完成最远点采样：
# 从一个输入点云中按照所需要的点的个数npoint采样出足够多的点，
# 并且点与点之间的距离要足够远。
# 返回结果是npoint个采样点在原始点云中的索引。
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
    # 直到采样点达到npoint，否则进行如下迭代：
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


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist2 = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist2: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist2 = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist2 += torch.sum(src ** 2, -1).view(B, N, 1)
    dist2 += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist2  # 注意，这里的距离没有开根号，也就是说不是欧氏距离，而是欧氏距离的平方


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


class GAM_Module_revise_by_xmilu(nn.Module):
    """ Group attention module
        S之间，也就是组之间做注意力
        这个的内存和计算消耗要小很多了。。。
    """

    def __init__(self):
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


def sample_and_group(xyz, points, npoint, K_num):
    """
    input:
        xyz：坐标
        points：特征
        npoint：采样多少个中心点
        K_num：邻域点个数
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


class Double_SA(nn.Module):
    """
    对特征points分组，组内按距离权重调整，然后再与中心点坐标做concat，concat之后来一个LBR
    """

    def __init__(self, D_in):
        super(Double_SA, self).__init__()
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels=2 * D_in, out_channels=D_out, kernel_size=1),
        #     nn.BatchNorm2d(num_features=D_out),
        #     nn.ReLU()
        # )
        self.neiSA_new = Neighbor_SA_New(D_in)
        self.gropSA = GAM_Module_revise_by_xmilu()

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
        self.double_sa = Double_SA(D_in=D_in + D_pos)

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


class DecoderBlock(nn.Module):
    def __init__(self, D_in, D_out):
        # 例如D_in=384, D_out=[256, 128],表示输入的维度是384,是forward的两个输入points的维度和，输出先一个LBR映射到256，再一个LBR映射到128
        super(DecoderBlock, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = D_in
        for out_channel in D_out:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):  # xyz是坐标，point是特征feature
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, D', N] # 上采样后的点
        """
        # xyz1 = xyz1.permute(0, 2, 1)  # xyz1是多的点，是等待被插值特征的点
        # xyz2 = xyz2.permute(0, 2, 1)  # xyz2是少的点，是采样后的点，是有更深层语义特征的点
        #
        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # 当采样的点（后层）的个数只有一个的时候，采用repeat直接复制成N个点（前层）
            interpolated_points = points2.repeat(1, N, 1)
        else:

            idx, dists = knn(xyz2.contiguous(), xyz1.contiguous(), 3)

            dist_recip = 1.0 / (dists + 1e-8)  # 距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 对于每一个点的权重再做一个全局的归一化
            # 获得插值点,也就是说，每个邻域点，挑出离自己最近的三个中心点，按照距离对三个中心点插值，得到该邻域点的插值特征
            # index_points(points2, idx)返回【B，N，K=3, D】，与weight.view(B, N, 3, 1)相乘
            # sum函数的dim=2参数表示将K这个维度，求和，也就是将三个最近的中心点加权求和
            # 虽说index_points函数是从以idx为坐标从points里面索引数据，但是即便是idx[B, N, 3]的第二个维度(即点数)比points的多，也没关系，返回的值的第二个维度是N
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1),
                                            dim=2)  # 【B，N，D】
        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            # 拼接上下采样前对应点SA层的特征
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(-1, -2)
        # 对拼接后每一个点都做一个MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.transpose(-1, -2)
        return new_points


class ClabelEmbeding(nn.Module):
    """
    cls token 嵌入
    """

    def __init__(self, num_cls=16, D_out=None):
        super(ClabelEmbeding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_cls, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(),
            nn.Linear(D_out, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x:[B,16]
        """
        x = self.mlp(x)
        return x


class Classifier(nn.Module):
    def __init__(self, D_in, num_part):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(D_in, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_part, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, points):
        points = points.transpose(-1, -2)
        points = self.mlp(points)
        points = points.transpose(-1, -2)
        return points


# 有 label
# class get_model(nn.Module):
#     def __init__(self, num_cls=16, num_part=50):
#         super(get_model, self).__init__()
#         self.embeding = InputEmbeding(D_in=3 + 3, D_out=64)
#
#         self.label_embeding = ClabelEmbeding(num_cls, D_out=64)
#
#         self.encoder1 = StageBlock(D_in=64, D_out=128, D_pos=32, npoint=512, K_num=24)
#         self.encoder2 = StageBlock(D_in=128, D_out=256, D_pos=32, npoint=256, K_num=24)
#         self.encoder3 = StageBlock(D_in=256, D_out=512, D_pos=32, npoint=128, K_num=24)
#         self.encoder4 = StageBlock(D_in=512, D_out=1024, D_pos=32, npoint=64, K_num=24)
#
#         self.decoder4 = DecoderBlock(D_in=512 + 1024, D_out=[1024, 512])
#         self.decoder3 = DecoderBlock(D_in=256 + 512, D_out=[512, 256])
#         self.decoder2 = DecoderBlock(D_in=128 + 256, D_out=[256, 128])
#         self.decoder1 = DecoderBlock(D_in=3 + 64 + 64 + 128, D_out=[256, 128])
#
#         self.classifier = Classifier(D_in=128, num_part=num_part)
#
#     def forward(self, xyz_norm, cls_label):
#         B, N, D = xyz_norm.shape
#
#         xyz_l0, points_l0 = self.embeding(xyz_norm)
#
#         xyz_l1, points_l1 = self.encoder1(xyz_l0, points_l0)
#         xyz_l2, points_l2 = self.encoder2(xyz_l1, points_l1)
#         xyz_l3, points_l3 = self.encoder3(xyz_l2, points_l2)
#         xyz_l4, points_l4 = self.encoder4(xyz_l3, points_l3)
#
#         points_l3 = self.decoder4(xyz_l3, xyz_l4, points_l3, points_l4)  # [B,N,512]  # 输入：上一级点坐标，当前点坐标，上级点特征，当前点特征
#         points_l2 = self.decoder3(xyz_l2, xyz_l3, points_l2, points_l3)  # [B,N,256]
#         points_l1 = self.decoder2(xyz_l1, xyz_l2, points_l1, points_l2)  # [B,N,128]
#
#         label_token = self.label_embeding(cls_label).unsqueeze(1).repeat(1, N, 1)
#         points_l0 = self.decoder1(xyz_l0, xyz_l1, torch.cat([xyz_l0, points_l0, label_token], dim=-1),
#                                   points_l1)  # [B,N,128]
#
#         return self.classifier(points_l0)

class get_model(nn.Module):
    def __init__(self, num_cls=16, num_part=50):
        super(get_model, self).__init__()
        self.embeding = InputEmbeding(D_in=3, D_out=64)

        # self.label_embeding = ClabelEmbeding(num_cls, D_out=64)

        self.encoder1 = StageBlock(D_in=64, D_out=128, D_pos=32, npoint=512, K_num=24)
        self.encoder2 = StageBlock(D_in=128, D_out=256, D_pos=32, npoint=256, K_num=24)
        self.encoder3 = StageBlock(D_in=256, D_out=512, D_pos=32, npoint=128, K_num=24)
        self.encoder4 = StageBlock(D_in=512, D_out=1024, D_pos=32, npoint=64, K_num=24)

        self.decoder4 = DecoderBlock(D_in=512 + 1024, D_out=[1024, 512])
        self.decoder3 = DecoderBlock(D_in=256 + 512, D_out=[512, 256])
        self.decoder2 = DecoderBlock(D_in=128 + 256, D_out=[256, 128])
        self.decoder1 = DecoderBlock(D_in=3 + 64 + 128, D_out=[256, 128])

        self.classifier = Classifier(D_in=128, num_part=num_part)

    def forward(self, xyz_norm, cls_label):
        B, N, D = xyz_norm.shape

        xyz_l0, points_l0 = self.embeding(xyz_norm)

        xyz_l1, points_l1 = self.encoder1(xyz_l0, points_l0)
        xyz_l2, points_l2 = self.encoder2(xyz_l1, points_l1)
        xyz_l3, points_l3 = self.encoder3(xyz_l2, points_l2)
        xyz_l4, points_l4 = self.encoder4(xyz_l3, points_l3)

        points_l3 = self.decoder4(xyz_l3, xyz_l4, points_l3, points_l4)  # [B,N,512]  # 输入：上一级点坐标，当前点坐标，上级点特征，当前点特征
        points_l2 = self.decoder3(xyz_l2, xyz_l3, points_l2, points_l3)  # [B,N,256]
        points_l1 = self.decoder2(xyz_l1, xyz_l2, points_l1, points_l2)  # [B,N,128]

        # label_token = self.label_embeding(cls_label).unsqueeze(1).repeat(1, N, 1)
        points_l0 = self.decoder1(xyz_l0, xyz_l1, torch.cat([xyz_l0, points_l0], dim=-1),
                                  points_l1)  # [B,N,128]

        return self.classifier(points_l0)


#############################################################################################################
import numpy as np
import open3d as o3d


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


num_part = 50


def ply2xyzlArray(ply_path, pth_name="in_used.pth"):
    '''
    给出ply的路径，去预测出一个numpy的矩阵变量，矩阵的形状为[N,7],每一列分别为x，y，z，label
    '''
    classifier = get_model()
    checkpoint = torch.load(f"applications/view/pc/network2/seg/pth/{pth_name}", map_location='cpu')
    # checkpoint = torch.load(
    #     r"C:\Users\84075\PycharmProjects\PointCloudIntelligVis-pear-admin-flask-V2\applications\view\pc\network\seg_model2_shapenetpart.pth",
    #     map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'], )
    classifier = classifier.eval()
    pc = o3d.io.read_point_cloud(ply_path)
    pc = np.asarray(pc.points)
    pc = pc_normalize(pc)

    seg_pred = classifier(torch.Tensor(pc).unsqueeze(0), cls_label=None)
    seg_pred = seg_pred.contiguous().view(-1, num_part)
    pred_choice = seg_pred.data.max(1)[1]
    pred_choice = pred_choice.numpy().reshape(-1, 1)

    out = np.concatenate([pc, pred_choice], axis=1)
    return out


if __name__ == '__main__':
    ply2xyzlArray(r"C:\Users\84075\Desktop\shapenetpart_sample_cls0_0.ply")
