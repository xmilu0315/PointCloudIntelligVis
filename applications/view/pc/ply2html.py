import os
import numpy as np
import k3d

import open3d as o3d


def pack(r, g, b):
    """
        (r,g,b) tuple to hex. Each r,g,b can be column arrays

    """
    return (
            (np.array(r).astype(np.uint32) << 16)
            + (np.array(g).astype(np.uint32) << 8)
            + np.array(b).astype(np.uint32)
    )


def pack_single(rgb):
    """
        [r,g,b] one line array to hex
    """
    return (
            (rgb[0] << 16)
            + (rgb[1] << 8)
            + rgb[2]
    )


def generate_k3d_plot(xyz, rgb=None, mask_map=None, mask_color=None,
                      name_map=None, old_plot=None):
    """
        Generates a k3d snapshot of a set of 3d points, mapping them either
        to their true rgb color or a colour corresponding to their label.
        Labels are also mapped to names, so that they can be easily toggled
        inside the visualization tool.

    Args:
        xyz: array of [x, y, z] points
        rgb: array of [r, g, b] points inside [0, 255]
        mask_map: dict mapping each label to a mask over xyz, which allows to
            select points from each class
        mask_color: dict mapping each label to a single color
        name_map: map each numeric label to a descriptive string name

    Returns:
        k3d snapshot that can be saved as html for visualization

    参数的中文解释：
    xyz，就是点的坐标，形状为【N，3】的numpy矩阵，最好归一化到-1~1，不然显示起来会有问题
    rgb，就是形状为【N,3】的颜色，值范围为0-255，行顺序与xyz对应
    mask_map，类型为字典，字典的元素数为类别数，每个元素为N长度的一维numpy的矩阵，矩阵的类型为bool类型，样例如下：
                    {0: array([False,  True,  True, ...,  True,  True,  True]), #array的长度为N
                     1: array([False, False, False, ..., False, False, False]),
                     2: array([False, False, False, ..., False, False, False]),
                     3: array([ True, False, False, ..., False, False, False]),
                     4: array([False, False, False, ..., False, False, False])}
    mask_color，类型为字典，存储每一个类别颜色值，值为0-255，样例如下：
                     {0: [161, 201, 244],
                      1: [255, 180, 130],
                      2: [208, 187, 255],
                      3: [255, 159, 155],
                      4: [141, 229, 161]}
    name_map，类型为字典，存储每一个类别的标签名，值为Str，样例如下:
                     {0: 'terrain',
                      1: 'construction',
                      2: 'urban_asset',
                      3: 'vegetation',
                      4: 'vehicle'}
    """
    kwargs = dict()
    if rgb is None:
        pass
    else:
        assert mask_color is None
        kwargs["colors"] = pack(rgb[:, 0], rgb[:, 1], rgb[:, 2])

    if old_plot is None:
        plot = k3d.plot()
    else:
        plot = old_plot

    if mask_map is None:
        plt_points = k3d.points(positions=xyz,
                                point_size=0.03,
                                shader="mesh",
                                **kwargs)
        plot += plt_points
    else:
        for label in mask_map:
            mask = mask_map[label]
            if name_map is None:
                legend_label = f"label {label}"
            else:
                legend_label = f"{name_map[label]}"
            if mask_color is None:
                colors = kwargs["colors"][mask]
                plt_points = k3d.points(positions=xyz[mask],
                                        point_size=0.03,
                                        shader="mesh",
                                        name=legend_label,
                                        colors=colors)
                plot += plt_points
            else:
                color = pack_single(mask_color[label])
                plt_points = k3d.points(
                    positions=xyz[mask],
                    point_size=0.03,
                    shader="mesh",
                    name=legend_label,
                    color=color,
                )
                plot += plt_points
    plot.camera_mode = 'orbit'
    plot.grid_auto_fit = False
    # plot.grid = np.concatenate((np.min(xyz, axis=0), np.max(xyz, axis=0)))
    plot.grid_visible = False
    return plot


def pc_normalize(pc):
    """
        pc:numpy ndarray
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


######################################################################################
# 用于分类任务的
def ply_cls2html(ply_path, html_out_path):
    pc = o3d.io.read_point_cloud(ply_path)

    xyz = np.asarray(pc.points)
    xyz = pc_normalize(xyz)
    if len(pc.colors):
        rgb = np.asarray(pc.colors)
        rgb = rgb / np.max(rgb) * 255
        plot = generate_k3d_plot(xyz=xyz, rgb=rgb)
    else:
        gray = np.asarray([[200, 200, 200]])
        gray = np.repeat(gray, len(pc.points), axis=0)
        plot = generate_k3d_plot(xyz=xyz, rgb=gray)

    snapshot = plot.get_snapshot(9)
    with open(html_out_path, 'w') as fp:
        fp.write(snapshot)
        print(f"ply_cls2html输出文件到：{html_out_path}")


######################################################################################
# 用于分割任务的
# {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#            'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
class_dict = {
    0: 'Airplane_part1',
    1: 'Airplane_part2',
    2: 'Airplane_part3',
    3: 'Airplane_part4',
    4: 'Bag_part1',
    5: 'Bag_part2',
    6: 'Cap_part1',
    7: 'Cap_part2',
    8: 'Car_part1',
    9: 'Car_part2',
    10: 'Car_part3',
    11: 'Car_part4',
    12: 'Chair_part1',
    13: 'Chair_part2',
    14: 'Chair_part3',
    15: 'Chair_part4',
    16: 'Earphone_part1',
    17: 'Earphone_part2',
    18: 'Earphone_part3',
    19: 'Guitar_part1',
    20: 'Guitar_part2',
    21: 'Guitar_part3',
    22: 'Knife_part1',
    23: 'Knife_part2',
    24: 'Lamp_part1',
    25: 'Lamp_part2',
    26: 'Lamp_part3',
    27: 'Lamp_part4',
    28: 'Laptop_part1',
    29: 'Laptop_part2',
    30: 'Motorbike_part1',
    31: 'Motorbike_part2',
    32: 'Motorbike_part3',
    33: 'Motorbike_part4',
    34: 'Motorbike_part5',
    35: 'Motorbike_part6',
    36: 'Mug_part1',
    37: 'Mug_part2',
    38: 'Pistol_part1',
    39: 'Pistol_part2',
    40: 'Pistol_part3',
    41: 'Rocket_part1',
    42: 'Rocket_part2',
    43: 'Rocket_part3',
    44: 'Skateboard_part1',
    45: 'Skateboard_part2',
    46: 'Skateboard_part3',
    47: 'Table_part1',
    48: 'Table_part2',
    49: 'Table_part3',
}
color_map = [[31, 119, 180],
             [255, 127, 14],
             [44, 160, 44],
             [214, 39, 40],
             [148, 103, 189],
             [140, 86, 75],
             [227, 119, 194],
             [127, 127, 127],
             [188, 189, 34],
             [31, 119, 180],
             [174, 199, 232],
             [255, 127, 14],
             [255, 187, 120],
             [44, 160, 44],
             [152, 223, 138],
             [214, 39, 40],
             [255, 152, 150],
             [148, 103, 189],
             [197, 176, 213],
             [140, 86, 75],
             [196, 156, 148],
             [227, 119, 194],
             [247, 182, 210],
             [127, 127, 127],
             [199, 199, 199],
             [188, 189, 34],
             [219, 219, 141],
             [23, 190, 207],
             [228, 26, 28],
             [55, 126, 184],
             [77, 175, 74],
             [152, 78, 163],
             [255, 127, 0],
             [255, 255, 51],
             [166, 86, 40],
             [247, 129, 191],
             [102, 194, 165],
             [252, 141, 98],
             [141, 160, 203],
             [231, 138, 195],
             [166, 216, 84],
             [255, 217, 47],
             [229, 196, 148],
             [141, 211, 199],
             [255, 255, 179],
             [190, 186, 218],
             [251, 128, 114],
             [128, 177, 211],
             [253, 180, 98],
             [179, 222, 105],
             [252, 205, 229],
             [217, 217, 217],
             [188, 128, 189]]


def ply_seg2html(xyzl_ndarray, html_out_path):
    # xyz，就是点的坐标，形状为【N，3】的numpy矩阵，最好归一化到-1~1，不然显示起来会有问题
    # rgb，就是形状为【N,3】的颜色，值范围为0-255，行顺序与xyz对应
    # mask_map，类型为字典，字典的元素数为类别数，每个元素为N长度的一维numpy的矩阵，矩阵的类型为bool类型，样例如下：
    #                 {0: array([False,  True,  True, ...,  True,  True,  True]), #array的长度为N
    #                  1: array([False, False, False, ..., False, False, False]),
    #                  2: array([False, False, False, ..., False, False, False]),
    #                  3: array([ True, False, False, ..., False, False, False]),
    #                  4: array([False, False, False, ..., False, False, False])}
    # mask_color，类型为字典，存储每一个类别颜色值，值为0-255，样例如下：
    #                  {0: [161, 201, 244],
    #                   1: [255, 180, 130],
    #                   2: [208, 187, 255],
    #                   3: [255, 159, 155],
    #                   4: [141, 229, 161]}
    # name_map，类型为字典，存储每一个类别的标签名，值为Str，样例如下:
    #                  {0: 'terrain',
    #                   1: 'construction',
    #                   2: 'urban_asset',
    #                   3: 'vegetation',
    #                   4: 'vehicle'}

    xyz = xyzl_ndarray[:, 0:3].astype(np.float32)
    xyz = pc_normalize(xyz).astype(np.float32)

    l = xyzl_ndarray[:, -1].astype(int)
    unique_l_list = np.unique(l).tolist()

    mask_map = {}
    for unique_l in unique_l_list:
        mask_map[int(unique_l)] = (l == int(unique_l))

    mask_color = {}
    for unique_l in unique_l_list:
        mask_color[int(unique_l)] = color_map[int(unique_l)]

    name_map = {}
    for unique_l in unique_l_list:
        name_map[int(unique_l)] = class_dict[int(unique_l)]

    plot = generate_k3d_plot(xyz=xyz, mask_map=mask_map, mask_color=mask_color, name_map=name_map)

    snapshot = plot.get_snapshot(9)
    with open(html_out_path, 'w') as fp:
        fp.write(snapshot)
        print(f"ply_seg2html输出文件到：{html_out_path}")


if __name__ == '__main__':
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    from network.seg_model_2 import ply2xyzlArray

    xyzl = ply2xyzlArray(r"C:\Users\84075\Desktop\shapenetpart_sample_cls0_0.ply")

    ply_seg2html(xyzl, "./dawdawdadwadaw.html")
