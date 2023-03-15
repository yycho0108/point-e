#!/usr/bin/env python3

import pyrender
# import open3d as o3d


# def custom_draw_geometry_with_rotation(pcd):
#     def rotate_view(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(10.0, 0.0)
#         return False
#     o3d.visualization.draw_geometries_with_animation_callback(
#         [pcd], rotate_view)


def main():
    pcd = o3d.io.read_point_cloud('/tmp/out/cloud-00.ply')
    custom_draw_geometry_with_rotation(pcd)
    pass
from pyrender import M
