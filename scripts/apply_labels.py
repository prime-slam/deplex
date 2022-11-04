#!/bin/bash

import sys
import os
import open3d as o3d
import numpy as np

from readers.tum_reader import read_depth


def apply_labels(pcd_path: str, labels_path: str):
    labels_table = np.genfromtxt(labels_path, delimiter=",").astype(np.uint8)
    labels = labels_table.reshape(labels_table.size)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    colors = np.array(pcd.colors)
    label_to_color = {0: np.zeros(shape=(1, 3))}
    color_set = {(0.0, 0.0, 0.0)}

    for label in np.unique(labels):
        if label not in label_to_color:
            col = np.random.uniform(0, 1, size=(1, 3))
            while tuple(col[0]) in color_set:
                col = np.random.uniform(0, 1, size=(1, 3))
            label_to_color[label] = col
            color_set.add(tuple(col[0]))
        colors[labels == label] = label_to_color[label]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def main():
    image_path, labels_path = sys.argv[1:]
    pcd = read_depth(image_path)
    tmp_pcd_path = image_path + "__GEN_PCD.pcd"
    o3d.io.write_point_cloud(tmp_pcd_path, pcd)

    try:
        apply_labels(tmp_pcd_path, labels_path)
    finally:
        if os.path.exists(tmp_pcd_path):
            os.remove(tmp_pcd_path)


if __name__ == '__main__':
    main()
