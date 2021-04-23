import argparse
import os

from matplotlib import pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import transform_matrix, Quaternion

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

import numpy as np

from PIL import Image
from skimage.draw import polygon


def get_args():
    parser = argparse.ArgumentParser(description="nuScenes")
    parser.add_argument("--nusc_path", type=str, required=True,
                        help="Path to the root of nuScenes dataset")
    parser.add_argument("--nusc_version", type=str, default="v1.0-mini",
                        help="version of nuscenes dataset")

    return parser.parse_args()


def get_mask_angle(ego_pose, cam_pose):
    cam2ego = transform_matrix(cam_pose['translation'], Quaternion(cam_pose['rotation']), inverse=False)
    ego2world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
    cam2world = np.dot(ego2world, cam2ego)

    rel_unit = np.dot(cam2world, [0, 0, 1, 0])
    angle = 180 - 180 * np.arctan2(rel_unit[0], rel_unit[1]) / np.pi

    return angle

def get_fov_mask(cam_data):
    occ_width, occ_height = 256, 256

    sensor_data = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

    filename = cam_front_data['filename']
    img = Image.open(f'{data_root}{filename}')

    f_x = sensor_data['camera_intrinsic'][0][0]
    h = img.size[0]

    fov_x = 2 * np.arctan(h / (2 * f_x))
    alpha = (np.pi - fov_x) / 2

    polygons = [
        [0, occ_height - occ_width * np.tan(alpha) / 2],
        [occ_width / 2, occ_height],
        [occ_width, occ_height - occ_width * np.tan(alpha) / 2],
        [occ_width, occ_height],
        [0, occ_height]
    ]
    
    polygons = np.array(polygons)

    rr, cc = polygon(polygons[:, 1], polygons[:, 0], (occ_width, occ_height))
    ans = np.ones((occ_width, occ_height), dtype='uint8')
    ans[rr, cc] = 0

    return ans


if __name__ == "__main__":
    args = get_args()

    data_root = args.nusc_path
    nusc = NuScenes(version=args.nusc_version, dataroot=data_root, verbose=True)

    occ_width, occ_height = 256, 256
    sensor = 'CAM_FRONT'

    output_dir = f'{data_root}/static_gt'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    old_location = None
    fov_mask = None

    for sample in nusc.sample:
        my_sample = nusc.get('sample', sample['token'])

        cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
        ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        cam_pose = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        scene = nusc.get('scene', my_sample['scene_token'])
        log = nusc.get('log', scene['log_token'])

        if old_location != log['location']:
            nusc_map = NuScenesMap(dataroot=data_root, map_name=log['location'])
            old_location = log['location']

        # in meters
        scale = 32

        patch_box = [
            ego_pose['translation'][0],
            ego_pose['translation'][1],
            2 * scale,
            2 * scale
        ]
        patch_angle = get_mask_angle(ego_pose, cam_pose)
        layer_names = ['drivable_area']

        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, (2 * occ_width, 2 * occ_height))[0]
        map_mask = map_mask[:occ_height, int(occ_width / 2): -int((occ_width + 1) / 2)]
        map_mask = np.flip(map_mask, 1)

        if fov_mask is None:
            fov_mask = get_fov_mask(cam_front_data)

        img = Image.fromarray(fov_mask * map_mask * 255)
        img.save(os.path.join(output_dir, cam_front_data['filename'].split('/')[-1]))
