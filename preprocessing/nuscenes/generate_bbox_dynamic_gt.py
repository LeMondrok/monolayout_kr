import argparse
import os

from matplotlib import pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import transform_matrix, Quaternion

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


def abs2ego(ego_pose, cam_pose, annotation_data):
    occ_width, occ_height = 256, 256
    scale = 4
    
    ego2cam = transform_matrix(cam_pose['translation'], Quaternion(cam_pose['rotation']), inverse=True)
    world2ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)
    world2cam = np.dot(ego2cam, world2ego)
    
    obj2world = transform_matrix(annotation_data['translation'], Quaternion(annotation_data['rotation']), inverse=False)
    obj2cam = np.dot(world2cam, obj2world)
    
    vehicle_first_edge = annotation_data['size'].copy()
    vehicle_first_edge = [x / 2 for x in vehicle_first_edge]
    vehicle_first_edge[1], vehicle_first_edge[0] = vehicle_first_edge[0], vehicle_first_edge[1]
    
    vehicle_second_edge = vehicle_first_edge.copy()
    vehicle_second_edge[1] *= -1
    vehicle_third_edge = [-x for x in vehicle_first_edge]
    vehicle_fourth_edge = [-x for x in vehicle_second_edge]

    first_edge = np.dot(obj2cam, vehicle_first_edge + [1])    
    second_edge = np.dot(obj2cam, vehicle_second_edge + [1])
    third_edge = np.dot(obj2cam, vehicle_third_edge + [1])
    fourth_edge = np.dot(obj2cam, vehicle_fourth_edge + [1])    
        
    polygons = []
    polygons.append(first_edge)
    polygons.append(second_edge)
    polygons.append(third_edge)
    polygons.append(fourth_edge)
    
    polygons = np.array(polygons)

    rr, cc = polygon(
        occ_width - scale * polygons[:, 2], occ_width / 2 + scale * polygons[:, 0], (occ_width, occ_height)
    )
    ans = np.zeros((occ_width, occ_height), dtype='uint8')
    ans[rr, cc] = 255

    return ans

def get_fov_mask(cam_data):
    occ_width, occ_height = 256, 256

    sensor_data = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

    filename = cam_front_data['filename']
    img = Image.open(f'{data_root}{filename}')

    f_x = sensor_data['camera_intrinsic'][0][0]
    w = img.size[0]

    fov_x = 2 * np.arctan(w / (2 * f_x))
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

classes = [
    'car',
    'truck',
    'bus',
]

if __name__ == "__main__":
    args = get_args()

    data_root = args.nusc_path
    nusc = NuScenes(version=args.nusc_version, dataroot=data_root, verbose=True)

    occ_width, occ_height = 256, 256
    
    for cls_ in classes:
        sensor = 'CAM_FRONT'

        output_dir = f'{data_root}/dynamic_gt/{cls_}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fov_mask = None

        for sample in nusc.sample:
            my_sample = nusc.get('sample', sample['token'])
            new_pic = np.zeros((occ_width, occ_height), dtype=np.uint8)
            new_pic.fill(0)

            cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
            ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            cam_pose = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

            for annotation in my_sample['anns']:
                annotation_metadata = nusc.get('sample_annotation', annotation)
#                 if int(annotation_metadata['visibility_token']) > 2 and annotation_metadata['category_name'].split('.')[0] == 'vehicle':
                if int(annotation_metadata['visibility_token']) > 2 and len(annotation_metadata['category_name'].split('.')) > 1 and annotation_metadata['category_name'].split('.')[1] == cls_:
                    new_pic += abs2ego(ego_pose, cam_pose, annotation_metadata)

            if fov_mask is None:
                fov_mask = get_fov_mask(cam_front_data)

            img = Image.fromarray(fov_mask * new_pic)
            img.save(os.path.join(output_dir, cam_front_data['filename'].split('/')[-1]))