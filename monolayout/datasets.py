
from __future__ import absolute_import, division, print_function

import os
import random

import PIL.Image as pil

import numpy as np

import torch.utils.data as data

import torch

from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def process_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    topview_n = np.zeros(topview.shape)
    topview_n[topview == 255] = 1  # [1.,0.]
    return topview_n


def resize_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    return topview


def process_discr(topview, size):
    topviews_n = []
    for view_ind in range(len(topview)):
        temp_topview = resize_topview(topview[view_ind], size)
        topview_n = np.zeros((size, size, 2))
        topview_n[temp_topview == 255, 1] = 1.
        topview_n[temp_topview == 0, 0] = 1.
        
        topviews_n.append(topview_n)
    return topviews_n


class MonoDataset(data.Dataset):
    def __init__(self, opt, filenames, use_osm=True, is_train=True):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.data_path = self.opt.data_path
        self.filenames = filenames
        self.is_train = is_train
        self.height = self.opt.height
        self.width = self.opt.width
        self.interp = pil.ANTIALIAS
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.use_osm = use_osm

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):

        inputs["color"] = color_aug(self.resize(inputs["color"]))

        for key in inputs.keys():
            if key != "color" and "discr" not in key:
                for img_ind in range(len(inputs[key])):
                    inputs[key][img_ind] = process_topview(
                        inputs[key][img_ind], self.opt.occ_map_size)
                
            if key != "color":
                for img_ind in range(len(inputs[key])):
                    inputs[key][img_ind] = self.to_tensor(inputs[key][img_ind])
                    
                inputs[key] = torch.cat(inputs[key])
                
                if "discr" in key:
                    inputs[key] = inputs[key].reshape(2, inputs[key].shape[0] // 2, inputs[key].shape[1], inputs[key].shape[2])
            else:
                inputs[key] = self.to_tensor(np.array(inputs[key]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path

        inputs["color"] = self.get_color(folder, frame_index, do_flip)
        if self.opt.type == "static":
            if self.is_train:
                inputs["static"] = self.get_static(
                    folder, frame_index, do_flip)
                if self.use_osm:
                    inputs["discr"] = process_discr(
                        self.get_osm(
                            self.opt.osm_path,
                            do_flip),
                        self.opt.occ_map_size)
                else:
                    inputs["discr"] = process_discr(
                        inputs["static"], self.opt.occ_map_size)
                    
            else:
                inputs["static_gt"] = self.get_static_gt(
                    folder, frame_index, do_flip)
            # inputs["osm"] = self.get_osm(folder, frame_index, do_flip)
        elif self.opt.type == "dynamic":
            if self.is_train:
                inputs["dynamic"] = self.get_dynamic(
                    folder, frame_index, do_flip)
                inputs["discr"] = process_discr(
                    inputs["dynamic"], self.opt.occ_map_size)
            else:
                inputs["dynamic_gt"] = self.get_dynamic_gt(
                    folder, frame_index, do_flip)
        else:
            if self.is_train:
                inputs["static"] = self.get_static(
                    folder, frame_index, do_flip)
                inputs["dynamic"] = self.get_dynamic(
                    folder, frame_index, do_flip)
                if self.use_osm:
                    inputs["static_discr"] = process_discr(self.get_osm(
                        self.opt.osm_path, do_flip), self.opt.occ_map_size)
                else:
                    inputs["static_discr"] = process_discr(
                        inputs["static"], self.opt.occ_map_size)
                inputs["dynamic_discr"] = process_discr(
                    inputs["dynamic"], self.opt.occ_map_size)
            else:
                inputs["dynamic_gt"] = self.get_dynamic_gt(
                    folder, frame_index, do_flip)
                inputs["static_gt"] = self.get_dynamic_gt(
                    folder, frame_index, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_static(self, folder, frame_index, do_flip):
        path = self.get_static_path(folder, frame_index)
        
        if not isinstance(path, list):
            path = [path]
                
        tvs = []
        for p in path:
            tv = self.loader(p)

            if do_flip:
                tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        
            tv = tv.convert('L')
            tvs.append(tv)
        
        return tvs

    def get_dynamic(self, folder, frame_index, do_flip):
        path = self.get_dynamic_path(folder, frame_index)
        
        if not isinstance(path, list):
            path = [path]
                
        tvs = []
        for p in path:
            tv = self.loader(p)

            if do_flip:
                tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        
            tv = tv.convert('L')
            tvs.append(tv)
        
        return tvs

    def get_osm(self, root_dir, do_flip):
        osm = self.loader(self.get_osm_path(root_dir))
        return osm

    def get_static_gt(self, folder, frame_index, do_flip):        
        path = self.get_static_gt_path(folder, frame_index)
        
        if not isinstance(path, list):
            path = [path]
                
        tvs = []
        for p in path:
            tv = self.loader(p)        
            tv = tv.convert('L')
            tvs.append(tv)
        
        return tvs

    def get_dynamic_gt(self, folder, frame_index, do_flip):
        path = self.get_dynamic_gt_path(folder, frame_index)
        
        if not isinstance(path, list):
            path = [path]
                
        tvs = []
        for p in path:
            tv = self.loader(p)        
            tv = tv.convert('L')
            tvs.append(tv)
        
        return tvs


class KITTIObject(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIObject, self).__init__(*args, **kwargs)
        self.root_dir = "./data/object"

    def get_image_path(self, root_dir, frame_index):
        image_dir = os.path.join(root_dir, 'image_2')
        img_path = os.path.join(image_dir, "%06d.png" % int(frame_index))
        return img_path

    def get_dynamic_path(self, root_dir, frame_index):
        tv_dir = os.path.join(root_dir, 'vehicle_256')
        tv_path = os.path.join(tv_dir, "%06d.png" % int(frame_index))
        return tv_path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)

    def get_static_gt_path(self, root_dir, frame_index):
        pass


class KITTIOdometry(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIOdometry, self).__init__(*args, **kwargs)
        self.root_dir = "./data/odometry/sequences/"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace("road_dense128", "image_2")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        return self.get_static_path(root_dir, frame_index)

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass


class KITTIRAW(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIRAW, self).__init__(*args, **kwargs)
        self.root_dir = "./data/raw/"

    def get_image_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, frame_index)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_256"))
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_bev_gt"))
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass


class KITTIRAWGT(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIRAWGT, self).__init__(*args, **kwargs)
        self.root_dir = "./data/raw/"

    def get_image_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, frame_index)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_bev_gt"))
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_bev_gt"))
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass


class Argoverse(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Argoverse, self).__init__(*args, **kwargs)
        self.root_dir = "./data/argo"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "stereo_front_left").replace(
            "png", "jpg")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_dynamic_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "car_bev_gt").replace(
            "png", "jpg")
        path = os.path.join(root_dir, file_name)
        return path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir,
            frame_index).replace(
            "road_bev",
            "road_gt")
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(self, root_dir, frame_index)

class nuScenesFront(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(nuScenesFront, self).__init__(*args, use_osm=False, **kwargs) 
        self.root_dir = "./data/nuscenes"
        self.static_classes = [
            'drivable_area',
            'ped_crossing',
            'walkway'
        ]
        self.dynamic_classes = [
            'car',
            'truck',
            'bus'
        ]

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "static_gt", "samples/CAM_FRONT")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        ans = []
        
        path = os.path.join(root_dir, frame_index)
            
        for cls_ in self.static_classes:
            ans.append(
                path.replace(
                    "static_gt",
                    os.path.join("static_gt", cls_)
                )
            )
            
        return ans

    def get_dynamic_path(self, root_dir, frame_index):
        ans = []
        
        path = os.path.join(root_dir, frame_index)
            
        for cls_ in self.dynamic_classes:
            ans.append(
                path.replace(
                    "static_gt",
                    os.path.join("dynamic_gt", cls_)
                )
            )
            
        return ans

    def get_static_gt_path(self, root_dir, frame_index):
        return self.get_static_path(root_dir, frame_index)

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)
