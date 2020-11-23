"""

inference on CenterNet101 on a coco-custom datasets
trained model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
from alfred.utils.log import logger as logging
from alfred.dl.torch.common import device
import torch
from models.model import create_model, load_model
from utils.image import get_affine_transform
from models.decode import ctdet_decode
from utils.post_process import ctdet_post_process2, ctdet_post_process
from external.nms import soft_nms
from alfred.vis.image.det import visualize_det_cv2, visualize_det_cv2_style0
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
import time


arch = 'res_101'  # res_18, res_101, hourglass
heads = {'hm': 3, 'reg': 2, 'wh': 2}
head_conv = 64  # 64 for resnets
model_path = './exp/ctdet/coco_helmet_r101/model_last.pth'
mean = [0.408, 0.447, 0.470]  # coco and kitti not same
std = [0.289, 0.274, 0.278]
classes_names = ['hat', 'person', 'dog']
cls_colors = [(255, 255, 0), (0, 0, 255), (0, 255, 255)]
num_classes = len(classes_names)
test_scales = [1]
pad = 31  # hourglass not same
input_shape = (512, 512)
# input_shape = None  # None for original input
down_ratio = 4
K_outputs = 100

crop_area = [(490, 265), (1400, 800)]


class CenterNetDetector(object):
    def __init__(self):
        logging.info('Creating model...')
        self.model = create_model(arch, heads, head_conv)
        if os.path.exists(model_path):
            self.model = load_model(self.model, model_path)
        else:
            logging.info("skip load model since can not found model file.")
        self.model = self.model.to(device)
        self.model.eval()
        logging.info('model loaded.')

        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = num_classes
        self.scales = test_scales
        self.pad = pad
        self.mean = mean
        self.std = std
        self.down_ratio = down_ratio
        self.input_shape = input_shape
        self.K = K_outputs
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.input_shape != None:
            inp_height, inp_width = self.input_shape
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.pad) + 1
            inp_width = (new_width | self.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) /
                     self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(
            1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            torch.cuda.synchronize()
            dets = ctdet_decode(hm, wh, reg=reg, K=self.K)
        return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process2(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        return dets

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run(self, image_or_path_or_tensor, meta=None):
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        detections = []
        cost = 0

        all_dets = 0
        hats = 0
        for scale in self.scales:
            images, meta = self.pre_process(image, scale, meta)
            images = images.to(device)
            # print('input shape: {}'.format(images.shape))
            torch.cuda.synchronize()
            tic = time.time()
            _, dets = self.process(images, return_time=True)
            cost = time.time() - tic
            print('cost: {}, fps: {}'.format(cost, 1 / cost))
            torch.cuda.synchronize()
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            detections.append(dets)
        res = visualize_det_cv2_style0(
            image, detections[0], classes_names, cls_colors=cls_colors, suit_color=True, thresh=0.5, line_thickness=1, font_scale=0.4, counter_on=True, counter_pos=(30, 220))
        cv2.putText(res, 'Runtime: {:.4f}s  FPS: {:.4f}'.format(cost, 1/cost), (30, 160), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                    (0, 0, 255), 2)
        return res


if __name__ == '__main__':
    detector = CenterNetDetector()
    data_f = 'images/33887522274_eebd074106_k.jpg'
    if len(sys.argv) > 1:
        data_f = sys.argv[1]
    if 'mp4' in os.path.basename(data_f):
        cam = cv2.VideoCapture(data_f)
        # shall we save video to local?
        print('video size: {}x{}'.format(cam.get(3), cam.get(4)))
        target_f = os.path.basename(data_f)
        video_writer = cv2.VideoWriter(target_f, cv2.VideoWriter_fourcc(*'DIVX'),
                                       24, (int(cam.get(3)), int(cam.get(4))))
        print(target_f)
        while True:
            _, img = cam.read()
            if img is not None:
                # crop img to predict
                in_img = img
                if crop_area:
                    in_img = img[crop_area[0][1]: crop_area[1][1], crop_area[0][0]: crop_area[1][0]]
                    print(in_img.shape)
                res = detector.run(in_img)
                img[crop_area[0][1]: crop_area[1][1], crop_area[0][0]: crop_area[1][0]] = res
                # res = cv2.resize(res, None, fx=0.5, fy=0.5)
                cv2.rectangle(img, crop_area[0], crop_area[1], (0, 0, 255), 2)
                cv2.imshow('centernet_video', img)
                cv2.waitKey(1)
                video_writer.write(img)
            else:
                video_writer.release()
                break
        print('video saved to: {}'.format(target_f))
    else:
        res = detector.run(data_f)
        cv2.imshow('res', res)
        cv2.waitKey(0)
