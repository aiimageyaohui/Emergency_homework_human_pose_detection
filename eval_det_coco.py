"""

this file will run a model
save the detection result on evaluation dataset
then load them with GT to evaluation

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
import glob
import json



arch = 'res_101'  # res_18, res_101, hourglass
heads = {'hm': 4, 'reg': 2, 'wh': 2}
head_conv = 64  # 64 for resnets
# model_path = './weights/ctdet_pascal_trafficlight_last.pth'
model_path = './weights/ctdet_pascal_trafficlight_120.pth'
mean = [0.408, 0.447, 0.470]  # coco and kitti not same
std = [0.289, 0.274, 0.278]
classes_names = ['tlt_red', 'tlt_green', 
                 'tlt_yellow', 'tlt_black']
cls_colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (96, 96, 96)]
num_classes = len(classes_names)
test_scales = [1]
pad = 31  # hourglass not same
input_shape = (512, 512)
# input_shape = None  # None for original input
down_ratio = 4
K_outputs = 100


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
        res = visualize_det_cv2_style0(image, detections[0], classes_names, cls_colors=cls_colors, thresh=0.3, suit_color=True)
        cv2.putText(res, 'fps: {0:.4f}'.format(1/cost), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
         (0, 0, 255), 2)
        return res
    
    def run_on_img_file(self, image_or_path_or_tensor, meta=None):
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        detections = []
        cost = 0
        for scale in self.scales:
            images, meta = self.pre_process(image, scale, meta)
            images = images.to(device)
            torch.cuda.synchronize()
            tic = time.time()
            _, dets = self.process(images, return_time=True)
            cost = time.time() - tic
            print('cost: {}, fps: {}'.format(cost, 1 / cost))
            torch.cuda.synchronize()
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            detections.append(dets)
        res = visualize_det_cv2_style0(image, detections[0], classes_names, cls_colors=cls_colors, thresh=0.3, suit_color=True)
        return detections[0]


def eval_folder(data_f, inferencer, categories=None):
    """
    inferencer is a model or Class
    """
    all_imgs = glob.glob(os.path.join(data_f, '*.jpg'))
    logging.info('eval on all {} images.'.format(len(all_imgs)))
    json_dict = {"images":[], "type": "instances", "annotations": [], "categories": []}
    i = 0
    for line in all_imgs:
        print("Processing %s"%(line))
        img_f = line
        filename = os.path.basename(img_f)
        image_id = i
        height, width = cv2.imread(img_f).shape[:-1]
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)

        # inference on image get dets
        dets = inferencer.run_on_img_file(img_f)
        # print(dets)
        bnd_id = 0
        for det in dets:
            category_id = int(det[0])
            xmin = round(float(max(det[2], 0)), 2)
            ymin = round(float(max(det[3], 0)), 2)
            xmax = round(float(det[4]), 2)
            ymax = round(float(det[5]), 2)
            # print('{}, {}, {}, {}'.format(xmin, ymin, xmax, ymax))
            # assert(xmax > xmin)
            # assert(ymax > ymin)
            o_width = round(float(abs(xmax - xmin)), 2)
            o_height = round(float(abs(ymax - ymin)), 2)
            ann = {'area': round(o_width*o_height, 2), 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        # image_id plus 1
        i += 1

        # if i == 10:
            # break

    if categories:
        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
    json_file = 'eval_result_coco.json'
    logging.info('converted coco format will saved into: {}'.format(json_file))
    json_fp = open(json_file, 'w')
    # print(json_dict)
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    logging.info('done.')



def eval_on_json(json_f, inferencer, img_root=None, categories=None):
    """
    Using json_f (eval annotations) to get images, and generates
    all inferenced results
    """
    from pycocotools.coco import COCO
    coco = COCO(json_f)
    # get all image ids
    all_img_ids = coco.getImgIds()

    logging.info('eval on all {} images.'.format(len(all_img_ids)))
    json_dict = {"images":[], "type": "instances", "annotations": [], "categories": []}
    i = 0
    for img_id in all_img_ids:
        # get image filename by img_id
        one_img = coco.loadImgs([img_id])[0]
        filename = one_img['file_name']
        img_f = os.path.join(img_root, filename)
        print("Processing %s"%(img_f))
        image_id = i
        height, width = cv2.imread(img_f).shape[:-1]
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)

        # inference on image get dets
        dets = inferencer.run_on_img_file(img_f)
        # print(dets)
        bnd_id = 0
        for det in dets:
            category_id = int(det[0])
            xmin = round(float(max(det[2], 0)), 2)
            ymin = round(float(max(det[3], 0)), 2)
            xmax = round(float(det[4]), 2)
            ymax = round(float(det[5]), 2)
            # print('{}, {}, {}, {}'.format(xmin, ymin, xmax, ymax))
            # assert(xmax > xmin)
            # assert(ymax > ymin)
            o_width = round(float(abs(xmax - xmin)), 2)
            o_height = round(float(abs(ymax - ymin)), 2)
            ann = {'area': round(o_width*o_height, 2), 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': [], 'score': float(det[1])}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        # image_id plus 1
        i += 1

        # if i == 10:
            # break

    if categories:
        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
    json_file = 'eval_result_coco.json'
    logging.info('converted coco format will saved into: {}'.format(json_file))
    json_fp = open(json_file, 'w')
    # print(json_dict)
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    logging.info('done. generated {} images detections.'.format(i))
    return json_dict


if __name__ == '__main__':
    # using the model inference on eval dataset


    detector = CenterNetDetector()
    data_f = 'images/33887522274_eebd074106_k.jpg'
    if len(sys.argv) > 1:
        data_f = sys.argv[1]
    if 'mp4' in os.path.basename(data_f):
        cam = cv2.VideoCapture(data_f)
        while True:
            _, img = cam.read()
            res = detector.run(img)
            cv2.imshow('centernet_video', res)
            cv2.waitKey(1)
    elif os.path.isdir(data_f):
        # inference on folder to get final result
        eval_folder(data_f, detector)
    elif '.json' in data_f:
        img_root_folder = os.path.join(os.path.dirname(data_f), 'JPEGImages')
        logging.info('inference on image root folder: {}'.format(img_root_folder))
        logging.info('with json file: {}'.format(data_f))
        coco_results = eval_on_json(data_f, detector, img_root=img_root_folder)
        # print(coco_results)

        # load json and inferenced, calculated mAP
        from pycocotools.cocoeval import COCOeval
        from pycocotools.coco import COCO
        coco_gt = COCO(data_f)
        coco_dt = coco_gt.loadRes(coco_results['annotations'])
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    else:
        res = detector.run(data_f)
        cv2.imshow('res', res)
        cv2.waitKey(0)