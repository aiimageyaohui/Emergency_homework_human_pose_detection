from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import torch.utils.data as data
from alfred.utils.log import logger as logging

"""


custom  pscal VOC dataset training

etc:
['trafficlight_black', 'trafficlight_red', 'trafficlight_green', 'trafficlight_yellow']

"""


class PascalVOCCustom(data.Dataset):
  num_classes = 4
  default_resolution = [512, 512]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(PascalVOCCustom, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, opt.dataset)
    self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
    _ann_name = {'train': 'trainval0712', 'val': 'test2007'}
    self.annot_path = os.path.join(self.data_dir, 'coco_trafficlight_{}.json'.format(split))
    self.max_objs = 50
    self.class_name = ['__background__', 'trafficlight_black', 'trafficlight_red', 'trafficlight_green', 'trafficlight_yellow']
    self._valid_ids = np.arange(1, 5, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    logging.info('==> initializing pascal {} data.'.format(self.annot_path))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)
    logging.info('Loaded {} samples'.format(self.num_samples))
    logging.info('image root dir: {}'.format(self.img_dir))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      for j in range(1, self.num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    os.system('python tools/reval.py ' + \
              '{}/results.json'.format(save_dir))