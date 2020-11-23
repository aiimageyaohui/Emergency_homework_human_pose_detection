# CenterNet

this is a version support pytorch 1.4+ and tested on ubuntu20.04, original implementation is [here](https://github.com/xingyizhou/CenterNet).
**centernet** is the very first method that combines all computer vision task togather within one single solution. it can do almost all computer vision task at the same time,here I mainly use it to do human key points detection as my emergency homework.
![](images/1.png)

## Abstract 

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point -- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## Highlights

- **Simple:** One-sentence method summary: use keypoint detection technic to detect the bounding box center point and regress to all other object properties like bounding box size, 3d information, and pose.

- **Versatile:** The same framework works for object detection, 3d bounding box estimation, and multi-person pose estimation with minor modification.

- **Fast:** The whole process in a single network feedforward. No NMS post processing is needed. Our DLA-34 model runs at *52* FPS with *37.4* COCO AP.

- **Strong**: Our best single model achieves *45.1*AP on COCO test-dev.

- **Easy to use:** We provide user friendly testing API and webcam demos.

## Main results

### Object Detection on COCO validation

| Backbone     |  AP / FPS | Flip AP / FPS|  Multi-scale AP / FPS |
|--------------|-----------|--------------|-----------------------|
|Hourglass-104 | 40.3 / 14 | 42.2 / 7.8   | 45.1 / 1.4            |
|DLA-34        | 37.4 / 52 | 39.2 / 28    | 41.7 / 4              |
|ResNet-101    | 34.6 / 45 | 36.2 / 25    | 39.3 / 4              |
|ResNet-18     | 28.1 / 142| 30.0 / 71    | 33.2 / 12             |

### Keypoint detection on COCO validation

| Backbone     |  AP       |  FPS         |
|--------------|-----------|--------------|
|Hourglass-104 | 64.0      |    6.6       |
|DLA-34        | 58.9      |    23        |

### 3D bounding box detection on KITTI validation

|Backbone|FPS|AP-E|AP-M|AP-H|AOS-E|AOS-M|AOS-H|BEV-E|BEV-M|BEV-H| 
|--------|---|----|----|----|-----|-----|-----|-----|-----|-----|
|DLA-34  |32 |96.9|87.8|79.2|93.9 |84.3 |75.7 |34.0 |30.5 |26.8 |


# Installation
0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~

1. Install pytorch1.4:

    ~~~
   	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ~~~
    
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional

    ~~~
    cd $CenterNet_ROOT/models/networks/dcnv2
    ./make.sh
    ~~~

6. Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/external
    make
    ~~~

7. Download pertained models for [pose estimation]() and move them to `$CenterNet_ROOT/`. More models can be found in [pertained models for pose estimation](链接：https:/pan.baidu.com/s/1ip5Dd8zw8euekz0kS2kh5g 
提取码：b8li 
).

The code was tested on Ubuntu 20.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.4. NVIDIA GPUs are needed for both training and testing.

## Use CenterNet

We support demo for image/ image folder, video, and webcam.

for human pose estimation, run:

~~~
python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth
~~~
The result for the example images should look like:
<p align="center">  <img src='images/pose1.png' align="center" height="200px"> <img src='images/pose2.png' align="center" height="200px"> <img src='images/pose3.png' align="center" height="200px">  </p>

you can run:
~~~
python demo.py multi_pose --demo Pulp_Fiction.mp4 --load_model multi_pose_dla_3x.pth
~~~
to get this result:
![](output/pulp_fiction.gif)
(pulp fiction is one of my favourite movie)
## 

