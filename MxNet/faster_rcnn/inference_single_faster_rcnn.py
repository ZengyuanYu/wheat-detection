import os
import cv2
import pandas as pd
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

# MXNET_CUDNN_AUTOTUNE_DEFAULT = 0

# 自己数据集的类别
CLASSES = ['wheat']

# 模型和参数，置信度
NetWork = 'faster_rcnn_resnet101_v1d_custom'
trained_model = '/home/yu/MxNet/model/fasterrcnn/faster_rcnn_resnet101_v1d_custom_best.params'
image_path = '../VOCtemplate/VOC-wheat/JPEGImages/000288.jpg'
Confidence_Thresh = 0.8

if __name__ == '__main__':
    ctx = [mx.gpu(1)]

    # 载入模型
    net = gcv.model_zoo.get_model(NetWork, classes=CLASSES, pretrained=False, pretrained_base=False)
    net.load_parameters(trained_model)
    net.set_nms(0.3, 200)
    net.collect_params().reset_ctx(ctx=ctx)

    # 载入图片
    ax = None
    x, img = presets.rcnn.load_test(image_path, short=net.short, max_size=net.max_size)
    # x, img = presets.rcnn.load_test(image, short=3024, max_size=4032)
    print(img.shape)
    print(net.short, net.max_size, net.classes)

    # 送入GPU并inference
    x = x.as_in_context(ctx[0])
    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]

    txt_file = open('./faster_rcnn_bboxes.txt', 'w')
    for i in range(len(bboxes)):
        if scores[i] >= Confidence_Thresh:
            # print(scores[i])
            score = float(scores[i])
            # print(score)
            y_min = bboxes[i][0]
            x_min = bboxes[i][1]
            y_max = bboxes[i][2]
            x_max = bboxes[i][3]
            # print(y_min, y_max, x_min, x_max)
            txt_file.write(str(score) + "\t")
            txt_file.write(str(y_min) + "\t")
            txt_file.write(str(x_min) + "\t")
            txt_file.write(str(y_max) + "\t")
            txt_file.write(str(x_max))
            txt_file.write('\n')

    txt_file.close()
