import mmcv
import os
import glob
import cv2
import pandas as pd
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, show_result_yu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cfg = mmcv.Config.fromfile('/home/yu/mmdetection/my_config/cascade_rcnn_r101_wheat.py')
cfg.model.pretrained = None

save_path = './infrenced_img'
out_file = os.path.join(save_path, 'test.jpg')
image_path = './data/coco-wheat/test/000288.jpg'
# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './data/result/cascade_rcnn_r101_wheat/epoch_940.pth')

# 测试单张图片
# 取图片ROI
# img = cv2.imread(image_path)
# img = img[800:, :]
# 读取
img = mmcv.imread(image_path)

# infrence
print(img.shape)
result = inference_detector(model, img, cfg)
bboxes = np.vstack(result)

print(bboxes[0])

txt_file = open('cascade.txt', 'w')
for i in range(len(bboxes)):
    score = bboxes[i][-1]
    y_min = bboxes[i][0]
    x_min = bboxes[i][1]
    y_max = bboxes[i][2]
    x_max = bboxes[i][3]
    txt_file.write(str(score) + "\t")
    txt_file.write(str(y_min) + "\t")
    txt_file.write(str(x_min) + "\t")
    txt_file.write(str(y_max) + "\t")
    txt_file.write(str(x_max))
    txt_file.write('\n')

show_result_yu(img, result, out_file=out_file)
