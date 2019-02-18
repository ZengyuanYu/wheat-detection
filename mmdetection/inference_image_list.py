import mmcv
import os
import numpy as np
import glob
import cv2
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, inference_detector_yu, show_result, show_result_yu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

save_path = './infrenced_img'


# construct the model and load checkpoint
cfg = mmcv.Config.fromfile('/home/yu/mmdetection/my_config/cascade_rcnn_r101_wheat.py')
cfg.model.pretrained = None
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './data/result/cascade_rcnn_r101_wheat-little/epoch_940.pth')

# test a list of images
imgs = sorted(glob.glob('./data/coco-wheat/test/*.jpg'))

for i, result in enumerate(inference_detector(model, imgs[:2], cfg, device='cuda:0')):
    print(i, imgs[i])
    save_name = imgs[i].split('/')[-1]
    print(save_name)
    out_file = os.path.join(save_path, save_name)
    print(out_file)

    #存储bboxes坐标
    bboxes = np.vstack(result)
    txt_file = open(os.path.join(save_path, save_name.split('.')[0] + ".txt"), 'w')
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
    txt_file.close()
    show_result_yu(imgs[i], result, show=False, out_file=out_file)