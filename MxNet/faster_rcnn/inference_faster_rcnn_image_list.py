"""Faster RCNN Demo script."""
import os
import cv2
import time
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

# 测试脚本
#  python inference_fater_rcnn_image_list.py --network faster_rcnn_resnet101_v1d_custom --model /home/yu/MxNet/model/fasterrcnn/faster_rcnn_resnet101_v1d_custom_epoch_300/faster_rcnn_resnet101_v1d_custom_best.params --save-path ../result/


parser = argparse.ArgumentParser(description='Test Bbox model acc and display save it')
parser.add_argument('--network', type=str, default='faster_rcnn_resnet101_v1d_custom',
                    help="网络结构")
parser.add_argument('--model', type=str, default='/home/yu/MxNet/model/fasterrcnn/faster_rcnn_resnet101_v1d_custom_best.params',
                    help='已经训练好的model')
parser.add_argument('--img-path', type=str, default='../VOCtemplate/VOC-wheat/', help='VOC格式测试图片的位置')
parser.add_argument('--save-path',  type=str, default='./', help='存储图片的位置')
args = parser.parse_args()

# args = parser.parse_args()
# 自己数据集的类别
# CLASSES = ['plate', 'light', 'car', 'window']
CLASSES = ['wheat']
# root_dir = '../VOCtemplate/VOC2018/'
root_dir = args.img_path
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
# 模型和参数
NetWork = args.network
trained_model = args.model
save_path = args.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)
all_list = []
test_image_txt = '../VOCtemplate/VOC-wheat/ImageSets/Main/test.txt'
f = open(test_image_txt)
for line in f:
    all_list.append('{}'.format(line.replace('\n', '')))
Confidence_Thresh = 0.8


def count_xml_bbox_num(xml_path):
    """
    :param xml_path: xml的路径
    :return: 返回本xml的Bbox数目
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    return len(root.findall('object'))
    # print("当前读入图片的bbox个数为：", ))


if __name__ == '__main__':


    begin_time = time.time()
    ctx = [mx.gpu(1)]
    # ctx = [mx.cpu(0)]

    # 载入模型
    net = gcv.model_zoo.get_model(NetWork, classes=CLASSES, pretrained=False, pretrained_base=False)
    net.load_parameters(trained_model)
    net.set_nms(0.3, 200)
    net.collect_params().reset_ctx(ctx=ctx)

    # 载入图片
    ax=None
    GT_bbox_count_list = []
    Predict_bbox_count_list = []
    img_count = 0
    for file in all_list[:]:
        image_path = os.path.join(img_dir, file + '.jpg')
        xml_path = os.path.join(ann_dir, file + '.xml')
        # 计算读取的GT bbox数目
        GT_bbox_count = count_xml_bbox_num(xml_path)
        print(image_path, xml_path)
        img_count += 1

        x, img = presets.rcnn.load_test(image_path, short=net.short, max_size=net.max_size)

        # 送入GPU/CPU并inference
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]

        # 可视化操作
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=Confidence_Thresh,
                                         class_names=net.classes, ax=ax)
        Predict_bbox_count = gcv.utils.viz.bbox.draw_save_bbox_yu(image_path, img, bboxes,
                                                     save_path,True,False, scores, ids,
                                             thresh=Confidence_Thresh,
                                             class_names=net.classes)
        GT_bbox_count_list.append(GT_bbox_count)
        Predict_bbox_count_list.append(Predict_bbox_count)
        print(GT_bbox_count, Predict_bbox_count)
        print("GTbbox数目为：{}，预测个数为：{}, 单张预测误差率为：{}%".\
                    format(GT_bbox_count, Predict_bbox_count,
                           (GT_bbox_count-Predict_bbox_count)*100/GT_bbox_count))
    end_time = time.time()-begin_time
    print("总的GTBbox个数为：{}，总的预测个数为：{}，误差为：{},"
          "predicted {} image,Total cost time{}s"\
          .format(sum(GT_bbox_count_list),
                  sum(Predict_bbox_count_list),
                  (sum(GT_bbox_count_list) - sum(Predict_bbox_count_list))*100/sum(GT_bbox_count_list),
                  img_count, end_time))
