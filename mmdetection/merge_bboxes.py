import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

########### 小于自己的目标检测库  #############
###########################################
def txt2list(bboxes_path, confidence_thresh=0.8):
    bboxes = []
    with open(bboxes_path) as f:
        for line in f:
            value = line.strip()
            score = value.split('\t')[0]
            y_min = value.split('\t')[1]
            x_min = value.split('\t')[2]
            y_max = value.split('\t')[3]
            x_max = value.split('\t')[4]
            if float(score) >= confidence_thresh:
                bboxes.append([y_min, x_min, y_max, x_max])
    f.close()
    return bboxes


def count_xml_bbox_num(xml_path):
    """
    :param xml_path: xml的路径
    :return: 返回本xml的Bbox数目
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    return len(root.findall('object'))
    # print("当前读入图片的bbox个数为：", ))

def compute_iou(rec1, rec2):

    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    #  转成 float
    S_rec1 = (float(rec1[2]) - float(rec1[0])) * (float(rec1[3]) - float(rec1[1]))
    S_rec2 = (float(rec2[2]) - float(rec2[0])) * (float(rec2[3]) - float(rec2[1]))

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(float(rec1[1]), float(rec2[1]))
    right_line = min(float(rec1[3]), float(rec2[3]))
    top_line = max(float(rec1[0]), float(rec2[0]))
    bottom_line = min(float(rec1[2]), float(rec2[2]))

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def draw_bbox(img,
              bboxes_path,
              confidence_thresh=0.8,
              color=(0,255,0),
              thinkness=2):
    """
    :param img: CV2.imread读取到内存的图像
    :param bboxes_path: 置信度Bbox的坐标
                        score \t y_min \t x_min \t y_max \t x_max \n
                        score \t y_min \t x_min \t y_max \t x_max \n
                        ...
    :param color: bbox框的颜色
    :param thinkness: bbox框的粗细
    :return:返回画好Bbox的image
    """
    # 此张图片绘制bbox的数量统计
    bboxes_count = 0
    bboxes = txt2list(bboxes_path, confidence_thresh=0.8)
    for bbox in bboxes:
        left_top = (int(float(bbox[0])), int(float(bbox[1])))
        right_bottom = (int(float(bbox[2])), int(float(bbox[3])))
        cv2.rectangle(
            img, left_top, right_bottom, color, thinkness)
        bboxes_count += 1
        # TODO 加入文本标签和置信度
    return img, bboxes_count


def draw_bbox_1(img,
                org_bboxes_path,
                bboxes_path,
                color):
    # 先找出第一个绘制的Bbox坐标
    org_bboxes = txt2list(org_bboxes_path)
    # 需要加入的Bbox坐标
    bboxes = txt2list(bboxes_path)

    # 因为坐标是缩小为（800,600,3）图像上，加至（4032,3024,3）上
    width_plus = 4032 / 800
    height_plus = 3024 / 600
    # 计数总共加入了多少个Bbox
    bboxes_count = 0
    for bbox in bboxes:
        # 对坐标进行原图标准化
        left_top = (int(float(bbox[0])*width_plus), int(float(bbox[1])*height_plus))
        right_bottom = (int(float(bbox[2])*width_plus), int(float(bbox[3])*height_plus))
        # bbox str---> float 并归一化到原图尺寸
        bbox = [float(bbox[0])*height_plus,
                float(bbox[1])*width_plus,
                float(bbox[2])*height_plus,
                float(bbox[3])*width_plus]
        # 判断已存在图片的IOU
        iou_list = []
        # 此时新的Bbox与所有的已存在的Bboxes做IOU比较
        for j in range(len(org_bboxes)):
            org_bbox = [org_bboxes[j][0], org_bboxes[j][1], org_bboxes[j][2], org_bboxes[j][3]]
            iou = compute_iou(bbox, org_bbox)
            iou_list.append(iou)
        # 若所有比较的值中最大的小于此阈值，则判断为新的Bbox
        if max(iou_list) < 0.5:
            cv2.rectangle(
                img, left_top, right_bottom, color, 5)
            bboxes_count += 1
        else:
            pass
            # print(max(iou_list))
    return img, bboxes_count


if __name__ == '__main__':

    image_path = './data/coco-wheat/test/000288.jpg'
    xml_path = '/home/yu/MxNet/VOCtemplate/VOC-wheat/Annotations/000288.xml'
    gt_bboxes_count = count_xml_bbox_num(xml_path)
    img = cv2.imread(image_path)
    img, count_1 = draw_bbox(img, 'cascade.txt', (0, 255, 0))
    img, count_2 = draw_bbox_1(img, 'cascade.txt', 'faster.txt', (255, 0, 0))
    cv2.imwrite('out.jpg', img)
    print("第一次绘制bboxes:{},第二次绘制bboxes:{},共绘制bboxes:{} \n 实际有bboxes:{}".format(
                                                                count_1,count_2, count_1+count_2,
                                                                gt_bboxes_count))

