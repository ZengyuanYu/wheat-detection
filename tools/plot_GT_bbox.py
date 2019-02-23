import os
import glob
import cv2
import xml.etree.ElementTree as ET


def plot_GT_bbox_signle(image_path,
                        xml_path,
                        display_label=None,
                        color=(0,0,255),
                        thickness=3
                        ):
    """
    :param image_path: 单张图片的路径
    :param xml_path: 此图片对用xml的路径
    :param display_label: 是否显示label,还没写
    :param color: bbox颜色
    :param thickness: bbox线粗细
    :return: 已绘制bbox的图片
    """
    img = cv2.imread(image_path)
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = name
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            labels.append(label)
            bboxes.append(bbox)
        for bbox in bboxes:
            left_top = (int(float(bbox[0])), int(float(bbox[1])))
            right_bottom = (int(float(bbox[2])), int(float(bbox[3])))
            cv2.rectangle(
                img, left_top, right_bottom, color=color, thickness=thickness)
        if display_label:
            pass
        else:
            pass
    return img


def plot_GT_bbox_batch(image_floder,
                        xml_floder,
                        save_folder,
                        display_label=None,
                        color=(0,0,255),
                        thickness=3
                        ):
    """
    :param image_floder: 图片存放文件夹
    :param xml_floder: xml存放文件夹
    :param save_folder: 已绘制图片的存储文件夹
    :param display_label: 是否显示label
    :param color: bbox颜色
    :param thickness: bbox线粗细
    :return: None
    """
    image_list = sorted(glob.glob(os.path.join(image_floder + '*.jpg')))
    for image_path in image_list:
        image_name = image_path.split('/')[-1]
        xml_name = image_name.split('.')[0] + '.xml'
        xml_path = os.path.join(xml_floder, xml_name)
        print(image_path, xml_path)
        img = plot_GT_bbox_signle(image_path, xml_path, display_label, color, thickness)
        cv2.imwrite(os.path.join(save_folder, image_name), img)

if __name__ == '__main__':

    # # plot_GT_bbox_signle测试
    # img = plot_GT_bbox_signle('./data/images/000000.jpg', './data/xml/000000.xml')
    # cv2.imwrite('./test.jpg', img)

    #
    plot_GT_bbox_batch('./data/images/', './data/xml', './')