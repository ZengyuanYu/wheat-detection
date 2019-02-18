import mmcv
import numpy as np
import torch
import cv2
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes


def _prepare_data(img, img_transform, cfg, device):
    # 小于添加
    img = mmcv.imread(img)
    # 小于添加
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device):
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_single_yu(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        print(img)
        yield _inference_single(model, img, img_transform, cfg, device)


def _inference_generator_yu(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()
    print(not isinstance(imgs, list))
    if not isinstance(imgs, list):
        print("single")
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        print("进入generator")
        return _inference_generator(model, imgs, img_transform, cfg, device)


def inference_detector_yu(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator_yu(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3):
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)


def show_result_yu(img, result, dataset='coco', score_thr=0.3, out_file=None, show=False):
    class_names = ['wheat']
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        out_file=out_file)

