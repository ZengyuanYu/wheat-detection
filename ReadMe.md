### 利用目标检测算法完成麦穗计数
 
**实验表明，用不同的神经网络框架进行结果融合效果最好**

**本工程意图使用mmdetection进行融合，只需要MxNet给出预测的坐标**

### mmdetection

 - `inference_single.py` 利用已训练模型完成单张图片的预测
 - `inference_image_list.py` 利用已训练模型预测多张图片，保存对应bboxes
 - `merge_bboxes.py` 利用两种算法预测的bboxes结果根据IOU合并
 - `merge_bboxes_v2.py` 更新算法合并，直接一个`.py`文件完成所有操作
 - mmdet
   - api

     - `inference.py` 图片推演和显示相关函数，更改部分为：

     ```python
     def _prepare_data(img, img_transform, cfg, device):
        img = mmcv.imread(img)
     def show_result_yu(img, result, dataset='coco', score_thr=0.3, out_file=None, show=False):
     ```

### MxNet
- `inference_single_faster_rcnn.py` 利用训练模型完成单张图片的预测，并保存为 `txt` 文件


