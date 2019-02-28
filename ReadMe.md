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
  - my_config 训练模型文件
    - `faster_rcnn_r101(50)_fpn_wheat` 在 `wheat` 数据上的模型
    - `cascade_rcnn_r101_Soft_NMS_wheat` 更改 `NMS` 为 `Soft-NMS`
    - `cascade_rcnn_r101_GNwheat` 更改 `BN` 为 `GN`

### MxNet
- `inference_single_faster_rcnn.py` 利用训练模型完成单张图片的预测，并保存为 `txt` 文件

- `inference_faster_rcnn_image_list.py` 利用训练模型完成测试集的预测，并保存预测结果和显示准确率

  ```shell
  python inference_fater_rcnn_image_list.py --network faster_rcnn_resnet101_v1d_custom
  --model /home/yu/MxNet/model/fasterrcnn/faster_rcnn_resnet101_v1d_custom_epoch_300/faster_rcnn_resnet101_v1d_custom_best.params --save-path ../result/
  
  ```


### tools
*一些目标检测相关的处理工具*

- `plot_GT_bbox.py` 对标注好的数据进行可视化
  

  


