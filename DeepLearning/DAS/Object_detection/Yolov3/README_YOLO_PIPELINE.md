# DAS YOLOv8 Transfer Pipeline (Train -> Predict -> Review)

你这个目录名虽然是 `Yolov3`，但下面流程已经改成 **YOLOv8 迁移学习**。

## 0) 环境准备（在云服务器）
```bash
pip install ultralytics pillow
```

## 1) 生成训练集划分与 dataset.yaml
```bash
cd /home/levono/MachineLearning/DeepLearning/DAS/Object_detection/Yolov3
python prepare_dataset.py
```

输出：
- `datasets/das_yolo/dataset.yaml`
- `datasets/das_yolo/train.txt`
- `datasets/das_yolo/val.txt`

## 2) 两阶段迁移学习训练（推荐）
```bash
python finetune_yolov8_transfer.py \
  --model yolov8n.pt \
  --device 0 \
  --batch 16 \
  --imgsz 640 \
  --epochs-stage1 30 \
  --epochs-stage2 120 \
  --name das_transfer
```

说明：
- `stage1`：冻结部分层，先稳定收敛。
- `stage2`：全部解冻，做完整微调。
- 最终权重：`runs/das/das_transfer_stage2/weights/best.pt`

## 3) 用最佳模型全量预测（自动打标签）
```bash
python predict_ultralytics.py \
  --weights runs/das/das_transfer_stage2/weights/best.pt \
  --source ../DAS_object_detection/images_pro \
  --conf 0.20 \
  --iou 0.45 \
  --device 0 \
  --name predict_all
```

输出 YOLO txt：
- `runs/das/predict_all/labels/`

## 4) 把预测 txt 转成 AnyLabeling 可编辑 json
```bash
python pred_txt_to_anylabeling_json.py \
  --pred-labels runs/das/predict_all/labels \
  --out pred_json \
  --min-conf 0.20
```

输出 json：
- `pred_json/`（目录结构与原数据一致）

## 5) 你在 AnyLabeling 里逐一修正
修正完成后，再执行你已有的：
```bash
python ../DAS_object_detection/labelme_to_yolo.py
```
然后继续下一轮训练。

## 备注：是否从零训练？
不建议。你当前标注规模下，从零训练通常明显不如迁移学习稳定。
