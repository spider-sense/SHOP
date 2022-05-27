# SHOP
Repo containing code + dataset for Small Handheld Object Pipeline (SHOP).

- Paper [[arxiv](https://arxiv.org/abs/2203.15228)]
- [SHOP dataset](https://github.com/spider-sense/SHOP/releases/tag/0.1.0)
- [Models and test results](https://drive.google.com/drive/u/0/folders/1DbA9OkVI6kw_TNvhMKQHpfm8U9v0gzC8) (to recreate results)
- Handheld dataset creation program [<a href="https://github.com/spider-sense/handheld-classification">source code</a>] 
[[website](https://spider-sense.github.io/handheld-classification/)]

## Installation
Requirements:
- Linux (only needed if using OpenPose)

Quick installation using Anaconda/miniconda (recommended):

This will install both PyTorch and TensorFlow.

```bash
git clone https://github.com/spider-sense/SHOP.git --depth 1
cd SHOP
# Linux/macOS
conda env create -f environment.yml
# Windows
conda env create -f environment_windows.yml
```

The `--depth 1` flag is recommended to reduce git clone size.

## Inference
After installing, you will need to move the weights inside our [drive folder](https://drive.google.com/drive/u/0/folders/1DbA9OkVI6kw_TNvhMKQHpfm8U9v0gzC8) to the correct locations in your SHOP repo.

After this, you are ready to run our pipeline! Run one of the following commands depending on your OS and cross-reference with our results on drive to make sure everything's running smoothly.

```bash
# Linux/macOS
python SHOP.py --weights ./yolov5/yolov5s.pt --weights_path ./DeblurGANv2/fpn_mobilenet.h5 --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model pose_estimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 1.1 --handheld --save-txt --source ./tests/
# Windows
python SHOP.py --weights ./yolov5/yolov5s.pt --weights_path ./DeblurGANv2/fpn_mobilenet.h5 --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model pose_estimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 1.1 --handheld --save-txt --source ./tests/ --poseNum -1
```

## Testing
Requirements:
- Linux (for creating non top-down results)

To replicate our results in the paper, you may have to take a few more steps.

Download the labels and blurLabels directory to the validation directory in your repo then extract all of the zips inside these directories. Additionally, download the images and blurImages zip and extract its contents into the validation directory.

After this, you are ready to generate results!

Generating Non-Deblur Results
1. Run the following inference command from the SHOP directory to create 3 directories of detections for yolov5s, yolov5m, and yolov5l.
```bash
python SHOP.py --weights ./yolov5/yolov5s.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --noDeblur --source ./validation/images/ --save-txt --handheld --nosave --noPose; python SHOP.py --weights ./yolov5/yolov5m.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --noDeblur --source ./validation/images/ --save-txt --handheld --nosave --noPose; python SHOP.py --weights ./yolov5/yolov5l.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --noDeblur --source ./validation/images/ --save-txt --handheld --noPose --nosave 
```

2. Move the detections directory filled with only txt files in the results folder (after appropriately renaming them) to the labels directory.

3. Run annotHandheldJson.py. Make sure that the variable labelDir is set to "labels".

4. Afterwards, run labelOff.py while making sure that the variable labelDir is set to "labels". After this program finishes running, you should be able to see each detection model's mAP and their PR curve in the generated comparison_graph.jpg file.

Generating Deblur Results
1. Run the following inference command from the SHOP directory to create 3 directories of detections for yolov5s, yolov5m, and yolov5l.
```bash
python SHOP.py --weights ./yolov5/yolov5s.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --source ./validation/blurImages/ --save-txt --handheld --nosave --noPose; python SHOP.py --weights ./yolov5/yolov5m.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --source ./validation/blurImages/ --save-txt --handheld --nosave --noPose; python SHOP.py --weights ./yolov5/yolov5l.pt --model mobilenet_thin_432x368 --det-model ./yolov5/crowdhuman_yolov5m.pt --pose-model ../yolov5-pose/poseEstimation/simdr_hrnet_w48_256x192.pth --upper-conf-thres 0.7 --conf-thres 0.001 --weights_path DeblurGANv2/fpn_mobilenet.h5 --source ./validation/blurImages/ --save-txt --handheld --noPose --nosave
```

2. Move the detections directory filled with only txt files in the results folder (after appropriately renaming them) to the blurLabels directory. 

3. Run annotHandheldJson.py. Make sure that the variable labelDir is set to "blurLabels".

4. Afterwards, run labelOff.py while making sure that the variable labelDir is set to "blurLabels". After this program finishes running, you should be able to see each detection model's mAP and their PR curve in the generated comparison_graph.jpg file.
