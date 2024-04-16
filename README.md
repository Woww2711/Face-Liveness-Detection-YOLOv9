# Face-Liveness-Detection-YOLOv9


![171246502863519](https://github.com/Woww2711/Face-Liveness-Detection-YOLOv9/assets/120792827/0e9cde53-f8f0-4646-99dc-45d7f41a0c09)
![17126524010443068](https://github.com/Woww2711/Face-Liveness-Detection-YOLOv9/assets/120792827/a942ca57-b020-4d83-b9ab-a1586127146e)
![img](https://github.com/Woww2711/Face-Liveness-Detection-YOLOv9/assets/120792827/6fc1b05b-a9c7-4135-88ad-ebcc3a0ae7ac)


## Overview
This project aims to create a decent Face Liveness Detection model from [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- The goal is to find and detect 3 kinds of face in input images or videos, and the predicted output will contain label and bounding boxes in every frame
- The 3 kinds of face are:
  + Replay: Images of faces from various people from many videos (0)
  + Real: Images of real faces (1)
  + Mask: Images of people wearing human-like mask or a mask (2)
- Data collected from [kaggle](kaggle.com), most come from [Training Data](https://www.kaggle.com/trainingdatapro) and [zalo](https://www.kaggle.com/datasets/hlly34/liveness-detection-zalo-2022)
- Guide can be found at [Face Liveness Detection](https://youtu.be/LqzPifvd09Q?si=8J1lmpr2wbDzrZ-h), go subscribe to his channel

## Data
Consists of 3807 images and 3807 text files, which sums up to 7164 files.
- Real: 2870 files
- Replay: 2938 files
- Mask: 1806 files
- 80% for training, 10% for validation, 10% for testing

## Training
Google Colab was used for training
- 
- GPU: T4 since it is free for some hours, took 3-4 days
- More to be updated

## Deploy
In progress...
