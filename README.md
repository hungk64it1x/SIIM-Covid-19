# SIIM-FISABIO-RSNA COVID-19 Detection
### 🥉Bronze medal in competition🥉
![project-image](https://user-images.githubusercontent.com/80585483/129514712-1b29cc82-1a1e-47f3-bc7a-9ec3008e39fa.png)

> Identify and localize COVID-19 abnormalities on chest radiographs.

---

## Table of Contents
You're sections headers will be used to reference location of destination.



  - [Description](#description)
  - [My solution](#my-solution)
  - [How To Use](#how-to-use)
      - [Installation](#installation)
  - [References](#references)
  - [Awesome resource](#awesome-resources)

---

## Description

Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very similar to other viral and bacterial pneumonias on chest radiographs, which makes it difficult to diagnose. Your computer vision model to detect and localize COVID-19 would help doctors provide a quick and confident diagnosis. As a result, patients could get the right treatment before the most severe effects of the virus take hold...

[Back To The Top](#siim-fisabio-rsna-covid-19-detection)

---

## My solution
![image](https://user-images.githubusercontent.com/80585483/129535127-203d85a1-220a-43f1-a7fa-255cd0b84bff.png)

[Back To The Top](#siim-fisabio-rsna-covid-19-detection)

---

## How To Use

#### Installation
+ Run
```
pip install -r requirements.txt
```
### 👉Train classification
+ Put the data contains image data and its masks in folder data.
+ You can download the data references: [data](https://www.kaggle.com/ipythonx/covid19-detection-890pxpng-study)
+ Run train.py in folder model to train.
+ Example:
```
!python train.py --out_dir path_to_output_dir --init_ckpt path_2_checkpoint -- is_mixed True -- batch_size 8 --init_lr 0.0001 
```
### 👉Train detection
+ #### Yolov5
  + Put the dataset like this:
    ```
    /parent_folder
        /dataset
            /images
                /train
                /val
            /labels
                /train
                /val
        /yolov5
    ```
  
    + after that build the yaml file and run train.py in folder yolov5  to train.
    + Example:
    ```
    !python train.py --img {IMG_SIZE} \
                        --batch {BATCH_SIZE} \
                        --epochs {100} \
                        --data data_fold_{fold}.yaml \
                        --weights yolov5s.pt \
                        --save_period 10\
                        --project yolov5-covid19-folds\
                        --name yolov5s-e-100-img-256-fold-{fold}
    ```
+ #### EfficientDet
  + Run train.py in folder EfficientDet 
  + Example:
    ```
    !python train.py --image_size 512 --batch_size 4 --epochs 30 --output path/output
    ```
🎉For competition I've trained 5 fold yolov5s 256 image size, 5 fold yolov5x 512 image size and I used EfficientDet D3 512 image size, finally I merged them with WBF with iou_thresh = 0.6

[Back To The Top](#siim-fisabio-rsna-covid-19-detection)

---

## References
- [Kaggle Competition](https://www.kaggle.com/c/siim-covid19-detection)
- [EfficientDet Baseline](https://www.kaggle.com/c/global-wheat-detection/discussion/152368)

---
## Awesome resources
🌟[Pytorch EfficientDet](https://github.com/rwightman/efficientdet-pytorch)\
🌟[Pytorch](https://github.com/pytorch/pytorch)\
🌟[Albumentations](https://github.com/albumentations-team/albumentations)\
🌟[Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)\
🌟[Yolov5](https://github.com/ultralytics/yolov5)
