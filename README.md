# AutoCalib_Sonar_Camera

Source code for paper: "Mutual Information -based Extrinsic Calibration of Camera-Sonar system leveraging
Sonar pseudo-pointcloud and Underwater Light Attenuation Prior"

## Data
You can obtain data from the following link: https://drive.google.com/drive/u/3/folders/1Ln7P6Haxj0c4bQeI53CX5XYzsy-R2j6t ,
then unzip to `data/` folder

## Run

```py
cd src
python script_[SCENARIO]_uw_[METHOD].py
```

- `SCENARIO` can be `simu` or `real`
- `METHOD` can be `RM` or `depth`

## Demo

## Reference code:
- https://robots.engin.umich.edu/SoftwareData/ExtrinsicCalib
- https://github.com/xmba15/automatic_lidar_camera_calibration
