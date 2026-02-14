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
- Simulation:
  - Before calibration
    <img width="256" height="256" alt="Screenshot from 2026-02-15 00-33-50" src="https://github.com/user-attachments/assets/753bf6a3-bcd3-4a2c-a200-b4f8fe468120" />

  - After calibration
    <img width="256" height="256" alt="Screenshot from 2026-02-15 00-38-41" src="https://github.com/user-attachments/assets/73e5e1be-473c-43f9-8b04-150d5b21e097" />

- Real underwater data (contact author from this paper: https://link.springer.com/article/10.1007/s10846-024-02095-2)
  - Before calibration
    <img width="640" height="480" alt="Screenshot from 2026-02-15 00-40-03" src="https://github.com/user-attachments/assets/cd4db06e-c10f-42b4-ab25-fbe8d29ea719" />

  - After calibration
    <img width="640" height="480" alt="Screenshot from 2026-02-15 00-43-53" src="https://github.com/user-attachments/assets/7e4dcb8b-04bf-48c9-b192-b9daf6405d4a" />

    
## Reference code:
- https://robots.engin.umich.edu/SoftwareData/ExtrinsicCalib
- https://github.com/xmba15/automatic_lidar_camera_calibration
