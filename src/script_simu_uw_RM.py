import cv2
import numpy as np

from auto_calibration_simu_uw_RM import AutoCalibration

import time



images = []
images_depth = []
sonars = []
masks = []


idxes = [3,4,5,6,7,8,10,11,12,13,15,16,17,20,21,22,25,26,27,28,29,30,31,34,35]

# 147
fnames = ['1743075589.460603','1743075589.727273','1743075590.027277','1743075590.260614','1743075590.560618','1743075590.927289','1743075591.327295','1743075591.493963','1743075592.260640','1743075592.693980','1743075592.927316']

scene_id = 147 
type_id = '3' 

assert(len(fnames) == len(np.unique(fnames)))

for fname in fnames:
    images.append(cv2.imread(f'../data/underwater/simu/rgb_uw_{scene_id}_type_{type_id}/{fname}.png'))
    sonar_im = cv2.flip(np.load(f'../data/underwater/simu/sonar_{scene_id}_polar/{fname}.npy'), 1)
    
    sonars.append(sonar_im)
    
    masks.append(cv2.imread(f"../data/norm/simu/masks_{scene_id}/{fname}.png"))


gt_params = {'x': 0.0, 'y': 0, 'z': -0.205, 'roll_deg': 0,'pitch_deg': 0, 'yaw_deg': 0}

init_ranges = {
    'x': np.array([0.05]), #np.random.uniform(low=0.03, high=0.07, size=(10)), #np.array([0]),
    'yaw': np.array([-3.5]) #np.random.uniform(low=-3.5, high=-1.7, size=(10)) #np.arange(3,5,0.25) + 0.25
}

print(init_ranges)

x_opt = []
yaw_opt = []
inlier_scores = []
results = []

for pair in zip(init_ranges['x'], init_ranges['yaw']):
    print(f'\n[*] x = {pair[0]} - yaw = {pair[1]}')

    init_params = {'x': pair[0], 'y': 0.0, 'z': -0.205, 'roll_deg': 0,'pitch_deg': 0, 'yaw_deg': pair[1]}
    
    config = {
        'camera_intrinsic': np.array([
            [256.,   0.        , 256.],
        [  0.        , 256., 256.],
        [  0.        ,   0.        ,   1.        ]])
    }

    calib = AutoCalibration(images, sonars, masks, init_params, config)

    #init_params = gt_params
    ext_vec_init = [init_params['x'], init_params['y'], init_params['z'], init_params['roll_deg'] * np.pi/180, init_params['pitch_deg'] * np.pi/180, init_params['yaw_deg'] * np.pi/180]

    # plot sample projections with init_params
    fid = 6 #0 #8 #6
    image_init = calib.project_sonar_pointcloud_to_image(ext_vec_init, sonars[fid], cv2.convertScaleAbs(calib.images_bgr[fid], alpha=1, beta=1 ))
    cv2.imshow('', image_init)
    cv2.waitKey(0)

    t = time.time()
    ext_vec_opt = calib.optimize()
    print(f"[optimization] time taken = {time.time() - t}")
    
    x_opt.append(ext_vec_opt[1])
    yaw_opt.append(ext_vec_opt[-1] * 180/np.pi)
    inlier_scores.append(calib.compute_inliers_score(ext_vec_opt))
    
    
    result = {'err_init':{'x': init_params['x']-gt_params['x'], 'yaw':init_params['yaw_deg']-gt_params['yaw_deg']}, 'err_final':{'x':x_opt[-1]-gt_params['y'],'yaw':yaw_opt[-1]-gt_params['yaw_deg']}}
    results.append(result)
    
    fid = 6
    image_opt = calib.project_sonar_pointcloud_to_image(ext_vec_opt, sonars[fid], cv2.convertScaleAbs(calib.images_bgr[fid], alpha=1, beta=1 ))
    cv2.imshow('', image_opt)
    cv2.waitKey(0)
    
    
print(f"results = {results}")

np.save('results.npy', results)