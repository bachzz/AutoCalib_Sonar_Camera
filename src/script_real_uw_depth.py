import cv2
import numpy as np
from auto_calibration_real_uw_depth import AutoCalibration

import time


images = []
images_depth = []
sonars = []
masks = []

idxes = [6,10,11,12,15,16,20,21,22,25,27,28,30,31,35]

model_type = 'depthany2-large' #'depthany2-base' #'udepth' #'depthany2-small'

for i in idxes:
    images.append(cv2.imread(f'../data/underwater/real/img/{i}.png'))
    
    ## depthany2
    images_depth.append(np.load(f'../data/underwater/real/depth_{model_type}/{i}_raw_depth_meter.npy'))
    
    # ## udepth
    # images_depth.append(cv2.imread(f'../data/underwater/real/depth_{model_type}/{i}.png')[:,:,0])
    
    sonar_im = cv2.imread(f"../data/underwater/real/sonar/{i}_sonar.png")[:,:,0]
    
    sonar_im[:50,:] = 0
    sonars.append(sonar_im)

    masks.append(cv2.imread(f"../data/underwater/real/masks/{i}.png"))

# breakpoint()



gt_params = {'x': 0.12, 'y': 0.02, 'z': -0.15, 'roll_deg': 0,'pitch_deg': 0, 'yaw_deg': 0}

np.random.seed(42)

init_ranges = {
    'x': gt_params['x'] +  np.array([0.03]), #np.random.uniform(low=0.03, high=0.07, size=(10)), #np.linspace(0.05, 0.1, 9)[1:], #np.linspace(0.0, 0.05, 9)[1:], #np.zeros(8), #np.array([0]),
    'yaw': gt_params['yaw_deg'] +  np.array([-3]) #np.random.uniform(low=-3.5, high=-1.7, size=(10)) #np.linspace(0,1,9)[1:]
}
print(init_ranges)

x_opt = []
yaw_opt = []
inlier_scores = []
results = []

for pair in zip(init_ranges['x'], init_ranges['yaw']):
    print(f'\n[*] x = {pair[0]} - yaw = {pair[1]}')

    init_params = {'x': pair[0], 'y': gt_params['y'], 'z': gt_params['z'], 'roll_deg': gt_params['roll_deg'],'pitch_deg': gt_params['pitch_deg'], 'yaw_deg': pair[1]}
    
    params_grid = {'x':[], 'y':[], 'z':[], 'roll_deg':[], 'pitch_deg':[], 'yaw_deg':[]}
    N = 15

    params_grid['x'] = np.linspace(-0.2, 0.2, N)
    params_grid['y'] = np.linspace(-0.2, 0.2, N)
    params_grid['z'] = np.linspace(-0.1, 0.1, N)
    params_grid['roll_deg'] = np.linspace(-5, 5, N) * np.pi/180
    params_grid['pitch_deg'] = np.linspace(-5, 5, N) * np.pi/180
    params_grid['yaw_deg'] = np.linspace(-10, 10, N) * np.pi/180


    config = {
        'camera_intrinsic': np.array([
            [696.74100912,   0.        , 623.61744062],
        [  0.        , 691.67676207, 337.89199858],
        [  0.        ,   0.        ,   1.        ]])
    }

    calib = AutoCalibration(images, images_depth, sonars, masks, init_params, config, gt_params=gt_params, params_grid=params_grid)
    
    fid = 3
    ext_vec_init = np.array([init_params['x'], init_params['y'], init_params['z'], init_params['roll_deg'] * np.pi/180, init_params['pitch_deg'] * np.pi/180, init_params['yaw_deg'] * np.pi/180 ])
    image_init = calib.project_sonar_pointcloud_to_image(ext_vec_init, sonars[fid], calib.images_bgr[fid] )
    cv2.imshow('', image_init)
    cv2.waitKey(0)
    
    t = time.time()
    ext_vec_opt = calib.optimize()
    print(f"[optimization] time taken = {time.time() - t}")
    
    x_opt.append(ext_vec_opt[0])
    yaw_opt.append(ext_vec_opt[-1] * 180/np.pi)
    inlier_scores.append(calib.compute_inliers_score(ext_vec_opt))
    
    result = {'err_init':{'x': init_params['x']-gt_params['x'], 'yaw':init_params['yaw_deg']-gt_params['yaw_deg']}, 'err_final':{'x':x_opt[-1]-gt_params['x'],'yaw':yaw_opt[-1]-gt_params['yaw_deg']}}
    results.append(result)
    
    fid = 3
    image_opt = calib.project_sonar_pointcloud_to_image(ext_vec_opt, sonars[fid], cv2.convertScaleAbs(calib.images_bgr[fid], alpha=1, beta=1 )) 
    cv2.imshow('', image_opt)
    cv2.waitKey(0)
    breakpoint()
    
print(f"x_opt = {x_opt} - mean = {np.array(x_opt).mean()} - err_mean = {(np.array(gt_params['x']) - np.array(x_opt)).mean()} - std = {np.array(x_opt).std()}")
print(f"yaw_opt = {yaw_opt} - mean = {np.array(yaw_opt).mean()} - err_mean = {gt_params['yaw_deg'] - np.array(yaw_opt).mean()} - std = {np.array(yaw_opt).std()}")
print(f"inlier_scores = {inlier_scores} - mean = {np.array(inlier_scores).mean()} - std = {np.array(inlier_scores).std()}")

print(f"results = {results}")
np.save('results.npy', results)