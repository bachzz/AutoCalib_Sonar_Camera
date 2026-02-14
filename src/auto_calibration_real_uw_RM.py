import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib
import matplotlib.pyplot as plt
import cv2


np.set_printoptions(precision=3, suppress=True)


D2R = np.pi / 180 # degree to radian const

phi_max = 10 #10 #5 #0 #10 #10 #0 #1 #5 #10
azi = 130 #120
minR = 0.1 #0.01 #0.1 #0.01
maxR = 4 #4 #10 #2
binsR = 418 #418 #512
binsA = 256 #256 #512

baseline_z = 0.15 #0.15 #0.205

acoustic_thres = 85 #105 #85 #85 #85 #80 #70 #80 #75

ranges = np.linspace(minR, maxR, binsR)
sonar_idx_arr = np.arange(-binsA/2, binsA/2+1, 1)
theta_i_arr = sonar_idx_arr * (azi/binsA)


class Util():
    def __init__(self):
        return
        
    
    def project_point_to_image(self, P_L, T, cam_K):
        # breakpoint()
        P_C = T @ np.array([[P_L[0], P_L[1], P_L[2], 1]]).T
        p_C = cam_K @ P_C[:3]
        # breakpoint()
        if np.unique(p_C)[0] == 0.0: return [np.nan, np.nan]
        # if p_C[2][0] < 0:
        #     print(p_C[2][0])
        return [int(p_C[0][0]/p_C[2][0]), int(p_C[1][0]/p_C[2][0]) ]
    
    
    def validate_image_point(self, p_C, image):
        # if p_C == np.nan: return False
        # breakpoint()
        return not np.isnan(p_C).any() and p_C[0] > 0 and p_C[0] < image.shape[1] and p_C[1] > 0 and p_C[1] < image.shape[0]
    
    
    def project_pointcloud_to_image(self, pointcloud, image, ext_vec, cam_K):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (euler-angle, radian)
        ###
        
        image_ = image.copy()
        
        t = np.array([ext_vec[:3]]).T
        # rot_mat = R.from_euler('zyx', ext_vec[3:]).as_matrix()
        rot_mat = R.from_euler('xyz', ext_vec[3:]).as_matrix()
        pointcloud_C = rot_mat @ pointcloud.T[:3,:] + t
        
        p_C = cam_K @ pointcloud_C
        p_C = p_C / p_C[2, :] # normalize with z
        p_C = p_C[:2,:]
        
        delta_C = p_C - np.array([[image.shape[1], image.shape[0]]]).T # check if pixel within image
        
        mask_C = (delta_C < 0) & (p_C > 0)
        mask_C = mask_C[0,:] & mask_C[1,:]
        
        p_C_intensity = np.append(p_C, np.array([pointcloud.T[3,:]]), axis=0)
        p_C_intensity_masked = np.where(mask_C, p_C_intensity, np.nan)
        p_C_intensity_masked = p_C_intensity_masked[:, ~np.isnan(p_C_intensity_masked).all(axis=0)]
        
        image_ = self.draw_pointcloud_on_image(p_C_intensity_masked.T, image_)
        
        return image_
    
    def project_sonar_pointcloud_to_image(self, sonar_im, image, ext_vec, cam_K):
        image_ = image.copy()
        
        ### extract closest points
        idxes_closest = np.ones((sonar_im.shape[1]), dtype=int) * -1
        for i in range(sonar_im.shape[1]):
            # idxes = np.argwhere(sonar_im[:,i] > 80) #95)#80)
            idxes = np.argwhere(sonar_im[:,i] > acoustic_thres)
            if len(idxes > 0):
                idxes_closest[i] = int(idxes.min())
                #print((i, idxes_closest[i]))
                # sonar_im_ = cv2.circle(sonar_im_, (i, idxes_closest[i]), radius=2, color=(0,255,0), thickness=1)
        # plt.imshow(sonar_im_); plt.show(block=True)
        points_closest = [{'r':ranges[idx], 'theta':theta_i_arr[i]} for i, idx in enumerate(idxes_closest) if idx != -1]
    
        # R_CS = R.from_euler("zyx", ext_vec[3:]).as_matrix() #np.eye(3)
        R_CS = R.from_euler("xyz", ext_vec[3:]).as_matrix() #np.eye(3)
        t_CS = np.array(ext_vec[:3])
        # print(points_closest)
        for point in points_closest:
            for phi in range(-phi_max,phi_max+1):
                r = point['r']
                theta = point['theta'] * np.pi/180
                phi = phi * np.pi/180 #10 * np.pi/180
                
                P_S = np.array([r * np.cos(-phi) * np.cos(theta), r * np.cos(-phi) * np.sin(theta), r * np.sin(-phi)]) # right-hand coord, z down
                P_C = (R_CS @ P_S) + t_CS
                P_C[[0, 1, 2]] = P_C[[1, 2, 0]] # convert to camera image coord	
                p_C = cam_K @ P_C
                p_C = (p_C / p_C[2]).astype(int)[:2]
                # print(p_C)
                if p_C[0] >= 0 and p_C[0] <= image.shape[1] and p_C[1] >= 0 and p_C[1] <= image.shape[0]:
                    image_ = cv2.circle(image_, p_C, radius=2, color=(0,255,0), thickness=1)
                
        return image_
    
    
    def colorCodingReflectivity(self, intensity):
        r, g, b = 0, 0, 0
        if intensity < 30:
            r = 0
            g = int(intensity * 255 / 30) & 0xFF
            b = 255
        elif intensity < 90:
            r = 0
            g = 0xFF
            b = int((90 - intensity) * 255 / 60 ) & 0xFF
        elif intensity < 150:
            r = int((intensity - 90) * 255 / 60 ) & 0xFF
            g = 0xFF
            b = 0
        else:
            r = 0xFF
            g = int((255-intensity) * 255 / (256-150) ) & 0xFF
            b = 0
        
        return (b, g, r)
    
    # def draw_pointcloud_on_image(self, projected_points, image):
    #     for point in projected_points:
    #         image = cv2.circle(image, center=point['coord'], radius=1, color=self.colorCodingReflectivity(point['intensity']), thickness=-1)
        
    #     return image
    
    def draw_pointcloud_on_image(self, projected_points, image):
        for point in projected_points:
            # print(point)
            image = cv2.circle(image, center=(int(point[0]), int(point[1])), radius=1, color=self.colorCodingReflectivity(point[2]), thickness=-1)
        
        return image


class HistogramHandler():
    def __init__(self, num_bins):
        self.num_bins = num_bins
        
        self.intensity_hist = None
        self.gray_hist = None
        self.joint_hist = None
    
        self.intensity_sum = None
        self.gray_sum = None
        
        self.total_points = None
        
        self.reset()
    
    
    def reset(self):
        self.intensity_hist = np.zeros(self.num_bins)
        self.gray_hist = np.zeros(self.num_bins)
        self.joint_hist = np.zeros([self.num_bins, self.num_bins])
    
        self.intensity_sum = 0
        self.gray_sum = 0
        
        self.total_points = 0
        
    def compute_stds(self):
        intensity_mean = self.intensity_sum / self.total_points
        gray_mean = self.gray_sum / self.total_points
        
        intensity_sigma = 0
        gray_sigma = 0
        for i in range(self.num_bins):
            intensity_sigma += self.intensity_hist[i] * (i - intensity_mean) ** 2
            gray_sigma += self.gray_hist[i] * (i - gray_mean) ** 2
        intensity_sigma = np.sqrt(intensity_sigma / self.total_points)
        gray_sigma = np.sqrt(gray_sigma / self.total_points)
        
        return intensity_sigma, gray_sigma
            
        

# plt.ion()
# fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
# plt.show()

from cv2.ximgproc import guidedFilter

class AutoCalibration():
    def __init__(self, images, sonars, masks, init_params, config, max_iters = 300, gt_params=None,  params_grid=None):
        
        ### general
        self.init_params = init_params
        self.gt_params = gt_params
        self.params_grid = params_grid
        
        self.images_bgr = images #image
        self.images = [] #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.images_RM = []
        
        self.masks = masks
        
        for image in images: #images_depth: #images:
            im_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            M_channel = im_RGB[:,:,1:].max(axis=2).astype(int)
            R_channel = im_RGB[:,:,0].astype(int)
            
            u0 = 0 #0.46353632 #0 #0.46353632 #0.06353632 #0.46353632 # Constant
            u1 = -1 #-0.38875134#-0.49598983 #-0.8 #-0.8 #-0.30598983 #-0.19598983#-0.19598983 #-0.49598983 # R
            u2 = 1 #0.49598983#0.38875134  #0.86 #0.86 #0.9#1.7 #+0.88875134 #+0.88875134 #+1.4875134 #+0.38875134 # M 
            # RM = u0 * np.ones(im_GRAY.shape) + u1 * R_channel + u2 * M_channel
            RM = u0 * np.ones(im_GRAY.shape)*255 + np.abs(u1 * R_channel + u2 * M_channel)
            # RM = (RM * 255).astype(np.uint8)
            RM = RM.astype(np.uint8)
            # RM = cv2.medianBlur(RM, 33)
            # cv2.imshow('', RM); cv2.waitKey(0); cv2.destroyAllWindows()
            # breakpoint()
            
            RM = cv2.medianBlur(RM, 31) # 31
            # RM = guidedFilter(im_GRAY, RM, 3, 2.5, -1)#15, 2.6, -1)
            # RM = cv2.bilateralFilter(RM, 9, 75,75)
            # RM = cv2.convertScaleAbs(RM, alpha=1.5)#2.5) #2)#1.5)
            # RM = cv2.convertScaleAbs(RM, alpha=3)
            
            # RM = cv2.fastNlMeansDenoising(RM, 10, 7,21)#7, 21)
            # RM = cv2.dilate(RM, np.ones((3,3),np.uint8),iterations=3)#3)
            # RM = cv2.medianBlur(RM, 17) #31)#21)
            # RM = cv2.medianBlur(RM, 31) #31)#21)
            # RM = cv2.convertScaleAbs(RM, alpha=1.5)
            # RM = cv2.bilateralFilter(RM,25,75,75)
            # RM = cv2.fastNlMeansDenoising(RM, 10, 7,21)#10, 7, 21)
            # RM = cv2.dilate(RM, np.ones((3,3),np.uint8),iterations=3)#1)#3)
            # RM = cv2.medianBlur(RM, 41) #31)#21)
            # RM = cv2.medianBlur(RM, 17) #31)#21)
            # RM = cv2.bilateralFilter(RM,31,55,55)
            # RM = cv2.convertScaleAbs(RM, alpha=2.5)
            # RM = cv2.addWeighted( RM, 3, RM, 0, 0)
            # RM[RM>75] = 255
            # breakpoint()
            
            # RM = ((RM / RM.max()) * 255 )#.astype(np.uint8)
            # # breakpoint()
            # _, mask_1 = cv2.threshold(RM, 150, 255, cv2.THRESH_BINARY_INV)
            # _, mask_2 = cv2.threshold(RM, 190, 255, cv2.THRESH_BINARY)
            # # breakpoint()
            # # RM[mask_1 == 255] *= 0.5
            # RM[mask_2 == 255] *= 1.5 
            # RM[RM > 255] = 255
            # RM = RM.astype(np.uint8)
            
            # RM = cv2.dilate(RM, np.ones((5,5),np.uint8),iterations=3)
            # RM = cv2.dilate(RM, np.ones((7,7),np.uint8),iterations=1)
            # RM = cv2.dilate(RM, np.ones((3,3),np.uint8),iterations=1)
            
            # fig, ax = plt.subplots(2,2); ax[0,0].imshow(cv2.cvtColor(self.images_RM[0], cv2.COLOR_GRAY2RGB)); ax[0,1].plot(np.histogram(self.images_RM[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], cv2.cvtColor(self.images_RM[0], cv2.COLOR_GRAY2BGR))); plt.show()
            
            # fig, ax = plt.subplots(2,3); ax[0,0].imshow(cv2.cvtColor(self.images_RM[0], cv2.COLOR_GRAY2RGB)); ax[0,1].plot(np.histogram(self.images_RM[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], cv2.cvtColor(self.images_RM[0], cv2.COLOR_GRAY2BGR))); ax[0,2].plot(self.hist_handler.intensity_hist); plt.show(block=False)
            
            self.images.append(im_GRAY)
            self.images_RM.append(RM)
            #breakpoint()
            
        self.image_H, self.image_W = self.images[0].shape
            
        self.sonars = sonars
        
        self.N = len(self.images)
        
        self.cam_K = config['camera_intrinsic']
        
        ### extrinsic matrix
        self.ext_vec_ = None
        
        
        ### gradient descent params     
        # step size params
        self.gamma_trans_ = 0.01 # translation
        self.gamma_trans_u_ = 0.07 #0.07 #0.1 #0.07 #0.03 #0.03 #0.03 #0.05 #0.06 #0.1 #0.1 # upper bound
        self.gamma_trans_l_ = 0.001 #0.001 #0.01 #0.001 # lower bound
        
        self.gamma_rot_ = 0.001 # rotation
        self.gamma_rot_u_ = 0.05 #0.05 #0.02 #0.02 #0.01 #0.02 #0.05 #3.14 * np.pi/180 #0.05 #3.14 * np.pi/180 #0.05 # upper bound
        self.gamma_rot_l_ = 0.0005 #0.1 * np.pi/180 #0.0005 #0.1 * np.pi/180 #0.0005 # lower bound
        
        self.eps_ = 1e-9
        
        # finite increments
        self.delta_ = np.array([0.01, 0.01, 0.01, 0.1 * D2R, 0.1 * D2R, 0.1 * D2R]) # x, y, z, r, p, y
        self.delta_thres = 0.0001 #0.0001
        
        # max inters
        self.max_iters_ = max_iters
        
        
        ### misc
        self.MAX_BINS = 32 #256 #256 #256
        self.num_bins = 32 #256 #256 #256
        self.bin_fraction = self.MAX_BINS / self.num_bins
        
        ### support
        self.hist_handler = HistogramHandler(self.num_bins)
        self.utils = Util()
    
    
    
    def plot_inliers_scores(self, flag):  
        flag_idxes = np.where(np.array(list(flag)) == '1')[0]
              
        X = list(self.params_grid.values())[flag_idxes[0]]
        Y = list(self.params_grid.values())[flag_idxes[1]]
        X_, Y_ = np.meshgrid(X, Y)
        # breakpoint()
        
        def compute_MI_lambda(x, y, flag): #flag):
            flag_idxes = np.where(np.array(list(flag)) == '1')[0]
            # breakpoint()
            ext_vec = np.array([self.gt_params['x'], self.gt_params['y'], self.gt_params['z'], self.gt_params['roll_deg'] * D2R, self.gt_params['pitch_deg'] * D2R, self.gt_params['yaw_deg'] * D2R ])
            ext_vec[flag_idxes[0]] = x
            ext_vec[flag_idxes[1]] = y
            return self.compute_inliers_score(ext_vec)
        
        mi_scores = np.vectorize(compute_MI_lambda)(X_, Y_, flag)
        
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}); ax.plot_surface(X_, Y_ * (1/D2R), mi_scores, cmap=matplotlib.cm.jet); plt.show()
        plt.colorbar(plt.pcolor(X_,Y_ * (1/D2R),mi_scores)); plt.show()
        # plt.colorbar(plt.pcolor(X_,Y_,mi_scores)); plt.show()
        
        breakpoint()    
    
    
    
    def plot_MI(self, flag):  
        flag_idxes = np.where(np.array(list(flag)) == '1')[0]
              
        X = list(self.params_grid.values())[flag_idxes[0]]
        Y = list(self.params_grid.values())[flag_idxes[1]]
        X_, Y_ = np.meshgrid(X, Y)
        # breakpoint()
        
        def compute_MI_lambda(x, y, flag): #flag):
            flag_idxes = np.where(np.array(list(flag)) == '1')[0]
            # breakpoint()
            ext_vec = np.array([self.gt_params['x'], self.gt_params['y'], self.gt_params['z'], self.gt_params['roll_deg'] * D2R, self.gt_params['pitch_deg'] * D2R, self.gt_params['yaw_deg'] * D2R ])
            ext_vec[flag_idxes[0]] = x
            ext_vec[flag_idxes[1]] = y
            return self.compute_MI(ext_vec)
        
        mi_scores = np.vectorize(compute_MI_lambda)(X_, Y_, flag)
        print(mi_scores.max())
        
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}); ax.plot_surface(X_, Y_ * (1/D2R), mi_scores, cmap=matplotlib.cm.jet); plt.show()
        
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}); ax.plot_surface(X_, Y_ , mi_scores, cmap=matplotlib.cm.viridis, alpha=0.85); plt.show(block=False); x = np.array([0.114, 0.114]); y = np.array([0.0, 0.0]);  z = np.array([4.25, 4.7]); ax.plot(x,y,z, 'r', linewidth=3);
        
        # plt.colorbar(plt.pcolor(X_,Y_ * (1/D2R),mi_scores)); plt.show()
        plt.colorbar(plt.pcolor(X_,Y_,mi_scores)); plt.show()
        
        breakpoint()
        
    
    def project_pointcloud_to_image(self, ext_vec, pointcloud, image):
        return self.utils.project_pointcloud_to_image(pointcloud, image, ext_vec, self.cam_K) 
    
    def project_sonar_pointcloud_to_image(self, ext_vec, sonar, image):
        return self.utils.project_sonar_pointcloud_to_image(sonar, image, ext_vec, self.cam_K) 

    
    def extract_sonar_pointcloud(self, sonar_im):
        # sonar_im_ = cv2.cvtColor(sonar_im.copy(), cv2.COLOR_GRAY2BGR)
        
        ### extract closest points
        idxes_closest = np.ones((sonar_im.shape[1]), dtype=int) * -1
        for i in range(sonar_im.shape[1]):
            # idxes = np.argwhere(sonar_im[:,i] > 80) #200) #80)
            idxes = np.argwhere(sonar_im[:,i] > acoustic_thres)
            if len(idxes > 0):
                idxes_closest[i] = int(idxes.min())
                #print((i, idxes_closest[i]))
                # sonar_im_ = cv2.circle(sonar_im_, (i, idxes_closest[i]), radius=2, color=(0,255,0), thickness=1)
        # plt.imshow(sonar_im_); plt.show(block=False)
        # breakpoint()
        
        points_closest = [{'r':ranges[idx], 'theta':theta_i_arr[i]} for i, idx in enumerate(idxes_closest) if idx != -1]
    
        pointclouds = []
        # breakpoint()
        
        for point in points_closest:
            for phi in range(-phi_max,phi_max+1):
                r = point['r']
                theta = point['theta'] * np.pi/180
                phi = phi * np.pi/180 #10 * np.pi/180
                
                P_S = np.array([r * np.cos(-phi) * np.cos(theta), r * np.cos(-phi) * np.sin(theta), r * np.sin(-phi), r]) # right-hand coord, z down, assume camera below sonar
                pointclouds.append(P_S)
        return np.array(pointclouds)
    
    
    def compute_inliers_score(self, ext_vec):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (zyx euler-angle, radian)
        ###
        
        #self.hist_handler.reset()
        total_points = 0
        inliers_num = 0
        
        for i in range(self.N):
            pointcloud = self.extract_sonar_pointcloud(self.sonars[i])
            # [[x, y, z, r]..]
            # breakpoint()
            
            ### project P_L to P_C
            t = np.array([ext_vec[:3]]).T
            # rot_mat = R.from_euler('zyx', ext_vec[3:]).as_matrix()
            rot_mat = R.from_euler('xyz', ext_vec[3:]).as_matrix()
            # pointcloud_C = rot_mat @ self.pointclouds[i].T[:3,:] + t
            pointcloud_C = rot_mat @ pointcloud.T[:3,:] + t
            pointcloud_C[[0, 1, 2],:] = pointcloud_C[[1, 2, 0],:] # convert to camera image coord	
            
            ### project P_C to p_C
            p_C = self.cam_K @ pointcloud_C
            p_C_ = p_C / p_C[2, :] # normalize with z
            p_C_ = p_C_[:2,:]
            
            p_C_[np.isnan(p_C_)] = 999999 # divided by 0 -> outliers
            p_C_ = p_C_.astype(int)
            
            delta_C = p_C_ - np.array([[self.image_W, self.image_H]]).T # check if pixel within image
            
            ### check: 1) wihin image, 2) non-negative pixel coord, 3) in front of camera
            mask_C = (delta_C < 0) & (p_C_ >= 0) & (p_C[2, :] > 0)
            mask_C = mask_C[0,:] & mask_C[1,:]
            
            p_C_masked = np.where(mask_C, p_C_, np.nan)
            p_C_masked = p_C_masked[:,~np.isnan(p_C_masked).any(axis=0)]
            
            mask_masked = self.masks[i][p_C_masked[1,:].astype(int), p_C_masked[0,:].astype(int),:]
            
            inliers_num += (mask_masked.shape[0] - (mask_masked.sum(axis=1)==0).sum() )
            total_points += mask_masked.shape[0]
        
        inliers_score = inliers_num / total_points
        print(f'inliers_score = {inliers_score} - ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }')
        return inliers_score
    
    
    def compute_hist(self, ext_vec):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (zyx euler-angle, radian)
        ###
        
        self.hist_handler.reset()
        
        for i in range(self.N):
            pointcloud = self.extract_sonar_pointcloud(self.sonars[i])
            # [[x, y, z, r]..]
            # breakpoint()
            
            ### project P_L to P_C
            t = np.array([ext_vec[:3]]).T
            # rot_mat = R.from_euler('zyx', ext_vec[3:]).as_matrix()
            rot_mat = R.from_euler('xyz', ext_vec[3:]).as_matrix()
            # pointcloud_C = rot_mat @ self.pointclouds[i].T[:3,:] + t
            pointcloud_C = rot_mat @ pointcloud.T[:3,:] + t
            pointcloud_C[[0, 1, 2],:] = pointcloud_C[[1, 2, 0],:] # convert to camera image coord	
            
            ### project P_C to p_C
            p_C = self.cam_K @ pointcloud_C
            p_C_ = p_C / p_C[2, :] # normalize with z
            p_C_ = p_C_[:2,:]
            
            p_C_[np.isnan(p_C_)] = 999999 # divided by 0 -> outliers
            p_C_ = p_C_.astype(int)
            
            delta_C = p_C_ - np.array([[self.image_W, self.image_H]]).T # check if pixel within image
            
            ### check: 1) wihin image, 2) non-negative pixel coord, 3) in front of camera
            mask_C = (delta_C < 0) & (p_C_ >= 0) & (p_C[2, :] > 0)
            mask_C = mask_C[0,:] & mask_C[1,:]
            
            # pointcloud_dists = np.sqrt((self.pointclouds[i].T[:3,:]**2).sum(axis=0))
            # pointcloud_dists = np.sqrt((pointclouds.T[:3,:]**2).sum(axis=0))
            p_C_intensity = np.append(p_C_, np.array([pointcloud.T[3,:]]), axis=0)
            p_C_intensity_masked = np.where(mask_C, p_C_intensity, np.nan)
            # breakpoint()
            
            p_C_intensity_masked_groups = np.array(np.split(p_C_intensity_masked, p_C_intensity_masked.shape[1]/ (phi_max*2+1), axis=1))
            # p_C_intensity_masked_groups = np.array(np.split(p_C_intensity_masked, p_C_intensity_masked.shape[1], axis=1))
            gray_intensity_masked_selected = []
            # breakpoint()
            for group_idx in range(int(p_C_intensity_masked.shape[1] / (phi_max*2+1))):
            # for group_idx in range(int(p_C_intensity_masked.shape[1])):
                if not np.isnan(p_C_intensity_masked_groups[group_idx,:,:]).any():
                    group = p_C_intensity_masked_groups[group_idx,:,:]
                    gray = np.array(self.images_RM[i][group[1,:].astype(int), group[0,:].astype(int)])
                    intensity = np.array(group[2,:])
                    
                    gray_intensity_masked_selected.append(np.append(np.array([gray]), np.array([intensity]), axis=0 ).mean(axis=1))
                    # gray_intensity_masked_selected.append(np.median(np.append(np.array([gray]), np.array([intensity]), axis=0 ), axis=1))
                    
                    # gray_intensity_masked_selected.append(np.append(np.array([gray]), np.array([intensity]), axis=0 ).max(axis=1))
                    # gray_intensity_masked_selected.append(np.append(np.array([gray]), np.array([intensity]), axis=0 ).min(axis=1))
            
            
            # p_C_intensity_masked = p_C_intensity_masked[:, ~np.isnan(p_C_intensity_masked).any(axis=0)] 
            
            # p_C_masked = p_C_intensity_masked[:2,:].astype(int)
            # intensity_masked = p_C_intensity_masked[2,:]
            
            # breakpoint()
            gray_intensity_masked_selected = np.array(gray_intensity_masked_selected)
            
            ### compute hist
            try:
                gray_masked_bin = gray_intensity_masked_selected[:, 0] / self.bin_fraction
                intensity_masked_bin = gray_intensity_masked_selected[:, 1] / self.bin_fraction 
                
                # ## remove outliers
                # # d = np.abs(gray_masked_bin - np.median(gray_masked_bin))
                # # mdev = np.median(d)
                # # s = d/mdev if mdev else np.zeros(len(d))
                # # mask = gray_masked_bin > 10 #s < 1.4 #np.abs(gray_masked_bin - gray_masked_bin.mean()) < 1 * gray_masked_bin.std()
                # mask = np.abs(gray_masked_bin - gray_masked_bin.mean()) < 0.6 * gray_masked_bin.std() # 1.9 #1.8 # 2
                # gray_masked_bin = gray_masked_bin[mask]
                # intensity_masked_bin = intensity_masked_bin[mask]
                # # breakpoint()
                # # breakpoint()
                
            except Exception as e:
                print(e)
                breakpoint()
            
            bins = self.num_bins #np.arange(0, self.num_bins+1)
            # bins = np.arange(0, self.num_bins + 1)
            self.hist_handler.gray_hist += np.histogram(gray_masked_bin, bins=bins)[0]
            self.hist_handler.intensity_hist += np.histogram(intensity_masked_bin, bins=bins)[0]
            self.hist_handler.joint_hist += np.histogram2d(gray_masked_bin, intensity_masked_bin, bins=(bins, bins))[0]
            
            self.hist_handler.gray_sum += gray_masked_bin.sum()
            self.hist_handler.intensity_sum += intensity_masked_bin.sum()
            
            self.hist_handler.total_points += len(gray_masked_bin)
            # breakpoint()
            
            # self.hist_handler.gray_hist = np.histogram(gray_masked_bin, bins=bins)[0]
            # self.hist_handler.intensity_hist = np.histogram(intensity_masked_bin, bins=bins)[0]
            # self.hist_handler.joint_hist = np.histogram2d(gray_masked_bin, intensity_masked_bin, bins=(bins, bins))[0]
            
            # self.hist_handler.gray_sum = gray_masked_bin.sum()
            # self.hist_handler.intensity_sum = intensity_masked_bin.sum()
            
            # self.hist_handler.total_points = len(gray_masked_bin)
        
        # breakpoint()
        # fid = 11
        # fig, ax = plt.subplots(2,3); ax[0,0].imshow(self.images[fid],cmap='gray'); ax[0,1].plot(np.histogram(self.images[fid].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[fid], self.images[fid]), cmap='gray');ax[1,1].set_xlabel('RM depth bins'); ax[1,1].set_ylabel('counts'); ax[0,2].plot(self.hist_handler.intensity_hist); ax[0,2].set_xlabel('acoustic depth bins'); ax[0,2].set_ylabel('counts'); plt.show(block=False); X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis); plt.show()
        
        # fig, ax = plt.subplots(2,3); ax[0,0].imshow(self.images[0],cmap='gray'); ax[0,1].plot(np.histogram(self.images[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], self.images[0]), cmap='gray'); ax[0,2].plot(self.hist_handler.intensity_hist); plt.show(block=False)
          
        # breakpoint()
        
 
            
        
    def estimate_MLE(self):
        prob_intensity = self.hist_handler.intensity_hist / self.hist_handler.total_points
        prob_gray = self.hist_handler.gray_hist / self.hist_handler.total_points
        prob_joint = self.hist_handler.joint_hist / self.hist_handler.total_points
        
        ### smoothing with KDE
        sigma_intensity, sigma_gray = self.hist_handler.compute_stds()
        
        ### bandwidth for KDE based on Silverman's rule of thumb
        sigma_intensity_bandwidth = 1.06 * np.sqrt(sigma_intensity) / (self.hist_handler.total_points ** 2)
        sigma_gray_bandwidth = 1.06 * np.sqrt(sigma_gray) / (self.hist_handler.total_points ** 2)
        
        # breakpoint()
        prob_intensity = cv2.GaussianBlur(prob_intensity, (0, 0), sigmaX=sigma_intensity_bandwidth)
        prob_gray = cv2.GaussianBlur(prob_gray, (0, 0), sigmaX=sigma_gray_bandwidth)
        prob_joint = cv2.GaussianBlur(prob_joint, (0, 0), sigmaX=sigma_gray_bandwidth, sigmaY=sigma_intensity_bandwidth)
        
        return prob_intensity, prob_gray, prob_joint
    
    
    def compute_MI(self, ext_vec, normalize = False):
        ### compute hist
        self.compute_hist(ext_vec)
        
        ### compute probs
        prob_intensity, prob_gray, prob_joint = self.estimate_MLE()
        
        prob_intensity_masked = prob_intensity[prob_intensity != 0]
        prob_gray_masked = prob_gray[prob_gray != 0]
        prob_joint_masked = prob_joint[prob_joint != 0]
        
        ### compute entropies
        H_intensity = -(prob_intensity_masked * np.log2(prob_intensity_masked) ).sum()
        H_gray = -(prob_gray_masked * np.log2(prob_gray_masked)).sum()
        H_joint = -(prob_joint_masked * np.log2(prob_joint_masked)).sum()
        
        ### compute MI
        mi_score = H_intensity + H_gray - H_joint
        mi_score_norm = 2 * mi_score / (H_intensity + H_gray)
        
        # print(f'mi_score = {mi_score}')
        
        # breakpoint()
        # X, Y = np.meshgrid(np.arange(0,256), np.arange(0,256)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, prob_joint, cmap=plt.cm.viridis);plt.show()
        
        # X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis);plt.show()
        
        # idx=25; plt.imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[idx], self.images_RM[idx]), cmap='gray'); plt.show()
        
        # print(f'mi_score = {mi_score} - ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }')
        # fig, ax = plt.subplots(2,3); ax[0,0].imshow(self.images_RM[0],cmap='gray'); ax[0,1].plot(np.histogram(self.images_RM[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], self.images_RM[0]), cmap='gray'); ax[0,2].plot(self.hist_handler.intensity_hist); plt.show(block=False); X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); ax = plt.subplots(subplot_kw={'projection':'3d'})[1]; ax.plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis); ax.set_xlabel('RM intensity bins',fontsize=16); ax.set_ylabel('Depth (range) bins', fontsize=16); ax.set_zlabel('Counts',fontsize=16); ax.xaxis.set_tick_params(labelsize=12); ax.yaxis.set_tick_params(labelsize=12); ax.zaxis.set_tick_params(labelsize=12); plt.show(block=False)
        
        # # plt.figure(); plt.imshow(self.images_RM[0], cmap='inferno_r'); plt.show(block=False)
        # # cv2.imshow('', self.images_RM[0]); cv2.waitKey(0); cv2.destroyAllWindows()
          
        # breakpoint()
        
        return mi_score_norm if normalize else mi_score
    
    
    def optimize(self):
        ###
        ### optimize ext_pose_ using Borwein (1988) gradient-descent approach
        ### ref: https://robots.engin.umich.edu/SoftwareData/ExtrinsicCalib
        ###
        
        
        ### initialize vars
        
        # max cost
        f_max = 0
        
        # extrinsic matrix
        ext_vec_prev = np.array([self.init_params['x'], self.init_params['y'], self.init_params['z'], self.init_params['roll_deg'] * D2R, self.init_params['pitch_deg'] * D2R, self.init_params['yaw_deg'] * D2R ])
        ext_vec = ext_vec_prev.copy()
        # print(f"ext_vec = {ext_vec}")
        print(f"ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }")
        
        # previous gradients
        grad_x_prev, grad_y_prev, grad_z_prev, grad_roll_prev, grad_pitch_prev, grad_yaw_prev = 0, 0, 0, 0, 0, 0
        
        ### optimization loop
        for idx in range(self.max_iters_):
    
            ### compute normalized gradients (in Eq. 15) for each component in ext mat
            
            # prev cost
            f_prev = self.compute_MI(ext_vec)
            if f_prev > f_max:
                f_max = f_prev
            
            # print(f"f_prev = {f_prev}")
            
            # increment & compute new cost for each component
            delta_x = ext_vec + np.array([self.delta_[0], 0, 0, 0, 0, 0]) 
            f = self.compute_MI(delta_x)
            grad_x = (f - f_prev) / self.delta_[0]
            
            delta_y = ext_vec + np.array([0, self.delta_[1], 0, 0, 0, 0])
            f = self.compute_MI(delta_y)
            grad_y = (f - f_prev) / self.delta_[1]
            
            delta_z = ext_vec + np.array([0, 0, self.delta_[2], 0, 0, 0])
            f = self.compute_MI(delta_z)
            grad_z = (f - f_prev) / self.delta_[2]
            
            delta_roll = ext_vec + np.array([0, 0, 0, self.delta_[3], 0, 0])
            f = self.compute_MI(delta_roll)
            grad_roll = (f - f_prev) / self.delta_[3]
            
            delta_pitch = ext_vec + np.array([0, 0, 0, 0, self.delta_[4], 0])
            f = self.compute_MI(delta_pitch)
            grad_pitch = (f - f_prev) / self.delta_[4]
            
            delta_yaw = ext_vec + np.array([0, 0, 0, 0, 0, self.delta_[5]])
            f = self.compute_MI(delta_yaw)
            grad_yaw = (f - f_prev) / self.delta_[5]
            
            if (grad_x==0 and grad_y==0 and grad_z==0) or (grad_roll==0 and grad_pitch==0 and grad_yaw==0):
                print("[!] Grads xyz or rpy are zeros. Stop optimization.")
                self.ext_vec_ = ext_vec
                break
                 
            # normalizing gradients
            # print(grad_x, grad_y, grad_z,grad_roll, grad_pitch, grad_yaw)
            grad_x = grad_x / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_y = 0 #grad_y / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_z = 0 #grad_z / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_roll = 0 #grad_roll / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            grad_pitch = 0 #grad_pitch / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            grad_yaw = grad_yaw / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            
            
            ### compute adative step size (separately for trans, rot)
            delta_ext_vec_trans = np.array([ext_vec[0], ext_vec[1], ext_vec[2]]) - np.array([ext_vec_prev[0], ext_vec_prev[1], ext_vec_prev[2]] )
            if np.sum(delta_ext_vec_trans ** 2) > 0:
                delta_grad_trans = np.array([grad_x, grad_y, grad_z]) - np.array([grad_x_prev, grad_y_prev, grad_z_prev])
                self.gamma_trans_ = np.sum(delta_ext_vec_trans ** 2) / (np.abs(np.array([delta_ext_vec_trans]) @ np.array([delta_grad_trans]).T )[0][0] + self.eps_)
            else:
                self.gamma_trans_ = self.gamma_trans_u_
            
            delta_ext_vec_rot = np.array([ext_vec[3], ext_vec[4], ext_vec[5]]) - np.array([ext_vec_prev[3], ext_vec_prev[4], ext_vec_prev[5] ])
            if np.sum(delta_ext_vec_rot ** 2) > 0:
                delta_grad_rot = np.array([grad_roll, grad_pitch, grad_yaw]) - np.array([grad_roll_prev, grad_pitch_prev, grad_yaw_prev])
                self.gamma_rot_ = np.sum(delta_ext_vec_rot ** 2) / (np.abs(np.array([delta_ext_vec_rot]) @ np.array([delta_grad_rot]).T )[0][0] + self.eps_)
            else:
                self.gamma_rot_ = self.gamma_rot_u_
            
            # bounded
            if self.gamma_trans_ > self.gamma_trans_u_:
                self.gamma_trans_ = self.gamma_trans_u_
            if self.gamma_trans_ < self.gamma_trans_l_:
                self.gamma_trans_ = self.gamma_trans_l_
        
            if self.gamma_rot_ > self.gamma_rot_u_:
                self.gamma_rot_ = self.gamma_rot_u_
            if self.gamma_rot_ < self.gamma_rot_l_:
                self.gamma_rot_ = self.gamma_rot_l_
        
            
            ### store ext_vec into prev
            ext_vec_prev = ext_vec.copy()
            
            ### update ext_vec
            #breakpoint()
            # print(self.gamma_trans_ * grad_x,self.gamma_trans_ * grad_y,self.gamma_trans_ * grad_z,self.gamma_rot_ * grad_roll,self.gamma_rot_ * grad_pitch,self.gamma_rot_ * grad_yaw)
            delta_mat = np.array([
                        self.gamma_trans_ * grad_x,
                        self.gamma_trans_ * grad_y,
                        self.gamma_trans_ * grad_z,
                        self.gamma_rot_ * grad_roll,
                        self.gamma_rot_ * grad_pitch,
                        self.gamma_rot_ * grad_yaw
                        ])
            # print(delta_mat)
            ext_vec = ext_vec + delta_mat
            
            
            ### compute new cost
            f = self.compute_MI(ext_vec)
            
            ### if cost decreases -> rollback & adjust step size (more conservative)
            #if f < f_prev:
            if f < f_prev:
                delta_mat = np.array([
                        self.gamma_trans_ * grad_x,
                        self.gamma_trans_ * grad_y,
                        self.gamma_trans_ * grad_z,
                        self.gamma_rot_ * grad_roll,
                        self.gamma_rot_ * grad_pitch,
                        self.gamma_rot_ * grad_yaw
                        ])
                ext_vec = ext_vec - delta_mat
                self.gamma_trans_u_ = self.gamma_trans_u_ / 1.2
                self.gamma_trans_l_ = self.gamma_trans_l_ / 1.2
                self.gamma_rot_u_ = self.gamma_rot_u_ / 1.2
                self.gamma_rot_l_ = self.gamma_rot_l_ / 1.2
            
                self.delta_ = self.delta_ / 1.1
                
                if self.delta_[0] < self.delta_thres:
                    self.ext_vec_ = ext_vec
                    break
                else:
                    continue
            
            ### update prev gradients
            grad_x_prev, grad_y_prev, grad_z_prev, grad_roll_prev, grad_pitch_prev, grad_yaw_prev = grad_x, grad_y, grad_z, grad_roll, grad_pitch, grad_yaw
            
            print(f"[iter={idx}] f = {f} - ext_vec = {ext_vec * [100, 100, 100, 1/D2R, 1/D2R, 1/D2R] }")
          
            
        print(f"[optimized] ext_vec = {ext_vec * [100, 100, 100, 1/D2R, 1/D2R, 1/D2R] }")
        
        
        # fig, ax = plt.subplots(2,3); ax[0,0].imshow(self.images_RM[0],cmap='gray'); ax[0,1].plot(np.histogram(self.images_RM[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], self.images_RM[0]), cmap='gray'); ax[0,2].plot(self.hist_handler.intensity_hist); plt.show(block=False); X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis);plt.show()
        
        # fig, ax = plt.subplots(2,3); ax[0,0].imshow(self.images_RM[0],cmap='gray'); ax[0,1].plot(np.histogram(self.images_RM[0].flatten(), bins=self.num_bins)[0]); ax[1,1].plot(self.hist_handler.gray_hist); ax[1,0].imshow(self.project_sonar_pointcloud_to_image(ext_vec, self.sonars[0], self.images_RM[0]), cmap='gray'); ax[0,2].plot(self.hist_handler.intensity_hist); plt.show(block=False); X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); ax = plt.subplots(subplot_kw={'projection':'3d'})[1]; ax.plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis); ax.set_xlabel('RM intensity bins',fontsize=16); ax.set_ylabel('Depth (range) bins', fontsize=16); ax.set_zlabel('Counts',fontsize=16); ax.xaxis.set_tick_params(labelsize=12); ax.yaxis.set_tick_params(labelsize=12); ax.zaxis.set_tick_params(labelsize=12); plt.show(block=False)
        
        # breakpoint()
        return self.ext_vec_ if self.ext_vec_ is not None else ext_vec
        