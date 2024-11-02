import pdb
import glob
import cv2
import os
import tqdm
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.RachitJain import utils
from src.RachitJain.some_folder import folder_func

class PanaromaStitcher():
    def __init__(self):
        pass
    
    def compute_homography(self, src_pts, dst_pts):
        # print(src_pts[1])
        N = src_pts.shape[0]
        homography = []
        
        for i in range(N):
            Xs, Ys = src_pts[i]
            Xd, Yd = dst_pts[i]
            homography.append([-Xs, -Ys, -1, 
                       0, 0, 0, 
                       Xs*Xd, Ys*Xd, Xd])
            homography.append([0, 0, 0, 
                     -Xs, -Ys, -1, 
                      Xs*Yd, Ys*Yd, Yd])
        
        homography = np.array(homography)
        U, S, Vh = np.linalg.svd(homography)
        
        H = np.reshape(Vh[8], (3, 3))
        H = H/H[-1][-1]
        
        return H

    def apply_homography(self, image, H):
        # Applying homography to the image
        output_pts = []
        N = image.shape[0]
      
        for row in image:
          input = np.array([row[0], row[1], 1])
          input = input.transpose()
          mapped_pts = np.matmul(H, input)
          output_pts.append(mapped_pts[0]/mapped_pts[2])
          output_pts.append(mapped_pts[1]/mapped_pts[2])
      
        output_pts = np.array(output_pts)
        output_pts = output_pts.reshape(N, 2)
        
        return output_pts
    
    def find_homography_RANSAC(self, src_pts, dest_pts, RANSAC_iter=1000, eps=5):
        
        H = np.zeros([3,3])

        inliers_ids = []
        inliers_counts = []
        
        n = src_pts.shape[0]
        iter = 0
        while iter < RANSAC_iter:
          inliers_id = []
          pts_index = random.sample(range(0, n), 4)
      
          src_new = []
          dest_new = []
          for pt in range(4):
            src_new.append(src_pts[pts_index[pt]][:])
            dest_new.append(dest_pts[pts_index[pt]][:])
      
          src_new = np.asarray(src_new)
          dest_new = np.asarray(dest_new)
          H = self.compute_homography(src_new, dest_new)
          # Xs = np.matrix(Xs)
          dest_predicted = self.apply_homography(src_pts, H)
          for i in range(n):
            SSD = ((np.round(dest_predicted[i][0]) - int(dest_pts[i, 0]))**2 + (np.round(dest_predicted[i][1]) - int(dest_pts[i, 1]))**2)
            
            if SSD < eps:
              if i not in inliers_id:
                inliers_id.append(i)
      
          inliers_ids.append(inliers_id)
          inliers_counts.append(len(inliers_id))
      
          iter += 1
      
        largest_count_index = inliers_counts.index(max(inliers_counts))
        best_inliers_id = inliers_ids[largest_count_index]
      
        src_inliers = []
        dest_inliers = []
        for i in best_inliers_id:
          src_inliers.append(src_pts[i][:])
          dest_inliers.append(dest_pts[i][:])
      
        src_inliers = np.array(src_inliers)
        dest_inliers = np.array(dest_inliers)
        H = self.compute_homography(src_inliers, dest_inliers)
        
        return H, best_inliers_id
    
    
    def warp_stitch(self, images, H):
        
        reference_index = 2  # reference image
        
        # Initialize lists
        warp_Img = [None] * len(images)
        chain_H = []
        
        for index, image in enumerate(images):
            if index < reference_index:
                transform_H = np.eye(3)
                for j in range(index, reference_index):
                    transform_H = H[j] @ transform_H  # Forward chaining
                chain_H.append(transform_H)
        
            elif index > reference_index:
                transform_H = np.eye(3)
                for j in range(index - 1, reference_index - 1, -1):
                    transform_H = np.linalg.inv(H[j]) @ transform_H  # Backward chaining
                chain_H.append(transform_H)
            else:
                warp_Img[index] = image
                chain_H.append(np.eye(3))
                continue
        
        # print('Compute corners')
        corners = [utils.GetCorners(img, (chain_H[i]/chain_H[i][2,2])) for i, img in enumerate(images)]
        # print('Compute corners done')
        
        # Compute min and max coords
        min_xy_coord = np.amin(np.array(corners), axis=(0, 1))
        max_xy_coord = np.amax(np.array(corners), axis=(0, 1))
        
        # final panoramic dimensions
        final_img_dim = max_xy_coord - min_xy_coord
        row = int(final_img_dim[1])
        column = int(final_img_dim[0]*1.3)
        pan_img = np.zeros((row, column, 3), dtype=np.uint8)
        # image canvas
        # pan_img = np.zeros((int(final_img_dim[1]), int(final_img_dim[0]), 3))
        
        # print('Create panorama image')
        for i, img in enumerate(images):
            stitched_image = utils.Final_Panaroma(pan_img, img, (chain_H[i]/chain_H[i][2,2]), min_xy_coord)
        
        return stitched_image

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        # Convert image paths to OpenCV image objects
        temp_images = [cv2.imread(img_path) for img_path in all_images]
        images = [cv2.resize(img, None, fx=0.5, fy=0.5) for img in temp_images]
        
        # print('---images---')
        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        self.say_hi()
        self.do_something()
        self.do_something_more()

        utils.some_func()
        folder_func.foo()

        homography_matrix_list =[]
        # Calculate homography matrices
        for i in range(len(images)-1):
            img1 = images[i]
            img2 = images[i + 1]

            # Detect and describe keypoints in both images
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # Match keypoints
            matcher = cv2.FlannBasedMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Calculate homography matrix
            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                
                homography, E = self.find_homography_RANSAC(src_pts, dst_pts)
                
                homography_matrix_list.append(homography)
        
        # print(homography_matrix_list)  
        
        stitched_image = self.warp_stitch(images, homography_matrix_list)      
        
        # Return Final panaroma
        # stitched_image = cv2.imread(panorama_image)
        
        return stitched_image, homography_matrix_list 
    
        
    def say_hi(self):
        print('Hii From Rachit Jain...')
    
    def do_something(self):
        return None
    
    def do_something_more(self):
        return None