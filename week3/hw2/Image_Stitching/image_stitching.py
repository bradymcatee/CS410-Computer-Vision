import cv2
import sys
import numpy as np

def computeHomography(pairs):
    A = []
    for [[x1, y1]], [[x2, y2]] in pairs:
        A.append([0,0,0,x1,y1,1,-y2*x1,-y2*y1,-y2])
        A.append([x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2])

    A = np.array(A)
    
    U,S,V = np.linalg.svd(A)

    H = np.reshape(V[-1], (3,3))

    H = (1/H.item(8)) * H

    return H
 
def d(pair, H):
    p1 = np.array(pair[0][0])
    p2 = np.array(pair[1][0])
    p1 = np.append(p1, 1)
    p2 = np.append(p2, 1)


    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    bestInliers = []
    bestH = None

    for i in range(max_num_trial):
        pairs = [list_pairs_matched_keypoints[i] for i in np.random.choice(len(list_pairs_matched_keypoints), 4)]
        H = computeHomography(pairs)

        inliers = []
        for c in list_pairs_matched_keypoints:
            if d(c, H) < threshold_reprojtion_error:
                inliers.append(c)


        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            bestH = H
            if len(bestInliers) > (len(list_pairs_matched_keypoints) * threshold_ratio_inliers):
                break

    print(len(bestInliers))
    print('----------------------------------------')
    print(len(list_pairs_matched_keypoints))

    return bestH

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================

    gray1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors of each image
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2,k=2)

    list_pairs_matched_keypoints = []
    for m,n in matches:
        if m.distance < ratio_robustness*n.distance:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            list_pairs_matched_keypoints.append([[p1],[p2]])

    return list_pairs_matched_keypoints

def reconstruct(img_1, x, y):
    if (x < 0 or x >= img_1.shape[1]-1 or 
        y < 0 or y >= img_1.shape[0]-1):
        return np.zeros(3) 
    
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, img_1.shape[1]-1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, img_1.shape[0]-1)
    
    wx = x - x0
    wy = y - y0
    
    p00 = img_1[y0, x0].astype(np.float32)
    p10 = img_1[y1, x0].astype(np.float32)
    p01 = img_1[y0, x1].astype(np.float32)
    p11 = img_1[y1, x1].astype(np.float32)
    
    return (1-wx) * (1 - wy) * p00 + \
           wx * (1 - wy) * p01 + \
           (1 - wx) * wy * p10 + \
           wx * wy * p11

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    H_1_inv = np.linalg.inv(H_1)
    
    # First, determine panorama dimensions by transforming corners
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    
    # Create corners of img_1
    corners_1 = np.array([[0, 0, 1],
                         [w1-1, 0, 1],
                         [0, h1-1, 1],
                         [w1-1, h1-1, 1]]).T
    
    # Transform corners
    transformed_corners = H_1.dot(corners_1)
    transformed_corners = transformed_corners / transformed_corners[2]
    
    # Find min and max points
    min_x = min(np.min(transformed_corners[0]), 0)
    max_x = max(np.max(transformed_corners[0]), w2)
    min_y = min(np.min(transformed_corners[1]), 0)
    max_y = max(np.max(transformed_corners[1]), h2)
    
    # Calculate offsets and size
    offset_x = int(max(0, -min_x))
    offset_y = int(max(0, -min_y))
    output_w = int(max_x + offset_x)
    output_h = int(max_y + offset_y)
    
    # Create output image
    warped_img = np.zeros((output_h, output_w, 3), dtype=np.float32)
    mask = np.zeros((output_h, output_w), dtype=np.uint8)
    
    # Warp img_1
    for y in range(output_h):
        for x in range(output_w):
            pt = np.array([x - offset_x, y - offset_y, 1])
            warped_pt = H_1_inv.dot(pt)
            
            if warped_pt[2] != 0:
                warped_x = warped_pt[0] / warped_pt[2]
                warped_y = warped_pt[1] / warped_pt[2]
                
                if (0 <= warped_x < img_1.shape[1]-1 and 
                    0 <= warped_y < img_1.shape[0]-1):
                    warped_img[y, x] = reconstruct(img_1, warped_x, warped_y)
                    mask[y, x] = 1
    
    # Place img_2 in the output space
    result = np.zeros_like(warped_img)
    result[offset_y:offset_y+h2, offset_x:offset_x+w2] = img_2.astype(np.float32)
    
    # Create masks for both images
    mask_warped = (mask > 0)
    mask_img2 = np.zeros_like(mask, dtype=bool)  # Changed to boolean
    mask_img2[offset_y:offset_y+h2, offset_x:offset_x+w2] = True
    
    # Convert to boolean arrays for logical operations
    overlap = np.logical_and(mask_warped, mask_img2)
    warped_only = np.logical_and(mask_warped, ~mask_img2)
    img2_only = np.logical_and(~mask_warped, mask_img2)
    
    # Copy warped_img where it's the only image
    result[warped_only] = warped_img[warped_only]
    
    # Average blend in the overlap region
    result[overlap] = (warped_img[overlap] + result[overlap]) / 2
    
    # Find the valid region to crop
    rows = np.any(result.sum(axis=2) > 0, axis=1)
    cols = np.any(result.sum(axis=2) > 0, axis=0)  # Fixed axis
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        result = result[ymin:ymax+1, xmin:xmax+1]
    
    return result

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)


    return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))