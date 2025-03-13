import cv2
import sys
import numpy as np
import pickle
import numpy as np
import os

BLUR_OCC = 3


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    h, w, _ = flow.shape
    holes = np.ones((h, w), dtype=np.float32)
    
    # Check for NaN or Inf values
    invalid_mask = np.logical_or(np.isnan(flow[:,:,0]), np.isnan(flow[:,:,1]))
    invalid_mask = np.logical_or(invalid_mask, np.isinf(flow[:,:,0]))
    invalid_mask = np.logical_or(invalid_mask, np.isinf(flow[:,:,1]))
    
    # Check for very large values (>10^9)
    u_too_large = np.abs(flow[:,:,0]) > 1e9
    v_too_large = np.abs(flow[:,:,1]) > 1e9
    flow_too_large = np.logical_or(u_too_large, v_too_large)
    
    # Combine all hole conditions
    combined_holes = np.logical_or(invalid_mask, flow_too_large)
    
    # Mark holes as 0, no holes as 1
    holes[combined_holes] = 0
    
    return holes



def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h, w, _ = flow.shape
    has_hole = 1
    updated_flow = flow.copy()
    updated_holes = holes.copy()
    
    while has_hole == 1:
        has_hole = 0
        # Loop through all pixels
        for y in range(0, h):
            for x in range(0, w):
                # Check if this pixel is a hole
                if updated_holes[y, x] == 0:
                    # Look at 8 surrounding pixels
                    valid_vectors = []
                    
                    # Define the 8 neighboring pixels
                    neighbors = [
                        (y-1, x-1), (y-1, x), (y-1, x+1),
                        (y, x-1),             (y, x+1),
                        (y+1, x-1), (y+1, x), (y+1, x+1)
                    ]
                    
                    # Check each neighbor
                    for ny, nx in neighbors:
                        # Make sure the neighbor is within the image bounds
                        if 0 <= ny < h and 0 <= nx < w:
                            # If neighbor is not a hole, add its flow vector to valid_vectors
                            if updated_holes[ny, nx] == 1:
                                valid_vectors.append(updated_flow[ny, nx])
                    
                    # If we have valid neighbors, fill the hole with the average flow
                    if len(valid_vectors) > 0:
                        avg_flow = np.mean(valid_vectors, axis=0)
                        updated_flow[y, x] = avg_flow
                        updated_holes[y, x] = 1
                        has_hole = 1  # We made a change, so continue the while loop
    
    return updated_flow

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    height,width,_ = flow0.shape
    occ0 = np.zeros([height,width],dtype=np.float32)
    occ1 = np.zeros([height,width],dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4',diff)

    # ==================================================
    # ===== main part of step 5
    # ==================================================
    # to be completed...

    # For each pixel in frame 0
    for y in range(height):
        for x in range(width):
            # Get the flow vector at this pixel
            fx, fy = flow0[y, x]
            
            # Calculate the corresponding position in frame 1
            x1 = int(round(x + fx))
            y1 = int(round(y + fy))
            
            # Check if the position is within frame 1
            if 0 <= x1 < width and 0 <= y1 < height:
                # Cross-check the flow
                back_fx, back_fy = flow1[y1, x1]
                
                # If the sum of absolute differences is > 0.5, mark pixel as occluded
                if abs(fx + back_fx) + abs(fy + back_fy) > 0.5:
                    occ0[y, x] = 1
            else:
                # If mapped outside the image, mark as occluded
                occ0[y, x] = 1
    
    # For frame 1 occlusion map, if pixel is not targeted by any flow, mark as occluded
    target_map = np.zeros([height, width], dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            fx, fy = flow0[y, x]
            x1 = int(round(x + fx))
            y1 = int(round(y + fy))
            
            if 0 <= x1 < width and 0 <= y1 < height:
                target_map[y1, x1] = 1
    
    # Pixels in frame 1 that aren't targeted by any flow are occluded
    occ1 = 1.0 - target_map

    return occ0, occ1

def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    height, width, _ = flow.shape
    
    # Initialize the flow at time t with a large value to indicate holes
    iflow = np.ones((height, width, 2), dtype=np.float32) * 1e10
    
    # Initialize an accumulation buffer for photo-consistency-based blending
    consistency_buffer = np.ones((height, width), dtype=np.float32) * 1e10
    
    # For each pixel in frame 0
    for y in range(height):
        for x in range(width):
            # Get the flow vector at this pixel
            fx, fy = flow[y, x]
            
            # Forward-warp this flow vector to position t using splatting
            # Use splatting radius of ±0.5 pixels
            for yy in np.arange(-0.5, 0.51, 0.5):
                for xx in np.arange(-0.5, 0.51, 0.5):
                    # Calculate the target position in the interpolated frame
                    xt = int(round(x + t * fx + xx))
                    yt = int(round(y + t * fy + yy))
                    
                    # Check if the target position is within the frame
                    if 0 <= xt < width and 0 <= yt < height:
                        # Check photo-consistency between source and target
                        if x + fx >= 0 and x + fx < width and y + fy >= 0 and y + fy < height:
                            # Calculate color difference as a measure of consistency
                            # Calculate the position in frame 1
                            x1 = int(min(max(0, x + fx), width - 1))
                            y1 = int(min(max(0, y + fy), height - 1))
                            
                            # Calculate the color difference
                            color_diff = np.mean(np.abs(frame0[y, x].astype(np.float32) - 
                                                      frame1[y1, x1].astype(np.float32)))
                            
                            # If this is more consistent than what we've seen before, update the flow
                            if color_diff < consistency_buffer[yt, xt]:
                                consistency_buffer[yt, xt] = color_diff
                                iflow[yt, xt, 0] = fx
                                iflow[yt, xt, 1] = fy
    
    return iflow

def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''
    height, width, _ = frame0.shape
    iframe = np.zeros_like(frame0).astype(np.float32)
    
    # For each pixel in the interpolated frame
    for y in range(height):
        for x in range(width):
            # Get the flow vector at this position in the interpolated frame
            fx, fy = iflow[y, x]
            
            # Calculate the pixel positions in the source frames
            x0 = x - t * fx
            y0 = y - t * fy
            x1 = x + (1 - t) * fx
            y1 = y + (1 - t) * fy
            
            # Check if both source positions are valid and not occluded
            valid0 = 0 <= x0 < width - 1 and 0 <= y0 < height - 1
            valid1 = 0 <= x1 < width - 1 and 0 <= y1 < height - 1
            
            # Get occlusion status
            occluded0 = False
            occluded1 = False
            
            if valid0:
                x0_int, y0_int = int(x0), int(y0)
                occluded0 = occ0[y0_int, x0_int] > 0.5
            else:
                occluded0 = True
                
            if valid1:
                x1_int, y1_int = int(x1), int(y1)
                occluded1 = occ1[y1_int, x1_int] > 0.5
            else:
                occluded1 = True
                
            # Blend based on occlusion status
            if not occluded0 and not occluded1:
                # Both pixels are visible, blend based on t
                color0 = bilinear_interpolate(frame0, x0, y0)
                color1 = bilinear_interpolate(frame1, x1, y1)
                iframe[y, x] = (1 - t) * color0 + t * color1
            elif not occluded0:
                # Only frame 0 is visible
                iframe[y, x] = bilinear_interpolate(frame0, x0, y0)
            elif not occluded1:
                # Only frame 1 is visible
                iframe[y, x] = bilinear_interpolate(frame1, x1, y1)
            else:
                # Both are occluded, use the average color of the two frames for this pixel
                iframe[y, x] = (frame0[y, x] + frame1[y, x]) / 2.0
    
    return iframe

def bilinear_interpolate(image, x, y):
    """
    Perform bilinear interpolation of the pixel at (x, y) in the image
    """
    height, width, _ = image.shape
    
    # Ensure x and y are within image bounds
    x = max(0, min(width - 1.001, x))
    y = max(0, min(height - 1.001, y))
    
    # Get the four neighboring pixels
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    
    # Get the interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Get the pixel values
    p00 = image[y0, x0]
    p01 = image[y0, x1]
    p10 = image[y1, x0]
    p11 = image[y1, x1]
    
    # Interpolate
    result = (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 + (1 - wx) * wy * p10 + wx * wy * p11
    
    return result

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    # to be completed ...
    return cv2.GaussianBlur(im, (5, 5), 0)

def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
