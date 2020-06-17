import numpy as np
from numpy.linalg import inv
import cv2
from matplotlib import pyplot as plt
import pdb
import matplotlib.patches as patches
from scipy.linalg import svd

def importVideo(path):

    # Description
    # Inputs
    # Outputs

    vid = cv2.VideoCapture(path);
    N_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT));
    vidArr = [];
    
    for i in range(N_frames):
        frame = vid.read();
        frame = frame[1];    # NOTE: frames are imported in BGR not RGB format
        vidArr.append(frame);

    vidArr = np.squeeze(np.array([vidArr]));
        
    return vidArr

def trapMask(height,width):
    
    # Frame a face video, extract a sequence of face trapezoids as described in paper "Cheel_ROI" [Insert real reference later]


    # Trapezoid Mask (centered)
    mask = np.zeros((height,width));
        
    # Parmas: 0.5 × width, 0.4 × width, and 0.58 × height

    # Trap pt1
    x = int((width-0.5*width)/2);
    y = int((height-0.58*height)/2);
    p1 = [x,y];

    # Trap pt2
    p2 = [int(x + 0.5*width), y];

    # Trap pt3
    p3 = [int(x + 0.45*width), int(y+0.58*height)];

    # Trap pt4
    p4 = [int(x + 0.05*width), int(y+0.58*height)];

    # Fill in trapezoid
    mask = cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4]), color = (1,1,1));
#    mask = np.repeat(mask[...,np.newaxis],dim,-1)

    return mask, [p1, p2, p3, p4]

def getTrapROIFromRect(img, x, y, w, h, first=1):

    # Given an input image and a rectangular ROI inside the image, extract a trapezoidal ROI from inside the rectangular ROI as in the "cheek ROI" paper

    [mask, pts] = trapMask(h, w);

    if len(img.shape) == 2:
        mask = np.zeros(img.shape);
        m,n = img.shape;
    elif len(img.shape) == 3:
        mask = np.zeros(img.shape[0:2]);
        m,n,k = img.shape;
    else:
        assert(1==0)


    for i in range(len(pts)):
        pts[i][0] = pts[i][0] + x;
        pts[i][1] = pts[i][1] + y;
    mask = cv2.fillConvexPoly(mask, np.array([pts[0], pts[1], pts[2], pts[3]]), color = (1,1,1));
    mask = np.uint8(mask);

    if first != 0:
        return mask
    else:
        # ROI parameters
        ROI_w = 0.12*w;
        ROI_h = 0.15*h;

        # ROI 1 (only) parameters
        ROI_1x = pts[3][0] - ROI_w*2/3;
        ROI_1y = pts[3][1] - 0.45*(pts[3][1]-pts[0][1]);

        # ROI 2 (only) parameters
        ROI_2x = pts[2][0] - ROI_w/3;
        ROI_2y = pts[2][1] - 0.45*(pts[2][1]-pts[0][1]);

        ROI1 = [int(ROI_1x), int(ROI_1y), int(ROI_w), int(ROI_h)];
        ROI2 = [int(ROI_2x), int(ROI_2y), int(ROI_w), int(ROI_h)];

        return mask, ROI1, ROI2


def getSURFThroughoutVid(vidArr):
    
    # Description:                                                    
    #
    # Inputs: 4D video data (the last dimension comes from RGB adding extra dim)        
    #                                               
    # Outputs: a list containing the coordinates of matched SURF points between adjacent video frames. The output is of the form:
    # [matches between 1st two frames, matches between 2nd two frames, ..., matches between N-1-th two franes]
    # each element in this list contains two lists inside of it (the first containing the coordinates in the ith frame and the second containing the coordinates of the i+t-th frame.

    N,m,n,k = vidArr.shape;    
    gray = np.zeros((N,m,n),dtype = 'uint8');
    faceParms = np.zeros((N,1,4)); # Defined by x, y, width, height rectangle parms
    face = np.zeros((N,m,n,k));
    SURF = [];
    KeyPoints = [];
    Descriptors = [];
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); 
    
    for i in range(N):

        # Extract Face in each frame
        gray[i,:,:] = np.array(cv2.cvtColor(vidArr[i,:,:,:], cv2.COLOR_BGR2GRAY), dtype='uint8');
        
        faceParms[i,:,:] = face_cascade.detectMultiScale(gray[i,:,:], 1.3, 5);
        
        x = int(faceParms[i,:,:][0][0]);
        y = int(faceParms[i,:,:][0][1]);
        w = int(faceParms[i,:,:][0][2]);
        h = int(faceParms[i,:,:][0][3]);
        
        #face = np.zeros((N,m,n,k));
        #face[i,y:y+h,x:x+w,:] = vidArr[i,y:y+h,x:x+w,:];

        face = vidArr[i,y:y+h,x:x+w,:];

        # Extract face-trapezoid and ROIs
        parms = getTrapROIFromRect(gray[i], x, y, w, h, i)
        if i != 0:
            mask = parms;
        else:
            mask, init_ROI1, init_ROI2 = parms;

        # Draw ROIs
        #fig,ax = plt.subplots(1)
        #ax.imshow(gray[0,:,:],cmap='gray');
        #
        #rect1 = patches.Rectangle((init_ROI1[0], init_ROI1[1]), init_ROI1[2], init_ROI1[3], facecolor=None)
        #ax.add_patch(rect1)
        #
        #rect2 = patches.Rectangle((init_ROI2[0], init_ROI2[1]), init_ROI2[2], init_ROI2[3], facecolor=None)
        #ax.add_patch(rect2)
        #plt.show()
        #pdb.set_trace()

        # Detect SURF Features                                                   
        surf = cv2.xfeatures2d.SURF_create();
        
        # Append keypoints and descriptors to global lists
        keypoints, descriptors = surf.detectAndCompute(gray[i],mask);
        KeyPoints.append(keypoints);
        Descriptors.append(descriptors);

    # Initialize empty list, which will contain coordinates of all matched keypoints between adjacent frames.

    Coor_Matches = [];

    # Match keypoints throughout video
    for i in range(len(KeyPoints)-1):

        # Get keypoints between adjacent frames
        k0 = KeyPoints[i];
        k1 = KeyPoints[i+1];

        # Get descriptors associated with keypoints
        d0 = Descriptors[i];
        d1 = Descriptors[i+1];

        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True);
        
        # Match descriptors.
        matches = bf.match(d0,d1)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Get list of keypoint coordinates between ith and i+t-th frames
        coors_i0 = [k0[match.queryIdx].pt for match in matches];
        coors_i1 = [k1[match.trainIdx].pt for match in matches];
        
        
        Coor_Matches.append([coors_i0, coors_i1]);
        
    return Coor_Matches, [init_ROI1, init_ROI2]

def KLT_Matrix(P1, P2):

    m,n = P1.shape

    # Build system of equations to find homography matrix.
    A = np.array([[ P1[0,0], P1[1,0], 0, 0, 1, 0, -P2[0,0]]]);
    B = np.array([[ 0, 0, P1[0,0], P1[1,0], 0, 1, -P2[1,0]]]);
    A = np.vstack((A,B));
    for i in range(n-1):
        B = np.array([[ P1[0,i+1], P1[1,i+1], 0, 0, 1, 0, -P2[0,i+1] ]]);
        A = np.vstack((A,B));
        B = np.array([[ 0, 0, P1[0,i+1], P1[1,i+1], 0, 1, -P2[1,i+1] ]]);
        A = np.vstack((A,B));
        
    # Solve system
    A = A.T@A;
    U, S, V = svd(A);
    h = V[-1,:];
    h = h/h[-1];
    H = np.array([[h[0], h[1], h[4]], [h[2], h[3], h[5]], [0, 0, 1]]);

    #for j = 1:length(matches)
    #    Ai = [x(j) y(j) 1 0 0 0 -xhat(j)*x(j) -xhat(j)*y(j) -xhat(j)
    #    0 0 0 x(j) y(j) 1 -yhat(j)*x(j) -yhat(j)*y(j) -yhat(j)];
    #A = [A; Ai];


    #A = (P2@P1.T)@inv(P1@P1.T)
       
    return H

def trackFeatures(video, featMatches):

    # Given that we have a video, a set of good features from each frame, and an ROI this function will determine the mapping that tracks the good features, which can then be used to track the ROI.

    N,m,n,k = vidArr.shape; 
    A_mats = [];

    for i in range(N-1):
        frame_i0 = featMatches[i][0];
        frame_i1 = featMatches[i][1];
        numFeats = len(frame_i0);
        feats_i0 = np.zeros((2,numFeats));
        feats_i1 = np.zeros((2,numFeats));
        
        for j in range(numFeats):
            feats_i0[:,j] = frame_i0[j];
            feats_i1[:,j] = frame_i1[j];

        # Plot featues to get feel
        #fig, (ax1,ax2) = plt.subplots(1,2);
        #ax1.imshow(video[i,:,:,::-1]);
        #ax1.scatter(feats_i0[0,:],feats_i0[1,:])
        #ax2.imshow(video[i+1,:,:,::-1]);
        #ax2.scatter(feats_i1[0,:],feats_i1[1,:])
        #plt.show()
        #pdb.set_trace();

        A = KLT_Matrix(feats_i0,feats_i1); 
        A_mats.append(A);
    
    return A_mats

def cheekSigs(vidArr, init_ROIs, A_mats):

    # Spatially average the cheek ROI on each frame to obtain two cheek signals

    N, m, n, k = vidArr.shape;

    # Initialize signal
    sig_l = np.zeros((3,N));
    sig_r = np.zeros((3,N));

    # First frame
    parms_l = init_ROIs[0];
    parms_r = init_ROIs[1];
    x_l, y_l, w_l, h_l = parms_l; # gather left ROI parameters
    x_r, y_r, w_r, h_r = parms_r; # gather right ROI parameters
    rect_l = np.array([[x_l, x_l+w_l, x_l+w_l, x_l],[y_l, y_l, y_l+h_l, y_l+h_l]]);
    rect_r = np.array([[x_r, x_r+w_r, x_r+w_r, x_r],[y_r, y_r, y_r+h_r, y_r+h_r]]);
    ROI_l = vidArr[0, y_l:y_l+h_l, x_l:x_l+w_l, :]; # Extract left ROI parameters
    ROI_r = vidArr[0, y_r:y_r+h_r, x_r:x_r+w_r, :]; # Extract right ROI parameters
    sig_l[:,0] = ROI_l.mean(axis=(0,1));
    sig_r[:,0] = ROI_r.mean(axis=(0,1));

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(ROI_l, cmap= 'gray')
    ax2.imshow(vidArr[0])        
    plt.show()
    pdb.set_trace();

    # Remaining frames
    for i in range(N-1):
        pdb.set_trace()
        A = A_mats[i];
        rect_l = A@np.vstack((rect_l,np.ones((1,4))));
        rect_r = A@np.vstack((rect_r,np.ones((1,4))));
        pdb.set_trace()
        rect_l = rect_l[:2,:];
        rect_r = rect_r[:2,:];
        
        pdb.set_trace()
        rectL = rect_l.astype(int);
        rectR = rect_r.astype(int);

        pdb.set_trace()
        mask_l = np.zeros((m,n));
        mask_l = cv2.fillConvexPoly(mask_l, np.array([rectL[:,0],rectL[:,1], rectL[:,2], rectL[:,3]]), color = (1,1,1));

        mask_r = np.zeros((m,n));
        mask_r = cv2.fillConvexPoly(mask_r, np.array([rectR[:,0],rectR[:,1], rectR[:,2], rectR[:,3]]), color = (1,1,1));

        for j in range(k):
            ROI_l = mask_l*vidArr[i+1,:,:,j];
            ROI_r = mask_r*vidArr[i+1,:,:,j];
            sig_l[:,i+1] = ROI_l[np.nonzero(ROI_l)].mean();
            sig_r[:,i+1] = ROI_r[np.nonzero(ROI_r)].mean();
        
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(ROI_l, cmap= 'gray')
        ax2.imshow(vidArr[i+1])        
        plt.show()
        pdb.set_trace();

    return sig_l, sig_r
    
print("Importing Video")
#vidArr = importVideo("/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/lower_quality/fcmh0/sa1-video-fjre0.avi");
vidArr = importVideo("/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/fram1-original.mov");
print("Calculating SURF points in the trapezoid ROI")
[featMatches, init_ROIs] =getSURFThroughoutVid(vidArr);
print("Tracking features")
A_mats = trackFeatures(vidArr, featMatches)
sig_l, sig_r = cheekSigs(vidArr, init_ROIs, A_mats);
pdb.set_trace();
