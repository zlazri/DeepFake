import numpy as np
from numpy.linalg import inv
import cv2
from matplotlib import pyplot as plt
import pdb
import matplotlib.patches as patches
from scipy.linalg import svd
from matplotlib.patches import Circle
import dlib
from imutils import face_utils
import imutils
from scipy.signal import butter, filtfilt, find_peaks
from scipy.linalg import hankel, svd
from pyts.decomposition import SingularSpectrumAnalysis as ssa

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

        return mask, ROI1, ROI2, pts


def getSURFThroughoutVid(vidpath):
    
    # Description:                                                    
    #
    # Inputs: Video path      
    #                                               
    # Outputs: a list containing the coordinates of matched SURF points between adjacent video frames. The output is of the form:
    # [matches between 1st two frames, matches between 2nd two frames, ..., matches between N-1-th two franes]
    # each element in this list contains two lists inside of it (the first containing the coordinates in the ith frame and the second containing the coordinates of the i+t-th frame.

    # Get video parameters
    cap = cv2.VideoCapture(vidpath);
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(300,3))

    # Detect Face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); 

    # Obtain Parameters of the face region
    faceParms = face_cascade.detectMultiScale(old_gray, 1.3, 5);
    x = int(faceParms[0][0]);
    y = int(faceParms[0][1]);
    w = int(faceParms[0][2]);
    h = int(faceParms[0][3]);
        
    face= old_gray[y:y+h,x:x+w];

    # Extract face-trapezoid and ROIs
    parms = getTrapROIFromRect(old_gray, x, y, w, h, 0)  
    mask, init_ROI1, init_ROI2, trap_pts = parms;
    [x1, y1, w1, h1] = init_ROI1;
    [x2, y2, w2, h2] = init_ROI2;
    [x_t, y_t, w_t, h_t] = trap_pts;

    # Start Creating Figure
    fig, ax = plt.subplots(1)
    ax.imshow(old_frame[:,:,::-1]);
    rect1 = patches.Rectangle((x,y),w, h, fill = False, color =(0,1,0));
    ROI1 = patches.Rectangle((x1,y1),w1, h1, fill = False, color =(1,0,0));
    ROI2 = patches.Rectangle((x2,y2),w2, h2, fill = False, color =(1,0,0));
    trapezoid = plt.Polygon(trap_pts,  fill=None, edgecolor='b')
    ax.add_patch(trapezoid)
    ax.add_patch(rect1);
    ax.add_patch(ROI1);
    ax.add_patch(ROI2);   
    ######################
        
    # Detect SURF keypoints
    surf = cv2.xfeatures2d.SURF_create();
    keypoints, descriptors = surf.detectAndCompute(old_gray,mask);
    keypoints = [keypoints[idx].pt for idx in range(0, len(keypoints))]
    kk = np.array([[keypoints[0][0],keypoints[0][1]]]);
    for i in range(len(keypoints)-1):
        kk = np.vstack((kk, keypoints[i+1]));
    keypoints = kk;

    # Finish Figure
    mCirc, nCirc = kk.shape;
    for i in range(mCirc):
        circ = patches.Circle((kk[i,0],kk[i,1]), edgecolor='y', fill = False)
        ax.add_patch(circ);
    plt.show()
    ########################
    

    # Set LK tracking Parameters
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initial keypoints/formatting
    p0 = np.array(kk, np.float32);
    p0 = p0.reshape(-1,1,2);

    # Initialize empty lists for tracked keypoints
    mask = np.zeros_like(old_frame)
    k0 = [];
    k1 = [];

    # Perform Tracking
    for i in range(N_frames-1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #pdb.set_trace()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Add points to list of matches
        k0.append(p0);
        k1.append(p1);
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

    # Save matched keypoints
    featMatches = [k0, k1];       
        
    return featMatches, [init_ROI1, init_ROI2]

def Affine_Matrix(P0, P1):

    m,n = P1.shape
    x_old = P0[:,1];
    y_old = P0[:,0];
    x_new = P1[:,1];
    y_new = P1[:,0];

    # Build system of equations to find homography matrix.
    A = np.array([[ x_old[0], y_old[0], 0, 0, 1, 0, -x_new[0]]]);
    B = np.array([[ 0, 0, x_old[0], y_old[0], 0, 1, -y_new[0]]]);
    A = np.vstack((A,B));
    for i in range(m-1):
        B = np.array([[ x_old[i+1], y_old[i+1], 0, 0, 1, 0, -x_new[i+1] ]]);
        A = np.vstack((A,B));
        B = np.array([[ 0, 0, x_old[i+1], y_old[i+1], 0, 1, -y_new[i+1] ]]);
        A = np.vstack((A,B));
    # Solve system
    A = A.T@A;
    U, S, V = svd(A);
    h = V[-1,:];
    h = h/h[-1];
    H = np.array([[h[0], h[1], h[4]], [h[2], h[3], h[5]], [0, 0, 1]]);
       
    return H


def cheekSigs2(vidpath, init_ROIs, featMatches):

    # Spatially average the cheek ROI on each frame to obtain two cheek signals
   
    # Get video parameters
    cap = cv2.VideoCapture(vidpath);
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    ret, old_frame = cap.read()
    frame = old_frame[:,:,::-1]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(300,3)) 
    m,n,k = old_frame.shape

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

    sig_l[0,0] = frame[y_l:y_l+h_l, x_l:x_l+w_l,0].mean();
    sig_r[0,0] = frame[y_r:y_r+h_r, x_r:x_r+w_r,0].mean();
    sig_l[1,0] = frame[y_l:y_l+h_l, x_l:x_l+w_l,1].mean();
    sig_r[1,0] = frame[y_r:y_r+h_r, x_r:x_r+w_r,1].mean();
    sig_l[2,0] = frame[y_l:y_l+h_l, x_l:x_l+w_l,2].mean();
    sig_r[2,0] = frame[y_r:y_r+h_r, x_r:x_r+w_r,2].mean();

    # Gather feature matches
    k0, k1 = featMatches;

    # Remaining frames
    for i in range(N-1):

        # Features from the current and next frame
        feat_i0 = k0[i].squeeze();
        feat_i1 = k1[i].squeeze();
        feat_i0 = feat_i0[:,::-1];
        feat_i1 = feat_i1[:,::-1];

        # Get mapping
        A = Affine_Matrix(feat_i0,feat_i1);
        #pdb.set_trace()

        rect_L = A@np.vstack((rect_l,np.ones((1,4))));
        rect_R = A@np.vstack((rect_r,np.ones((1,4))));

        rect_L = rect_L[:2,:];
        rect_R = rect_R[:2,:];
        
        rectL = rect_L.astype(int);
        rectR = rect_R.astype(int);

        mask_l = np.zeros((m,n));
        mask_l = cv2.fillConvexPoly(mask_l, np.array([rectL[:,0],rectL[:,1], rectL[:,2], rectL[:,3]]), color = (1,1,1));

        mask_r = np.zeros((m,n));
        mask_r = cv2.fillConvexPoly(mask_r, np.array([rectR[:,0],rectR[:,1], rectR[:,2], rectR[:,3]]), color = (1,1,1));

        ret,frame = cap.read()
        frame = frame[:,:,::-1];

        sig_l[0,i+1] = frame[mask_l*frame[:,:,0]>0,0].mean();
        sig_r[0,i+1] = frame[mask_r*frame[:,:,0]>0,0].mean();
        sig_l[1,i+1] = frame[mask_l*frame[:,:,1]>0,1].mean();
        sig_r[1,i+1] = frame[mask_r*frame[:,:,1]>0,1].mean();
        sig_l[2,i+1] = frame[mask_l*frame[:,:,2]>0,2].mean();
        sig_r[2,i+1] = frame[mask_r*frame[:,:,2]>0,2].mean();

        # Display tracked affine transform
        #plt.imshow((-1*mask_l+1)*(-1*mask_r+1)*frame[:,:,0], cmap = 'gray')
        #plt.show()

        rect_l = rect_L
        rect_r = rect_R

    return sig_l, sig_r

def CHROM(RGB, fs):

    M,N = RGB.shape;
    L = np.round(fs); # sliding window length (1s length)
    H = np.zeros((1, N));
    one = np.ones((3,1));
    CRM = np.array([[3,-2,0],[1.5,1,-1.5]]);

    for t in range(N-L+1): # in each sliding window
        C = RGB[:, t:t+L-1];
        AVE = C.mean(axis=1).reshape((3,1))*np.eye(3);
        NORMAL = inv(AVE);
        S = CRM @ (NORMAL @ C - one);
        P = np.array([[1, -np.std(S[0,:])/np.std(S[1,:])]]) @ S;
        H[0, t:t+L-1] = H[0, t:t+L-1]+(P-np.mean(P))/np.std(P);

    w = np.concatenate((np.array([np.linspace(1,L-1,L-1)]), L*np.ones((1, N-2*L+2)), np.array([np.linspace(L-1,1,L-1)])),axis=1)
    H=H/w;
    
    return H

def getAndTrackDlibFeatures(vidpath):

    # Get video parameters
    cap = cv2.VideoCapture(vidpath);
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(300,3))

    # Initialize DLIB face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/shape_predictor_68_face_landmarks.dat");

    rects = detector(old_gray,1);

    for (i,r) in enumerate(rects):
        rect = r

    shape = predictor(old_gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Plot DLIB face landmarks
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # show the face number
    cv2.putText(old_frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for number, (x, y) in enumerate(shape):
        text = str(number);
        text_origin = (x - 5, y - 5)

        if number == 33:
            y2 = y
        if number == 40:
            x1 = x;
            y_up1 = y;
        if number == 47:
            x2 = x;
            y_up2 = y;

        cv2.circle(old_frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(old_frame, text, text_origin, cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 0, 0));
    
    y1 = np.max([y_up1, y_up2]);

    # Define ROI

    x = x1
    y = y1;
    w = x2-x1;
    h = y2-y1;

    ROI = [x, y, w, h];

    cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 0, 255), 2);
    #cv2.imshow("Output", old_frame)
    #cv2.waitKey(0)
    #pdb.set_trace()

##################################    # Track face landmarks


    # Set LK tracking Parameters
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initial keypoints/formatting
    p0 = np.array(shape, np.float32);
    p0 = p0.reshape(-1,1,2);

    # Initialize empty lists for tracked keypoints
    mask = np.zeros_like(old_frame)
    k0 = [];
    k1 = [];

    # Perform Tracking
    for i in range(N_frames-1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #pdb.set_trace()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Add points to list of matches
        k0.append(p0);
        k1.append(p1);
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

    # Save matched keypoints
    featMatches = [k0, k1];       
        
    return featMatches, ROI
##################################################

##################################################
def cheekSigs(vidpath, ROI, featMatches):

    # Spatially average the cheek ROI on each frame to obtain two cheek signals
   
    # Get video parameters
    cap = cv2.VideoCapture(vidpath);
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    ret, old_frame = cap.read()
    frame = old_frame[:,:,::-1]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(300,3)) 
    m,n,k = old_frame.shape

    # Initialize signal
    sig = np.zeros((3,N));

    # First frame
    x, y, w, h = ROI; # Gather ROI parameters

    rect = np.array([[x, x+w, x+w, x],[y, y, y+h, y+h]]);

    sig[0,0] = frame[y:y+h, x:x+w,0].mean();
    sig[1,0] = frame[y:y+h, x:x+w,1].mean();
    sig[2,0] = frame[y:y+h, x:x+w,2].mean();

    # Gather feature matches
    k0, k1 = featMatches;

    # Remaining frames
    for i in range(N-1):

        # Features from the current and next frame
        feat_i0 = k0[i].squeeze();
        feat_i1 = k1[i].squeeze();
        feat_i0 = feat_i0[:,::-1];
        feat_i1 = feat_i1[:,::-1];

        # Get mapping
        A = Affine_Matrix(feat_i0,feat_i1);
        #pdb.set_trace()

        Rect = A@np.vstack((rect,np.ones((1,4))));
        Rect = Rect[:2,:];
        Rect = rect.astype(int);

        mask = np.zeros((m,n));
        mask = cv2.fillConvexPoly(mask, np.array([Rect[:,0],Rect[:,1], Rect[:,2], Rect[:,3]]), color = (1,1,1));

        ret,frame = cap.read()
        frame = frame[:,:,::-1];

        sig[0,i+1] = frame[mask*frame[:,:,0]>0,0].mean();
        sig[1,i+1] = frame[mask*frame[:,:,1]>0,1].mean();
        sig[2,i+1] = frame[mask*frame[:,:,2]>0,2].mean();

    return sig
###################################################

def greenSig(vidpath, RGB):

    # Get video parameters
    cap = cv2.VideoCapture(vidpath);
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(300,3))

    # Plot initial G signal
    plt.plot(RGB[2,:])
    plt.title('Initial G sig.')
    plt.show()
    ################################

    # Bandpass filter the signal between 0.8 and 5 Hz
    fs = cap.get(cv2.CAP_PROP_FPS);
    nyq = fs/2;
    lowcut = 0.8;
    highcut = 5;
    low = lowcut/nyq;
    high = highcut/nyq;
    b, a = butter(10, [low, high], btype = 'bandpass');
    G = RGB[1,:];
    G = filtfilt(b,a,G);

    # Plot bandpass filtered G signal
    plt.plot(G)
    plt.title('Bandpass filtered G sig. [0.8, 5] Hz')
    plt.show()
    ###################################

    # SSA decomposition and RC selection
    G = G.reshape(1,-1);
    SSA = ssa(window_size=30);
    G_RCs = SSA.fit_transform(G);
    RC_10 = G_RCs[0:10,:];

    # Plot first 4 RC signals
    fig, ax = plt.subplots(4,1);
    fig.suptitle('Reconstruction Components')
    ax[0].plot(RC_10[0,:])
    ax[1].plot(RC_10[1,:])
    ax[2].plot(RC_10[2,:])
    ax[3].plot(RC_10[3,:])
    plt.show()
    #######################     

    # Calculate FFT of 10 candiates and max peaks
    RC10_FFT = np.zeros(RC_10.shape);
    max_peaks = np.zeros((1,10)).flatten()
    for i in range(10):
        RC10_FFT[i,:] = np.fft.fft(RC_10[i,:]);

    # Plot FFT of first 4 RC signals
    fig, ax = plt.subplots(4,2);
#    fig.suptitle('FFT of Reconstruction Components')
#    ax[0].plot(fft_freq, np.abs(RC10_FFT[0,:]), label="FFT RC 1", color='r')
#    ax[1].plot(fft_freq, np.abs(RC10_FFT[1,:]), label="FFT RC 2", color='r')
#    ax[2].plot(fft_freq, np.abs(RC10_FFT[2,:]), label="FFT RC 3", color='r')
#    ax[3].plot(fft_freq, np.abs(RC10_FFT[3,:]), label="FFT RC 4", color='r')
    fig.suptitle('Magnitude')
#    plt.xlim(-10,10)
#    plt.ylim(-20,20)
#    plt.legend(loc=1)
#    plt.show()
#    ##################################
#    pdb.set_trace();

    # Calculate valid RCs
    tol = 0.05
    valid = np.zeros((1,10)).flatten();
    for i in range(10):
        fft = np.fft.fft(RC_10[i,:])
        fft_freq = np.fft.fftfreq(n=RC_10[i,:].size,d=1/fs)

        # Plot FFT of first 4 RC signals
        if i <= 7:
            ax.flatten()[i].plot(fft_freq, np.abs(RC10_FFT[i,:]), label="FFT RC " + str(i), color='r')
            ax.flatten()[i].set_xlim(-10,10)
            ax.flatten()[i].set_ylim(-20,20)
            ax.flatten()[i].legend(loc=1)
    
        peaks = find_peaks(np.abs(fft), height=2);
        peaks =peaks[0];
        idxInRange = np.where(np.logical_and(fft_freq[peaks]>=0.8, fft_freq[peaks]<=5));
        idxInRange = idxInRange[0];
        peaksInRange = peaks[idxInRange];

        if len(peaks)>0:
            max_peaks[i] = peaks[idxInRange[np.argmax(np.abs(fft[peaksInRange]))]];

        if len(peaksInRange)>=2 and 2*fft_freq[peaksInRange[0]] <= fft_freq[peaksInRange[1]] + tol and 2*fft_freq[peaksInRange[0]] >= fft_freq[peaksInRange[1]] - tol:
            valid[i] = 1;

    plt.show()

    if int(np.sum(valid)) > 0:
        validRC = RC_10[valid.astype(bool),:];
    else:
        validRC = RC_10;

    # Reconstructed Signal
    y = np.sum(validRC,0);

    # Plot RC Signal
    plt.plot(y);
    plt.title('Reconstructed G-sig after SSA')
    plt.show()

    # Apply Overlap Adding (Double check this later)
    winSz = 32;
    sigSz = len(y);
    end_sig = False
    i = 0
    y_overlap = np.zeros(y.shape);
    hanning = np.hanning(winSz)

    # Plot Hanning Window
    plt.plot(hanning);
    plt.show()
    #############################

    while end_sig == False: 
        beginIdx = int(i*(winSz/2));
        endIdx = int(i*(winSz/2)+winSz);
        y_overlap[beginIdx:endIdx] = y[beginIdx:endIdx]*hanning + y_overlap[beginIdx:endIdx];
        i = i + 1;
        if i*(winSz/2)+winSz > sigSz:
            end_sig = True

    endSz = int(winSz - (i*(winSz/2)+winSz - sigSz));
    y_overlap[int(i*(winSz/2)):] = y_overlap[int(i*(winSz/2)):] + hanning[:endSz]*y[int(i*(winSz/2)):];

    # Plot signal after overlap adding
    plt.plot(y_overlap);
    plt.title('Signal after overlap adding')
    plt.show()
    

    # Masking (NOTE: assumes videos are shorter than 10 seconds)
    y_fft = np.fft.fft(y_overlap);
    y_fft_freq = np.fft.fftfreq(n=RC_10[i,:].size,d=1/fs);
    peaks = find_peaks(np.abs(y_fft), height=2);
    peaks =peaks[0];
    idxInRange = np.where(np.logical_and(y_fft_freq[peaks]>=0.8, y_fft_freq[peaks]<=5));
    idxInRange = idxInRange[0];
    peaksInRange = peaks[idxInRange];
    maxIdx = peaks[idxInRange[np.argmax(np.abs(y_fft[peaksInRange]))]]; # Max freq of main sig
    RCInclude = np.zeros((1,10)).astype(np.bool).flatten();
    for i in range(10):
        if y_fft_freq[maxIdx]-0.05 <= fft_freq[int(max_peaks[i])] and fft_freq[int(max_peaks[i])] <= y_fft_freq[maxIdx] + 0.05:
            RCInclude[i] = True;
    g_sig = sum(RC_10[RCInclude,:], 0);

    plt.plot(g_sig);
    plt.title('Final G-channel rPPG signal')
    plt.show()
        
    # Go back and debug this section and function in general

    #plt.subplot(211)
    #plt.plot(fft_freq, np.abs(RC10_FFT[9,:]), label="Real part", color='r')
    #plt.xlim(-10,10)
    #plt.ylim(-20,20)
    #plt.legend(loc=1)
    #plt.title("FFT in Frequency Domain")
    #plt.scatter(fft_freq[peaksInRange], np.abs(fft[peaksInRange]));
    #
    #plt.subplot(212)
    #plt.plot(fft_freq, fft.imag, label="Imaginary part", color='r')
    #plt.legend(loc=1)
    #plt.xlim(-10,10)
    #plt.ylim(-10,10)
    
    plt.show()

    return g_sig
    
print("Importing Video")

vidpath="/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/fram1-original.mov"

#vidpath = '/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/lower_quality/fadg0/sa1-video-fram1.avi'

print("Tracking tracking SURF points using KLT algorithm")
[featMatches, init_ROIs] = getSURFThroughoutVid(vidpath);

print("Tracking ROI and calculating cheek signals")
sig_l, sig_r = cheekSigs2(vidpath, init_ROIs, featMatches);

print("Tracking DLIB Landmarks using KLT algorithm")
[featMatches, ROI] = getAndTrackDlibFeatures(vidpath);

print("Tracking ROI and calculating mid-region signal")
sig_m = cheekSigs(vidpath, ROI, featMatches);

print("Calculating mid-, left-, and right- region green PPG signals")
G_m = greenSig(vidpath, sig_m);
G_l = greenSig(vidpath, sig_l);
G_r = greenSig(vidpath, sig_r);

print("Calculating mid-, left-, right region chrominance PPG signals")
C_m = CHROM(sig_m, 48).flatten();
C_l = CHROM(sig_l, 48).flatten();
C_r = CHROM(sig_r, 48).flatten();

fig, ax = plt.subplots(3,2);
ax[0,0].plot(G_m)
ax[0,0].set_title("G_m")
ax[0,1].plot(C_m)
ax[0,1].set_title("C_m")
ax[1,0].plot(G_l)
ax[1,0].set_title("G_l")
ax[1,1].plot(C_l)
ax[1,1].set_title("C_l")
ax[2,0].plot(G_r)
ax[2,0].set_title("G_r")
ax[2,1].plot(C_r)
ax[2,1].set_title("C_r")
plt.show()


# TODO / DONE
#
# 1) Check signal extraction (Done)
# 2) Chrominance rPPG implementation (Done)
# 3) Mid-region signal extraction (Done)
# 4) Green signal extraction (TODO)

    # Step 1
    #-Bandpass filter the green channel [0.8, 5] Hz

    # Step 2
    #-Reshape the green signal into the Hanekl Matrix
    #-Perform singular value decompostion on the Hankel matrix
    #-Sort the eigentriple in descending order by singular value and reconstruct the top 10 as RC candidates by diagonal averging.
    #-Calculate the FT of each of the 10 candidates
    #-Retain the RC candidates whose two dominant frequencies are such that one is twice the other (i.e. first and second harmonics) (If there are none, then retain them all.)
    #-Sum the RCs to get the signal y

    # Step 3
    #-Select a hanning window of certain size
    #-multiply and shift the hanning window along the signal and add all overlapped signal sections to the output. The shift is half the size of the hanning windowo

    # Step 4
 


# 5) Simplify code and remove redundancies


# Considerations:
# 1) For ROI tracking, consider implementing tracking of ROI using interpolation for non-integer valued pixel locations rather than only transforming the corners of the rectangular ROI.




