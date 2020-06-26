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
from scipy.signal import butter, filtfilt
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
    mask, init_ROI1, init_ROI2 = parms;
        
    # Detect SURF keypoints
    surf = cv2.xfeatures2d.SURF_create();
    keypoints, descriptors = surf.detectAndCompute(old_gray,mask);
    keypoints = [keypoints[idx].pt for idx in range(0, len(keypoints))]
    kk = np.array([[keypoints[0][0],keypoints[0][1]]]);
    for i in range(len(keypoints)-1):
        kk = np.vstack((kk, keypoints[i+1]));
    keypoints = kk;

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
    sig_r[0,0]= frame[y_r:y_r+h_r, x_r:x_r+w_r,0].mean();
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
        sig_r[0,i+1]= frame[mask_r*frame[:,:,0]>0,0].mean();
        sig_l[1,i+1] = frame[mask_l*frame[:,:,1]>0,1].mean();
        sig_r[1,i+1] = frame[mask_r*frame[:,:,1]>0,1].mean();
        sig_l[2,i+1] = frame[mask_l*frame[:,:,2]>0,2].mean();
        sig_r[2,i+1] = frame[mask_r*frame[:,:,2]>0,2].mean();

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
    #cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # show the face number
    #cv2.putText(old_frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

        #cv2.circle(old_frame, (x, y), 1, (0, 0, 255), -1)
        #cv2.putText(old_frame, text, text_origin, cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 0, 0));
    
    y1 = np.max([y_up1, y_up2]);

    # Define ROI

    x = x1
    y = y1;
    w = x2-x1;
    h = y2-y1;

    ROI = [x, y, w, h];

    #cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 0, 255), 2);
    #cv2.imshow("Output", old_frame)
    #cv2.waitKey(0)

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

    pdb.set_trace()

    # SSA decomposition and RC selection
    G = G.reshape(1,-1);
    SSA = ssa(window_size=30);
    G_RCs = SSA.fit_transform(G);
    

    # SSA decomposition and RC selection
    #SSA(sig);
    #H = hankel(G);           # Hankel matrix
    #[U, S, V] = svd(H);      # Singular value decomposition
    #X = np.zeros(H.shape);
    ##for i in range(10):      # Generate resultant matrix
    ##    X = X + S[i,i]*U[:,i]@V[:,i].T
    #X_elem = np.array( [S[i] * np.outer(U[:,i], V[:,i]) for i in range(0,10)] )
    #
    #n = 10 # In case d is less than 12 for the toy series. Say, if we were to exclude the noise component...
    #for i in range(n):
    #    plt.subplot(4,4,i+1)
    #    title = "$\mathbf{X}_{" + str(i) + "}$"
    #    plot_2d(X_elem[i], title)
    #plt.tight_layout()
    #
    #fig = plt.subplot()
    #for i in range(n):
    #    F_i = X_to_TS(X_elem[i])
    #    np.linspace(1,len(G), len(G));
    #    pdb.set_trace()
    #    fig.axes.plot(t, F_i, lw=2)
    #
    #fig.axes.plot(t, F, alpha=1, lw=1)
    #fig.set_xlabel("$t$")
    #fig.set_ylabel(r"$\tilde{F}_i(t)$")
    #legend = [r"$\tilde{F}_{%s}$" %i for i in range(n)] + ["$F$"]
    #fig.set_title("The First 12 Components of the Toy Time Series")
    #fig.legend(legend, loc=(1.05,0.1));
    #
    # Implements green signal paper from DeepFake
    #pdb.set_trace()

    return g_sig

    
    
print("Importing Video")

#vidpath="/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/fram1-original.mov"

vidpath = '/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/DeepfakeTIMIT/lower_quality/fadg0/sa1-video-fram1.avi'

print("Tracking DLIB Landmarks using KLT algorithm")
[featMatches, ROI] = getAndTrackDlibFeatures(vidpath);

print("Tracking ROI and calculating mid-region signal")
sig_m = cheekSigs(vidpath, ROI, featMatches);

print("Getting PPG signal")
C_g = greenSig(vidpath, sig_m);

print("Calculating mid-region chrominance PPG signals")
C_m = CHROM(sig_m, 48).flatten();

pdb.set_trace()


print("Tracking tracking SURF points using KLT algorithm")
[featMatches, init_ROIs] =getSURFThroughoutVid(vidpath);

print("Tracking ROI and calculating cheek signals")
sig_l, sig_r = cheekSigs2(vidpath, init_ROIs, featMatches);

print("Calculating cheek chrominance PPG signals")
C_l = CHROM(sig_l, 48).flatten();
C_r = CHROM(sig_r, 48).flatten();
pdb.set_trace();


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







