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
    m = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize DLIB face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/zlazri/Documents/ENEE/Research/FakeCatcher/FakeCatcher/shape_predictor_68_face_landmarks.dat");

    # Define cheek and mid region signals
    sig_l = np.zeros((3,N_frames));
    sig_r = np.zeros((3,N_frames));
    sig_m = np.zeros((3,N_frames));

    for fnum in range(N_frames):

        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Detect face in the frame
        rects = detector(old_gray,1);

        for (i,r) in enumerate(rects):
            rect = r
    
        shape = predictor(old_gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Plot Face rectangle
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # show the face number
        cv2.putText(old_frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Plot DLIB facial landmarks
        for number, (x, y) in enumerate(shape):
            text = str(number);
            text_origin = (x - 5, y - 5)    
            cv2.circle(old_frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(old_frame, text, text_origin, cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 0, 0));
    
        # Define ROI of left cheek

        # First get all of the landmarks that will be used to generate the bounding lines of the ROI
        line1End1 = shape[2,:];
        line1End2 = shape[39,:];
        line2End1 = shape[36,:];
        line2End2 = shape[30,:];
        line3End1 = shape[27,:];
        line3End2 = shape[4,:];
        line4End1 = shape[1,:];
        line4End2 = shape[6,:];

        # Find the bounding lines
        m = (line1End1[1] - line1End2[1])/(line1End1[0]-line1End2[0]) #slope
        b = line1End1[1]-m*line1End1[0];
        line1 = (m,b);
        m = (line2End1[1] - line2End2[1])/(line2End1[0]-line2End2[0]) #slope
        b = line2End1[1]-m*line2End1[0];
        line2 = (m,b);
        m = (line3End1[1] - line3End2[1])/(line3End1[0]-line3End2[0]) #slope
        b = line3End1[1]-m*line3End1[0];
        line3 = (m,b);
        m = (line4End1[1] - line4End2[1])/(line4End1[0]-line4End2[0]) #slope
        b = line4End1[1]-m*line4End1[0];
        line4 = (m,b);

        # Get corners of convex polygon that defines ROI
        x1 = (line2[1]-line1[1])/(line1[0]-line2[0]);
        y1 = x1*line1[0] + line1[1];
        pt1_l = (int(x1),int(y1));

        x2 = (line3[1]-line2[1])/(line2[0]-line3[0]);
        y2 = x2*line2[0] + line2[1];
        pt2_l = (int(x2),int(y2));

        x3 = (line4[1]-line3[1])/(line3[0]-line4[0]);
        y3 = x3*line3[0] + line3[1];
        pt3_l = (int(x3),int(y3));
    
        x4 = (line1[1]-line4[1])/(line4[0]-line1[0]);
        y4 = x4*line4[0] + line4[1];
        pt4_l = (int(x4),int(y4));
    
        # Generate a mask for the ROI
        m,n,k = old_frame.shape
        mask_l = np.zeros((m,n));
        mask_l = cv2.fillConvexPoly(mask_l, np.array([pt1_l, pt2_l, pt3_l, pt4_l]), color=(1,1,1));
    
        # Define ROI of right cheek
    
        # First get all of the landmarks that will be used to generate the bounding lines of the ROI
        line1End1 = shape[14,:];
        line1End2 = shape[42,:];
        line2End1 = shape[45,:];
        line2End2 = shape[30,:];
        line3End1 = shape[27,:];
        line3End2 = shape[12,:];
        line4End1 = shape[15,:];
        line4End2 = shape[10,:];
    
        # Find the bounding lines
        m = (line1End1[1] - line1End2[1])/(line1End1[0]-line1End2[0]) #slope
        b = line1End1[1]-m*line1End1[0];
        line1 = (m,b);
        m = (line2End1[1] - line2End2[1])/(line2End1[0]-line2End2[0]) #slope
        b = line2End1[1]-m*line2End1[0];
        line2 = (m,b);
        m = (line3End1[1] - line3End2[1])/(line3End1[0]-line3End2[0]) #slope
        b = line3End1[1]-m*line3End1[0];
        line3 = (m,b);
        m = (line4End1[1] - line4End2[1])/(line4End1[0]-line4End2[0]) #slope
        b = line4End1[1]-m*line4End1[0];
        line4 = (m,b);
    
        # Get corners of convex polygon that defines ROI
        x1 = (line2[1]-line1[1])/(line1[0]-line2[0]);
        y1 = x1*line1[0] + line1[1];
        pt1_r = (int(x1),int(y1));
    
        x2 = (line3[1]-line2[1])/(line2[0]-line3[0]);
        y2 = x2*line2[0] + line2[1];
        pt2_r = (int(x2),int(y2));
    
        x3 = (line4[1]-line3[1])/(line3[0]-line4[0]);
        y3 = x3*line3[0] + line3[1];
        pt3_r = (int(x3),int(y3));
    
        x4 = (line1[1]-line4[1])/(line4[0]-line1[0]);
        y4 = x4*line4[0] + line4[1];
        pt4_r = (int(x4),int(y4));

        # Define ROI of mid region

        # First get all of the landmarks that will be used to generate the bounding lines of the ROI
        line1End1 = shape[39,:];
        line1End2 = shape[7,:];
        line2End1 = shape[39,:];
        line2End2 = shape[42,:];
        line3End1 = shape[42,:];
        line3End2 = shape[9,:];
        line4End1 = shape[13,:];
        line4End2 = shape[3,:];

        m = (line3End1[1] - line3End2[1])/(line3End1[0]-line3End2[0]) #slope
        b = line3End1[1]-m*line3End1[0];
        line3 = (m,b);

        # Find the bounding lines
        m = (line1End1[1] - line1End2[1])/(line1End1[0]-line1End2[0]) #slope
        b = line1End1[1]-m*line1End1[0];
        line1 = (m,b);
        m = (line2End1[1] - line2End2[1])/(line2End1[0]-line2End2[0]) #slope
        b = line2End1[1]-m*line2End1[0];
        line2 = (m,b);
        m = (line3End1[1] - line3End2[1])/(line3End1[0]-line3End2[0]) #slope
        b = line3End1[1]-m*line3End1[0];
        line3 = (m,b);
        m = (line4End1[1] - line4End2[1])/(line4End1[0]-line4End2[0]) #slope
        b = line4End1[1]-m*line4End1[0];
        line4 = (m,b);

        # Deals with issue if one side of convex polygon is vertical line
        if line1[0] == np.inf or line1[0] == -np.inf or line1[1] == np.inf or line1[1] == -np.inf or line3[0] == np.inf or line3[0] == -np.inf or line3[1] == np.inf or line3[1] == -np.inf:

            (x1,y1) = shape[39];
            (x2,y2) = shape[42];
            (x3,y3) = (x2, line4[0]*x2+line4[1]);
            (x4,y4) = (x1, line4[0]*x1+line4[1]);
            pt1_m = (int(x1),int(y1))
            pt2_m = (int(x2),int(y2))
            pt3_m = (int(x3),int(y3))
            pt4_m = (int(x4),int(y4))

        else:

            # Get corners of convex polygon that defines ROI
            x1 = (line2[1]-line1[1])/(line1[0]-line2[0]);
            y1 = x1*line1[0] + line1[1];
    
            pt1_m = (int(x1),int(y1));
    
            x2 = (line3[1]-line2[1])/(line2[0]-line3[0]);
            y2 = x2*line2[0] + line2[1];

            pt2_m = (int(x2),int(y2));
    
            x3 = (line4[1]-line3[1])/(line3[0]-line4[0]);
            y3 = x3*line3[0] + line3[1];
            pt3_m = (int(x3),int(y3));
        
            x4 = (line1[1]-line4[1])/(line4[0]-line1[0]);
            y4 = x4*line4[0] + line4[1];
            pt4_m = (int(x4),int(y4));
        
        # Generate a mask for the ROI
        m,n,k = old_frame.shape
        mask_m = np.zeros((m,n));
        mask_m = cv2.fillConvexPoly(mask_m, np.array([pt1_m, pt2_m, pt3_m, pt4_m]), color=(1,1,1));

        m,n,k = old_frame.shape
        mask_m = np.zeros((m,n));
        mask_m = cv2.fillConvexPoly(mask_m, np.array([pt1_m, pt2_m, pt3_m, pt4_m]), color=(1,1,1));
        mask_r = np.zeros((m,n));
        mask_r = cv2.fillConvexPoly(mask_r, np.array([pt1_r, pt2_r, pt3_r, pt4_r]), color=(1,1,1));
        mask_l = np.zeros((m,n));
        mask_l = cv2.fillConvexPoly(mask_l, np.array([pt1_l, pt2_l, pt3_l, pt4_l]), color=(1,1,1));

        Imask_l = -1*(mask_l-1)
        Imask_r = -1*(mask_r-1)
        Imask_m = -1*(mask_m-1)
        img = np.zeros(old_frame.shape)
        for l in range(k):
            img[:,:,l] = np.array(old_frame[:,:,l]*(Imask_l*Imask_r*Imask_m), dtype = np.int)

        img = np.array(img, dtype=np.uint8)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
        # Left cheek
        ROI = (mask_l*old_frame[:,:,2])
        sig_l[0,fnum] = ROI[ROI>0].mean() #r
        ROI = mask_l*old_frame[:,:,1]
        sig_l[1,fnum] = ROI[ROI>0].mean() #g
        ROI = mask_l*old_frame[:,:,0]
        sig_l[2,fnum] = ROI[ROI>0].mean() #b

        # Right cheek
        ROI = (mask_r*old_frame[:,:,2])
        sig_r[0,fnum] = ROI[ROI>0].mean() #r
        ROI = mask_r*old_frame[:,:,1]
        sig_r[1,fnum] = ROI[ROI>0].mean() #g
        ROI = mask_r*old_frame[:,:,0]
        sig_r[2,fnum] = ROI[ROI>0].mean() #b

        #Middle Region
        ROI = (mask_m*old_frame[:,:,2])
        sig_m[0,fnum] = ROI[ROI>0].mean() #r
        ROI = mask_m*old_frame[:,:,1]
        sig_m[1,fnum] = ROI[ROI>0].mean() #g
        ROI = mask_m*old_frame[:,:,0]
        sig_m[2,fnum] = ROI[ROI>0].mean() #b

    cv2.destroyAllWindows()
    cap.release()     
        
    return sig_l, sig_r, sig_m


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

print("Obtaining 3 ROIs")
[sig_l, sig_r, sig_m] = getAndTrackDlibFeatures(vidpath);

print("Calculating mid-, left-, and right- region green PPG signals")
G_m = greenSig(vidpath, sig_m);
G_l = greenSig(vidpath, sig_l);
G_r = greenSig(vidpath, sig_r);

print("Calculating mid-, left-, right region chrominance PPG signals")
C_m = CHROM(sig_m, 48).flatten();
C_l = CHROM(sig_l, 48).flatten();
C_r = CHROM(sig_r, 48).flatten();

pdb.set_trace()
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




