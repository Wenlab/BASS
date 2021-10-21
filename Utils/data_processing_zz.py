#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar. Email: gautam.sridhar@icm_institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

#data format library
#numpy
import numpy as np
import numpy.ma as ma
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
#load libraries
import json
import scipy.io as sio
import numpy as np
from scipy.interpolate import splev, splprep,interp1d
import time as T


class boutDef:
    """
    Class for bout objects to be extracted from Zebrazoom
    
    Attributes:
    
    data: Dictionary containing data extracted from Zebrazoom
    maxT: Total number of frames per bout to be considered
    """
    def __init__(self, data, maxT):
        self.numPoiss =  data['FishNumber']
        self.begMove =  data['BoutStart']
        self.endMove =  data['BoutEnd']
        self.boutnum = 0
        self.tailAngle = data['TailAngle_Raw']
        self.posHeadX =  data['HeadX']
        self.posHeadY =  data['HeadY']
        self.rawheading =  data['Heading_raw']
        self.correctedheading =  data['Heading']
        self.posTailXVideoReferential =  data['TailX_VideoReferential']
        self.posTailYVideoReferential =  data['TailY_VideoReferential']
        self.posTailXHeadingReferential = data['TailX_HeadingReferential']
        self.posTailYHeadingReferential=  data['TailY_HeadingReferential']
        self.tailAngleSmoothed=  data['TailAngle_smoothed']
        self.freq =  data['Bend_TimingAbsolute']
        self.freqX=  data['Bend_Timing']
        self.freqY =  data['Bend_Amplitude']
        self.param =  data['param']
        self.posHeadX_int = self.posHeadX
        self.posHeadY_int = self.posHeadY
        self.speed = 0
        self.frequency = 0
        self.amp = 0
        self.nosc = 0
        self.angspeed = 0
        self.deltahead = 0
        self.time = 0 
        self.dist = 0
        self.disp = 0
        self.avgspeed = 0
        self.ispeeds = np.zeros(25)
        self.welltype = 0
        self.filename = 0
        self.wellnum = 0
        self.likelihood = 0
        self.taillength = 0
        self.tailarea = 0
        self.tailpc1 = 0
        self.tailpc2 = 0
        self.tailpc3 = 0
        self.tailangles = np.zeros((maxT,7))
        self.ibi_prev = 0
        self.ibi_next = 0
        self.warning = []


    def calc_bout_posHead_interp(self,seed=42):
        np.random.seed(seed)
        f,u = splprep([self.posHeadX + .1*np.random.randn(len(self.posHeadX)), self.posHeadY + .1*np.random.randn(len(self.posHeadX))],s = 10)
        new_points = splev(u, f)
        return new_points[0], new_points[1]

    def calc_speed(self, fps, px_to_mm):
        """
        Speed in mm/sec
        """
        totaldist = self.calc_dist(px_to_mm)
        totaltime = self.calc_time(fps)
        return totaldist/totaltime

    def calc_ispeed(self, fps, px_to_mm):
        """
        Instantaneous speed in mm/sec
        """
        numps = 6
        ispeeds = np.zeros(25)
        for j in range(min(len(self.posHeadX)-1,25)):
            if j >= len(self.posHeadX):
                ispeeds[j] = 0
            else:
                bXs = np.concatenate((self.posTailXVideoReferential[j][-numps:],[self.posHeadX[j]]))
                bYs = np.concatenate((self.posTailYVideoReferential[j][-numps:],[self.posHeadY[j]]))
                theta = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))
                delx = (self.posHeadY_int[j+1] - self.posHeadY_int[j])*px_to_mm
                dely = (self.posHeadX_int[j+1] - self.posHeadX_int[j])*px_to_mm
                del_ = np.sqrt(delx**2 + dely**2)
                phi = np.arctan2(dely,delx)
                ispeeds[j] = del_*np.cos(theta - phi)*fps
        return ispeeds

    def calc_frequency(self,fps):
        """
        Frequency of oscillations in Hz
        """
        if type(self.freqX) is list:
            if len(self.freqX) > 1:
                return 0.5/(np.mean(np.asarray(self.freqX[1:]) - np.asarray(self.freqX[:-1]))/fps)
            else:
                return 0
    
    def calc_amp(self):
        """
        Amplitude of oscillations
        """
        return np.max(np.abs(self.freqY))*180/np.pi

    def calc_nosc(self):
        """
        Number of oscillations
        """
        if type(self.freq) is list:
            return len(self.freq)
        else:
            return 1.0

    def calc_angspeed(self,fps):
        """
        Mean angular speed in deg/sec
        """
        totaltime = self.calc_time(fps)
        return self.calc_deltahead()/totaltime

    def calc_deltahead(self):
        """
        Heading change in degrees
        """
        numps = 6
        bXs = np.concatenate((self.posTailXVideoReferential[0][-numps:],[self.posHeadX[0]]))
        bYs = np.concatenate((self.posTailYVideoReferential[0][-numps:],[self.posHeadY[0]]))
        slope0 = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))*180/np.pi
        
        bXs = np.concatenate((self.posTailXVideoReferential[-1][-numps:],[self.posHeadX[-1]]))
        bYs = np.concatenate((self.posTailYVideoReferential[-1][-numps:],[self.posHeadY[-1]]))
        slope1 = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))*180/np.pi
        delt = -(slope1 - slope0)
        if delt > 180:
            return 360 - delt
        elif delt < -180:
            return -(360 + delt)
        else:
            return delt

    def calc_time(self,fps):
        """
        Bout time in seconds
        """
        return len(self.posHeadX)/fps

    def calc_dist(self,px_to_mm):
        """
        total distance travelled in mm
        """
        dist1 = 0
        for j in range(len(self.posHeadX)-1):
            dist1 += np.sqrt((self.posHeadX_int[j+1] - self.posHeadX_int[j])**2 + (self.posHeadY_int[j+1] - self.posHeadY_int[j])**2)
        return dist1*px_to_mm

    def calc_disp(self,px_to_mm):
        """
        magnitude of displacement in mm
        """
        disp1 = np.sqrt((self.posHeadX_int[-1] - self.posHeadX_int[0])**2 + (self.posHeadY_int[-1] - self.posHeadY_int[0])**2)
        return disp1*px_to_mm

    def calc_avgspeed(self,fps,px_to_mm):
        """
        Average speed in mm/s
        """
        disp1 = self.calc_disp(px_to_mm)
        return disp1/self.calc_time(fps)

    def calc_taillength(self):
        """
        avg tail length in. mm
        """
        return np.sum(np.abs(np.diff(self.tailAngleSmoothed)))

    def calc_tailarea(self):
        """
        tail integral
        """
        return np.abs(np.sum(self.tailAngleSmoothed))

    def calc_tailangles(self,maxT):
        """
        tailangles for all points
        """
        numps = 3
        
        headx = self.posHeadX
        heady = self.posHeadY
        tailx = self.posTailXVideoReferential
        taily = self.posTailYVideoReferential
    
        tailangles_arr = np.zeros((maxT,7))
        for i in range(min(len(self.posHeadX),tailangles_arr.shape[0])):
            ang = np.arctan2(heady[i] - taily[i][-3],headx[i] - tailx[i][-3])
            for j in range(tailangles_arr.shape[1]):
                ang2 = np.arctan2(heady[i] - taily[i][j],headx[i] - tailx[i][j])
                delang = ang2 - ang
                if np.abs(delang) < np.pi:
                    tailangles_arr[i,j] = delang
                elif delang > np.pi:
                    tailangles_arr[i,j] = delang - 2*np.pi
                elif delang < -np.pi:
                    tailangles_arr[i,j] = 2*np.pi + delang
                #print(i,j,ang,ang2,tailangles_arr[i,j])
        return tailangles_arr

    def calc_heading(self):
        """
        calculate heading
        """
        return np.arctan2(self.posHeadY_int[-1] - self.posHeadY_int[-2],self.posHeadX_int[-1] - self.posHeadX_int[-2])*180.0/np.pi


def reject(b):
    """
    Reject recordings misclassified as bouts. Returns a binary value.

    Parameters:
    b: a bout object
    """

    if b.time < 0.04 or b.time > 1.2:
        return True
    if b.dist > 25 or b.dist < 0.0 :
        return True
    if b.speed > 50 or b.speed < 1:
        return True
    if np.abs(b.deltahead) > 180:
        return True
    return False


def get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT,seed,exp_type):
    """
    Extract bouts from Zebrazoom .txt file output.

    Parameters:
    foldername: Name of the folder where data is stored
    filenames: Names of the files to extract bouts from
    welltypes: Incase multiple well are used, an array to indicate which wells to keep
    px_to_mm: size of the pixel in mm for the recording used
    fps: frequency of the recording
    maxT: max number of frames of the bout to keep
    seed: random seed to be set for dataextraction
    exp_type: If recording contains multiple experiments, the index of the experiment to extract from

    Returns:
    bouts_all: All bouts extracted as a bout objects
    """
    bouts_all = []

    for k,f in enumerate(filenames):
        with open(foldername + f + '/results_' +f+'.txt') as name:
            data = json.load(name)
            welltype = welltypes[k]
            numrejects = 0
            numaccepts = 0

            for j in range(len(welltype)):
                if exp_type != welltype[j]:
                    continue

                if len(data['wellPoissMouv'][j][0]) == 0:
                    continue
                numbouts = len(data['wellPoissMouv'][j][0])
                for i in range(numbouts):
                    bouts_temp = data['wellPoissMouv'][j][0][i]
                    b = boutDef(bouts_temp,maxT)
                    b.boutnum = i
                    b.posHeadX_int,b.posHeadY_int = b.calc_bout_posHead_interp(seed)
                    b.speed = b.calc_speed(fps,px_to_mm)
                    b.frequency = b.calc_frequency(fps)
                    b.amp = b.calc_amp()
                    b.nosc = b.calc_nosc()
                    b.angspeed = b.calc_angspeed(fps)
                    b.deltahead = b.calc_deltahead()
                    b.time = b.calc_time(fps)
                    b.dist = b.calc_dist(px_to_mm)
                    b.disp = b.calc_disp(px_to_mm)
                    b.avgspeed = b.calc_avgspeed(fps,px_to_mm)
                    b.welltype = welltype[j]
                    b.filename = f
                    b.taillength = b.calc_taillength()
                    b.tailarea = b.calc_tailarea()
                    b.tailangles = b.calc_tailangles(maxT)
                    b.ispeeds = b.calc_ispeed(fps,px_to_mm)
                    b.wellnum = j
                    if i < numbouts-1:
                        bouts_temp_next = data['wellPoissMouv'][j][0][i+1]
                        b_next = boutDef(bouts_temp_next,maxT)
                        b.ibi_next = (b_next.begMove - b.endMove)/fps
                    if i > 0:
                        bouts_temp_prev = data['wellPoissMouv'][j][0][i-1]
                        b_prev = boutDef(bouts_temp_prev,maxT)
                        b.ibi_prev = (b.begMove - b_prev.endMove)/fps

                    if reject(b):
                        numrejects += 1
                        continue
                    else:
                        numaccepts += 1
                    bouts_all += [b]
            print(foldername + f + '/results_' +f+'.txt',numrejects, numaccepts, 1.0*numrejects/(numaccepts + numrejects + 1))
    return bouts_all

def pool_data(bout_dataset):
    """
    Pool together bout information from a dataset and store in a dictionary
    """
    numbouts = len(bout_dataset)
    data_collected = {'speeds':np.zeros(numbouts), 'frequencys':np.zeros(numbouts),'amps':np.zeros(numbouts),'noscs':np.zeros(numbouts),'angspeeds':np.zeros(numbouts),'deltaheads':np.zeros(numbouts),'dists':np.zeros(numbouts)\
                      ,'times':np.zeros(numbouts),'avgspeeds':np.zeros(numbouts),'disps':np.zeros(numbouts),'tailareas':np.zeros(numbouts)}

    for i,b in enumerate(bout_dataset):
        data_collected['speeds'][i] = b.speed
        data_collected['frequencys'][i] = b.frequency
        data_collected['amps'][i] = b.amp
        data_collected['noscs'][i] = b.nosc
        data_collected['angspeeds'][i] = b.angspeed
        data_collected['deltaheads'][i] = b.deltahead
        data_collected['times'][i] = b.time
        data_collected['dists'][i] = b.dist
        data_collected['disps'][i] = b.disp
        data_collected['avgspeeds'][i] = b.avgspeed
        data_collected['tailareas'][i] = b.tailarea
    return data_collected

def collect_two_consecutive_bouts(bout_dataset, fps, px_to_mm):
    """
    Collect two bout info
    """
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        if b_next.filename == currfilename and b_next.wellnum == currwellnum:
            ibi = (b_next.begMove - b.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            if ibi > 10 or dist_bout > 4:
                continue
            else:
                collection += [[b,b_next,ibi]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_three_consecutive_bouts(bout_dataset, fps, px_to_mm):
    """
    Collect three bout info
    """
    collection = []
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-2]):
        b_next = bout_dataset[i+1]
        b_nextnext = bout_dataset[i+2]
        if (b_next.filename == currfilename and b_next.wellnum == currwellnum) and (b_nextnext.filename == currfilename and b_nextnext.wellnum == currwellnum):
            ibi = (b_next.begMove - b.endMove)/fps
            ibi2 = (b_nextnext.begMove - b_next.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            dist_bout2 = px_to_mm*np.sqrt((b_nextnext.posHeadX[0] - b_next.posHeadX[-1])**2 + (b_nextnext.posHeadY[0] - b_next.posHeadY[-1])**2)

            if (ibi > 10 or dist_bout > 4) or (ibi2 > 10 or dist_bout2 > 4):
                continue
            else:
                collection += [[b,b_next,b_nextnext]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_four_consecutive_bouts(bout_dataset, fps, px_to_mm):
    """
    Collect four bout info
    """
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-3]):
        b_next = bout_dataset[i+1]
        b_nextnext = bout_dataset[i+2]
        b_nextnextnext = bout_dataset[i+3]
        if (b_next.filename == currfilename and b_next.wellnum == currwellnum) and (b_nextnext.filename == currfilename and b_nextnext.wellnum == currwellnum) and (b_nextnextnext.filename == currfilename and b_nextnextnext.wellnum == currwellnum):
            ibi = (b_next.begMove - b.endMove)/fps
            ibi2 = (b_nextnext.begMove - b_next.endMove)/fps
            ibi3 = (b_nextnextnext.begMove - b_nextnext.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            dist_bout2 = px_to_mm*np.sqrt((b_nextnext.posHeadX[0] - b_next.posHeadX[-1])**2 + (b_nextnext.posHeadY[0] - b_next.posHeadY[-1])**2)
            dist_bout3 = px_to_mm*np.sqrt((b_nextnextnext.posHeadX[0] - b_nextnext.posHeadX[-1])**2 + (b_nextnextnext.posHeadY[0] - b_nextnext.posHeadY[-1])**2)


            if (ibi > 10 or dist_bout > 4) or (ibi2 > 10 or dist_bout2 > 4) or (ibi3 > 10 or dist_bout3 > 4):
                continue
            else:
                collection += [[b,b_next,b_nextnext, b_nextnextnext]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_trajectories(bout_dataset, fps, px_to_mm):
    """
    Collect continuous set of bouts
    """
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    currtraj = [bout_dataset[0]]
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        if b_next.filename == currfilename and b_next.wellnum == currwellnum:
            currtraj += [b_next]
        else:
            if len(currtraj) > 30:
                collection += [currtraj]
            currtraj = [b_next]
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_trajectories_nospacings(bout_dataset, fps, px_to_mm):
    """
    Collect continuous set of bouts with no spacings between bouts
    """
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    currtraj = [bout_dataset[0]]
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        ibi = (b_next.begMove - b.endMove)/fps
        dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
        if b_next.filename == currfilename and b_next.wellnum == currwellnum and ibi < 5 and dist_bout < 4:
            currtraj += [b_next]
        else:
            if len(currtraj) > 30:
                collection += [currtraj]
            currtraj = [b_next]
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_data_hmm(trajs_nospacings):
    """
    Collect bout information required to perform GMM clustering and to run BASS.
    
    Parameters:
    trajs_nospacings: collection of bouts filtered to remove freezes (of the fish)

    Returns:
    data_hmm: Information of the bout. Size - nbouts x nfeatures
    lengths: array containing the size of each selected trajectory
    """

    nsamples = 0
    for t in trajs_nospacings:
        nsamples += len(t)

    data_hmm = np.zeros((nsamples,6))
    lengths = np.zeros(len(trajs_nospacings), dtype = int)
    for i,t in enumerate(trajs_nospacings):
        lengths[i] = len(t)
        for j in range(len(t)):
            data_hmm[np.sum(lengths[:i])+j][0] = np.abs(t[j].deltahead)
            data_hmm[np.sum(lengths[:i])+j][1] = t[j].speed
            data_hmm[np.sum(lengths[:i])+j][2] = t[j].taillength
            data_hmm[np.sum(lengths[:i])+j][3] = t[j].tailpc1
            data_hmm[np.sum(lengths[:i])+j][4] = t[j].tailpc2
            data_hmm[np.sum(lengths[:i])+j][5] = t[j].tailpc3

    return data_hmm, lengths

def collect_tailangles_hmm(trajs_nospacings,maxT,npoints):
    """
    Collect bout information required to perform GMM clustering and to run BASS.
    
    Parameters:
    trajs_nospacings: collection of bouts filtered to remove freezes (of the fish)

    Returns:
    tailangles_hmm: Information of the raw tail angles. Size - nbouts x nfeatures
    lengths: array containing the size of each selected trajectory

    """

    nsamples = 0
    for t in trajs_nospacings:
        nsamples += len(t)

    tailangles_hmm = np.zeros((nsamples,maxT,npoints))
    lengths = np.zeros(len(trajs_nospacings), dtype = int)
    for i,t in enumerate(trajs_nospacings):
        lengths[i] = len(t)
        for j in range(len(t)):
            tailangles_hmm[np.sum(lengths[:i])+j] = t[j].tailangles

    return tailangles_hmm, lengths


def collect_data_hmm_other(trajs_nospacings):
    """
    Collect non essential bout information..
    
    Parameters:
    trajs_nospacings: collection of bouts filtered to remove freezes (of the fish)

    Returns:
    data_hmm: Information of the bout. Size - nbouts x nfeatures
    lengths: array containing the size of each selected trajectory
    """
    nsamples = 0
    for t in trajs_nospacings:
        nsamples += len(t)

    data_hmm = np.zeros((nsamples,4))
    lengths = np.zeros(len(trajs_nospacings), dtype = int)
    for i,t in enumerate(trajs_nospacings):
        lengths[i] = len(t)
        for j in range(len(t)):
            data_hmm[np.sum(lengths[:i])+j][0] = np.mean(t[j].posHeadX)
            data_hmm[np.sum(lengths[:i])+j][1] = np.mean(t[j].posHeadY)
            data_hmm[np.sum(lengths[:i])+j][2] = t[j].dist
            data_hmm[np.sum(lengths[:i])+j][3] = t[j].angspeed

    return data_hmm, lengths

def collect_trajectory_hmm(traj_nospacings):
    data_hmm = np.zeros((len(traj_nospacings),4))
    for j in range(len(traj_nospacings)):
        data_hmm[j][0] = np.abs(traj_nospacings[j].angspeed)*1e-3
        data_hmm[j][1] = traj_nospacings[j].speed
        data_hmm[j][2] = traj_nospacings[j].time
        data_hmm[j][3] = traj_nospacings[j].amp

    return data_hmm

def get_tailangles(dataset, maxT, npoints):
    """
    Return tail angles extracted from the dataset
    """

    tailangles_all = np.zeros((len(dataset),maxT*npoints))
    for i,b in enumerate(dataset):
        tailangles_all[i] = np.abs(b.tailangles[:maxT,:].flatten())
    return tailangles_all

def update_tail_pcas(bouts,pcs):
    """
    Update bout object with extracted pc's
    """
    for i,b in enumerate(bouts):
        b.tailpc1 = pcs[i,0]
        b.tailpc2 = pcs[i,1]
        b.tailpc3 = pcs[i,2]
        b.tailpc4 = pcs[i,3]
    return bouts

