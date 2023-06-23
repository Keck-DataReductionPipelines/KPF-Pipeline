# Import packages
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import pandas as pd
import numpy as np
from astropy.time import Time
from scipy.optimize import minimize
from astropy.stats import mad_std
from scipy import special
import copy
import warnings


EM_gain = 1.48424 # Gain of CCD in KPF Exposure Meter (e-/ADU)
EMReadCorrection = 96  # additional time (ms) added to exposures for readout photon collection.

###WAVE_1 default segments bins
orderMin=np.array([4459.46696177, 4490.29501275, 4523.38675819, 4557.05458779,
       4591.26029616, 4625.98519778, 4661.22642997, 4697.11865509,
       4733.56137187, 4770.31038839, 4807.84197293, 4845.90169677,
       4884.65683057, 4923.91622418, 4963.86011921, 5004.47254332,
       5045.79771988, 5087.68410402, 5130.39668547, 5173.85895624,
       5218.0827394 , 5262.8539607 , 5308.65749705, 5355.05989347,
       5402.31336038, 5450.51645493, 5499.54766301, 5549.39060038,
       5600.26054301, 5652.01307975, 5704.54131982, 5758.20657154,
       5813.07251858, 5868.8604146 , 5925.74467262,5985.21722326, 
       6044.23181683, 6104.79400901, 6166.29736437,
       6228.98022399, 6293.22287209, 6358.73248561, 6425.46377452,
       6493.80546354, 6563.61188956, 6634.85903367, 6707.8117821 ,
       6782.16158887, 6857.94357206, 6936.17323281, 7015.75642862,
       7097.28080589, 7181.64784359, 7267.55675873, 7353.25710029,
       7443.11069118, 7533.76702707, 7628.84222384, 7725.98978929,
       7824.27615875, 7925.81232897, 8029.81799816, 8136.79494917,
       8246.56878874, 8359.54813322, 8475.53689391, 8594.53539206])


orderMax=np.array([4510.15718798, 4544.63245407, 4578.0331548 , 4612.13968987,
       4646.70151981, 4681.92771002, 4717.59588627, 4753.90773235,
       4790.47824322, 4828.03963219, 4865.8648726 , 4904.51583772,
       4943.66166133, 4983.53245343, 5023.96184046, 5065.01336326,
       5106.61429312, 5149.34071225, 5192.55762933, 5236.30737262,
       5280.90009451, 5326.5738249 , 5372.68734032, 5419.79036013,
       5467.76648802, 5516.60122831, 5565.92850089, 5616.54164345,
       5667.96290056, 5719.99892728, 5773.68503314, 5828.2036728 ,
       5883.48202188, 5939.98268543, 5997.58096177,6057.46300901, 
       6117.72543507, 6178.68921283, 6240.91404184,
       6304.829808  , 6369.70226009, 6436.22115619, 6503.92906851,
       6572.83282232, 6643.28546122, 6715.69211603, 6789.40058355,
       6864.40729093, 6944.00954099, 7020.59374494, 7101.06834923,
       7183.65614335, 7266.95273573, 7354.01275351, 7444.63098693,
       7534.06968023, 7629.30296739, 7721.82153409, 7819.79325259,
       7920.28129542, 8021.63508161, 8129.63040337, 8236.08025765,
       8347.62337351, 8461.8838267 , 8578.75255777, 8700.32041348])

class MidpointPhotonArrival:
    """Midpoint photon arrival time for barycentric correction.
    
    This module defines class 'MidpointPhotonArrival' and methods to calculate the chromatic midpoint arrival time
    from the exposure meter output.
    
    Attributes:
        df_EM (pandas.DataFrame): Table containing the L0 exposure meter output. 
        start (datetime): The start time of the science exposure.
        finish (datetime): The end time of the science exposure.
        orderMid(numpy.Array): An array the wavelengths (angstroms) representing the middle wavelength value of each segment.
        midPhoto(datetime): An array of estimated arrival times for the midpoint photon for each order.
        segmentMin(numpy.Array): An array representing the minimum of the segment wavelengths (angstroms).
        segmentMax(numpy.Array): An array representing the maximum of the segment wavelengths (angstroms). 
        clip (boolean): This switch turns on (True) and off (False) the sigma clipping feature, which interpolates over
        	outliers. Defaults to True.

    

    Notes:
        Currently, each wavelength is calculated independently.
    """
    
    def __init__(self, df_EM=None, start=None,finish=None,orderMid=None,midPhoto=None, segmentMin=None,segmentMax=None,clip=True):

        if df_EM is None:
            raise Exception("df_EM has to be specified.")
        else:
            self.df_EM=df_EM
        
        
        if (start is None) | (finish is None) :
            raise Exception("start and finish has to be specified.")
        else:
            self.start=start
            self.finish=finish
            
            
        if (start is None) | (finish is None) :
            raise Exception("start and finish has to be specified.")
        else:
            self.start=start
            self.finish=finish
            
        self.clip=clip
        self.orderMid=orderMid
        self.midPhoto=midPhoto
        self.segmentMin=segmentMin
        self.segmentMax=segmentMax
        self.orderMid=orderMid
        self.midPhoto=midPhoto

 

    def sigmaClip(self,exposures):
        """This function replaces outliers that significantly deviate from the nearest neighbor flux values with the average of the neighboring values. 
    
        Args:
            exposures (float): An array of exposure meter flux values ordered chronologically.     
                    
        Returns:
            float: A chronologically ordered array of flux values after replacing outliers.
        """

        ###Look for three sigma outliers and interpolated over those exposures
        deviationVal=np.ones(len(exposures)) ####Deviations
        middleVal=np.ones(len(exposures)) ### Interpolated Values
        madStd=0 ####Sigma Start
        thresSigma=3*np.sqrt(2)*special.erfcinv(1/len(exposures)) ###Sigma Threshold
    
        q=0
    
        #### keep re-running the process until no threshold crossings remain.
        while np.max(abs(deviationVal))>thresSigma*madStd:
        
            q+=1
            if q>len(exposures)/4:
                ######Limit the number of crossings to less than a fourth of the exposure length
                break
        
            for i in range(len(exposures)):
                ###Calculate the deviations using average of adjucent points
                if i==0:
                    middleVal[i]=(exposures[i+1]+exposures[i+2])/2
                    deviationVal[i]=min([(exposures[i]-middleVal[i]),(exposures[i]-exposures[i+1])],key=abs)
                    middleVal[i]=(deviationVal[i]-exposures[i])*-1
                elif i==len(exposures)-1:
                    middleVal[i]=(exposures[i-1]+exposures[i-2])/2
                    deviationVal[i]=min([(exposures[i]-middleVal[i]),(exposures[i]-exposures[i-1])],key=abs)
                    middleVal[i]=(deviationVal[i]-exposures[i])*-1
                else:
                    middleVal[i]=(exposures[i-1]+exposures[i+1])/2
                    deviationVal[i]=min([(exposures[i]-middleVal[i]),(exposures[i]-exposures[i-1]),(exposures[i]-exposures[i+1])],key=abs)
        
        
            ### Update threshold       
            madStd=mad_std(deviationVal)
        
            ###Replace exposures that cross threshold
            if np.max(abs(deviationVal))>thresSigma*madStd:
                inx=((np.where(abs(deviationVal)==np.max(abs(deviationVal))))[0][0])
                num_neighbor=3    
  
                left = exposures[:inx][-num_neighbor:]
                right= exposures[inx+1:num_neighbor+inx+1]
                neighborArray=np.append(left,[exposures[inx]])
                neighborArray=np.append(neighborArray,right)
            
                #Make sure the event is not a platue
                maxRange,minRange=np.max(neighborArray),np.min(neighborArray)
                if (exposures[inx]==maxRange) | (exposures[inx]==minRange):
                    exposures[inx]=middleVal[inx]
                else:
                    break
            
        return(exposures)




    def CalcCDF(self,x,exposures,expBeg,expEnd):
        """Calculates the photon cumulative distribution function at a given point in time (x).
    
        Args:
            x (datetime): The end time for the CDF calculation.
            exposures (float): An array of exposure meter flux values ordered chronologically.
            expBeg (datetime): An array of times that mark the begining of the exposure meter inetgrations.
            expEnd (datetime): An array of times that indicate the end of the exposure meter inetgrations.      
                    
        Returns:
            datetime: The estimated number of photons recieved up to time x.

        Notes:
            It is essential that exposures, expBeg, and expEnd all be arrays of the same length.            
        """
    
        cdfTot=0
        i=0
    
        while x > expEnd[i]:
        
            cdfTot+=exposures[i]
        
        
        ####Interpolation between exposures
            if i>1:
                rateRight=exposures[i]/((expEnd[i]-expBeg[i]).astype(float)+EMReadCorrection)
                rateLeft=exposures[i-1]/((expEnd[i-1]-expBeg[i-1]).astype(float)+EMReadCorrection)
                averRate=(rateRight+rateLeft)/2
                cdfTot+=averRate*((expBeg[i]-expEnd[i-1]).astype(float)-EMReadCorrection)
        
            elif i==1:
                rateRight=exposures[i]/((expEnd[i]-expBeg[i]).astype(float)+EMReadCorrection)
                rateLeft=exposures[i-1]/(expEnd[i-1]-expBeg[i-1]).astype(float)
                averRate=(rateRight+rateLeft)/2
                cdfTot+=averRate*((expBeg[i]-expEnd[i-1]).astype(float)-EMReadCorrection)    
            i+=1
    
    
        ####Calculate the number of photons within exposures 
        if x >= (expBeg[i]-np.timedelta64(int(EMReadCorrection),'ms')) :
            rateRight=exposures[i]/((expEnd[i]-expBeg[i]).astype(float)+EMReadCorrection)

            if i>0:
                rateLeft=exposures[i-1]/((expEnd[i-1]-expBeg[i-1]).astype(float)+EMReadCorrection)
                averRate=(rateRight+rateLeft)/2
                cdfTot+=averRate*((expBeg[i]-expEnd[i-1]).astype(float)-EMReadCorrection)
                cdfTot+=rateRight*((x.astype(float)-expBeg[i].astype(float)).astype(float)-EMReadCorrection)
            else:
                cdfTot+=rateRight*((x.astype(float)-expBeg[i].astype(float)).astype(float))

    
        ####Calculate the number of photons between exposures   
        else:
            rateRight=exposures[i]/((expEnd[i]-expBeg[i]).astype(float)+EMReadCorrection)
            rateLeft=exposures[i-1]/((expEnd[i-1]-expBeg[i-1]).astype(float)+EMReadCorrection)
            averRate=(rateRight+rateLeft)/2
            cdfTot+=averRate*((x-expEnd[i-1]).astype(float))
        
        return(cdfTot)

    def binOrder(self,EMdataFrame,orderMin,orderMax):
        """bin the flux measurements by order
    
        Args:
            EMdataFrame (pandas.DataFrame): The input data frame from the EM.
            orderMin (numpy.array): An array of values corresponding to the order minimums in angstroms.
            orderMax (numpy.array): An array of values corresponding to the order maximums in angstroms.
                    
        Returns:
            EMdataFrameBin (pandas DataFrame): The binned data frame.
        """
    

        EMdataFrameBin=pd.DataFrame()
    
        #Pull only the flux values
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df_foo=EMdataFrame.select_dtypes(include=numerics)
    
        redLab=np.char.mod('%d', (orderMin+orderMax)/2)
        dfLab=df_foo.columns.values.astype(float)*10
    
        #bin by order limits
        for i in range(len(redLab)):
            indx=np.where(((dfLab>orderMin[i]) & (dfLab<=orderMax[i])))
            if len(indx[0])>0:
                EMdataFrameBin[redLab[i]]=np.sum(df_foo.iloc[:,np.min(indx):(np.max(indx)+1)],axis=1)
            else:
                EMdataFrameBin[redLab[i]]=0
    

        return(EMdataFrameBin)

    

    def midPoint(self,start,finish,exposures,expBeg,expEnd,clip=True):
        """This function calculates the photon midpoint arrival time.
    
        Args:
            start (datetime): The start time of the science exposure.
            finish (datetime): The end time of the science exposure.
            exposures (float): An array of exposure meter flux values ordered chronologically.
            expBeg (datetime): An array of times that mark the beginning of the exposure meter integration.
            expEnd (datetime): An array of times that indicate the end of the exposure meter integration.      
            clip (boolean|True): This switch turns on (True) and off (False) the sigma clipping feature, which removes outliers.
                    
        Returns:
            datetime: The estimated time of arrival for the midpoint photon.

        Notes:
            It is essential that exposures, expBeg, and expEnd all be arrays of the same length.            
        
        """
 
        if np.min(exposures)<0:
            ### Make sure all the expsosures are positive
            exposures=exposures-np.min(exposures)+1
    
        if clip:
            ##Apply sigma clipping filter
            exposures=self.sigmaClip(exposures)
    
    
        #extrapolate edges
        if start<expBeg[0]:
            rate=exposures[0]/((expEnd[0]-expBeg[0]).astype(float))
            exposures[0]=(expEnd[0]-start).astype(float)*rate
            expBeg[0]=start

        if finish>expEnd[-1]:
            last_idx = len(exposures) - 1
            rate = exposures[last_idx] / ((expEnd[-1] - expBeg[-1]).astype(float))
            exposures[last_idx] = (finish - expBeg[-1]).astype(float) * rate
            # rate=exposures[-1]/((expEnd[-1]-expBeg[-1]).astype(float))
            # exposures[-1]=(finish-expBeg[-1]).astype(float)*rate
            expEnd[-1]=finish    
        

        ###calculate the middle number of photons
        startVal=self.CalcCDF(start,exposures,expBeg,expEnd)
        finalVal=self.CalcCDF(finish,exposures,expBeg,expEnd)
        midVal=(finalVal+startVal)/2
    
    
        ###Find the exposure containing the middle number of photons
        dummyVarBeg=np.ones(len(expBeg))
        dummyVarEnd=np.ones(len(expEnd))
    
        for i in range(len(expEnd)):
            dummyVarBeg[i]=self.CalcCDF(expBeg[i],exposures,expBeg,expEnd)
            dummyVarEnd[i]=self.CalcCDF(expEnd[i],exposures,expBeg,expEnd)
    
        indxHigh=np.min(np.where(dummyVarEnd>=midVal))
        indxLow=np.max(np.where(dummyVarBeg<=midVal))
    
    
        ####Use the rate to estimate the time the middle number of photons was recieved
        if indxLow==indxHigh:
            rate=exposures[indxLow]/(expEnd[indxLow]-expBeg[indxLow]).astype(float)
            timeAdd=(midVal-self.CalcCDF(expBeg[indxLow],exposures,expBeg,expEnd))/rate
            outTime=expBeg[indxLow]+np.timedelta64(int(timeAdd),'ms')

        else:
            rateRight=exposures[indxHigh]/(expEnd[indxHigh]-expBeg[indxHigh]).astype(float)
            rateLeft=exposures[indxLow]/(expEnd[indxLow]-expBeg[indxLow]).astype(float)
            averRate=(rateRight+rateLeft)/2
            if averRate>0:
                timeAdd=(midVal-self.CalcCDF(expEnd[indxLow],exposures,expBeg,expEnd))/averRate
                outTime=expEnd[indxLow]+np.timedelta64(int(timeAdd),'ms')
            else:
                outTime=np.datetime64("NaT")
    
        
        return(outTime)
    
    def orderedMidPoint(self):
        """This function calculates the photon midpoint arrival time for each user defined segment. If none are provided
            the default is the WAVE_1 orders.
        """
        
        ####if no segments provided, defaults to WAVE_1 values.
        if (self.segmentMin is None) | (self.segmentMax is None):
            self.segmentMin=orderMin
            self.segmentMax=orderMax
            warnings.warn("No segments bounds were provided. The default bounds for WAVE_1 are being asserted. ")
        
        i = 0
        for col in df_EM.columns:
            if col.lower().startswith('date'):
                i += 1
            else:
                break
        wav_str = self.df_EM.columns[i:] # string (center) wavelengths of each pixel
        wav = self.df_EM.columns[i:].astype(float) # float (center) wavelengths of each pixel

        # define disperion at each wavelength (nm per pixel)
        disp = wav*0+np.gradient(wav,1)*-1

        # define normalized flux array (e- / nm / time)
        df_EM_norm = self.df_EM[wav_str] * EM_gain /disp

        # define time arrays
        date_beg = np.array(self.df_EM["Date-Beg"], dtype=np.datetime64)
        date_end = np.array(self.df_EM["Date-End"], dtype=np.datetime64)

        # Bin wavelengths into orders
        df_EM_bin=self.binOrder(df_EM_norm,self.segmentMin,self.segmentMax)
        wav_str = df_EM_bin.columns

        #calculate the midpoint for all bands
        midBand = np.empty(len(wav_str), dtype='datetime64[ms]')
        for i in range(len(wav_str)):
            midBand[i]=(self.midPoint(self.start,self.finish,df_EM_bin[wav_str[i]].astype(float),date_beg,date_end,self.clip))
    
        self.orderMid=df_EM_bin.columns.astype(float)
        self.midPhoto=midBand    
    
        return (self.orderMid,self.midPhoto)   
       