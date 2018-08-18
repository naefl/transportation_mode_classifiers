import traceback
import pyproj
from bisect import bisect_left
import matplotlib.pyplot as plt
import scipy
import json
import time
import sys
from scipy.signal import butter, lfilter, freqz
import numpy as np
import re
import pandas as pd
import os
import glob
import sbb_features
from nitime.algorithms import autoregressive
import math

path = r'C:\Users\lucas\OneDrive\masters_thesis\\'

sensorList = [
    'accelerometer',
    # 'batterylevel',
    # 'compass',
    # 'cpu',
    'gps',
    # 'gyroscope',
    # 'humidity',
    # 'light',
    # 'magneticfield',
    # 'memory',
    # 'network',
    # 'pressure',
    # 'proximity',
    # 'signalstrength',
    # 'soundlevel'
    # 'steps',
    # 'temperature'
]

vehicleList = ['bus','car','bicycle','walking','tram','train']

def setcwd():
    os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\data\\training_data\\sensor_lab final data')


def changecwd(newPath):
   os.chdir(newPath)


def getVehicleAndSensorDfs():
    """
    :return:  returns  {'vehicletype':{'sensor':[DataFrame1,DataFrame2]}}
    """
    setcwd()
    sensorDataDictDirs = {}
    rangeOfVehicleLists = range(len(vehicleList))
    rangeOfSensorLists = range(len(sensorList))
    for i in rangeOfVehicleLists:
        vehicle = vehicleList[i]
        sensorDataDictDirs[vehicle] = {}
        for j in rangeOfSensorLists:
            listOfFilePaths = []
            sensor = sensorList[j]
            sensorDataDictDirs[vehicle][sensor] = []
            listOfFilePaths = glob.glob(
                'sensorlab_{}\**\{}.csv'.format(vehicle, sensor),
                recursive=True
            )
            for filePath in listOfFilePaths:
                directory = os.path.join(os.getcwd(), filePath)
                df = pd.read_csv(directory)
                filename = re.search(r'(.*)\\(\w)+\.csv',directory).group(1)
                df['filename'] = filename
                if 'gps.csv' in filePath:
                    data = json.load(open(filename + '\\_info.json'))
                    df['timestamp'] = data['start'] + df['time']
                sensorDataDictDirs[vehicle][sensor].append(df)
    return sensorDataDictDirs


def filterUnusableData(data):
    unusableLocations = []
    for vehicle in data:
        for DataFrame in data[vehicle]['gps']:
            if DataFrame.accuracy.mean() > 200:
                unusableLocations.append(DataFrame.filename[0])
        for sensor in data[vehicle]:
            for idx, DataFrame in enumerate(data[vehicle][sensor]):
               if len(DataFrame.filename) > 0 and len(unusableLocations) > 0:
                    if DataFrame.filename[0] in unusableLocations:
                        del data[vehicle][sensor][idx]
    return data



# def rollingVariance(normalizedAccelerometerArray, sigma_t_minusOne_squared, a):
#     N = len(normalizedAccelerometerArray)
#     sigma_t_squared = sigma_t_minusOne_squared + \
#                       np.average(normalizedAccelerometerArray)**2 -\
#                       np.average(np.append(normalizedAccelerometerArray, a))**2 + \
#                         (a**2+normalizedAccelerometerArray[0])/N
#     return sigma_t_squared


#returns variance over all data. must be performed for all data before breaking into windows
#as ootherwise we don't have enough far data back

#.✔unittested
def getVariance(accelerometerArray):
    N = 10
    varray = [ ]
    for i in range(len(accelerometerArray)):
         if i <= N-1:
             normalizedAccelerometerArray = accelerometerArray[:N]
         else:
             normalizedAccelerometerArray = accelerometerArray[i - N:i]
         σ_t_squared = np.var(normalizedAccelerometerArray)
         varray = np.append(varray,σ_t_squared)
         σ_t_minusOne_squared = σ_t_squared


    return varray

#✔unittested
def getARCoefficients(nparray):
    try:
        coeffs,sigma = autoregressive.AR_est_YW(nparray,5)
    except ValueError:
        print(nparray)
        coeffs, sigma = autoregressive.AR_est_YW(nparray,4)
        coeffs.extend([0])
    return coeffs

# def getTimepointsVariance(nparray, previousArray, sigma_squared_minusOne, avg_a_t_minusOne):
#     N = len(nparray)
#     sigma_squared_array = np.ndarray([])
#     for idx,t in enumerate(nparray):
#         a_t = nparray[idx]
#         if idx == 0:
#             if not len(avg_a_t_minusOne) >= 1:
#                 avg_a_t_minusOne = np.average(nparray)
#         else:
#             avg_a_t_minusOne = np.average(nparray[:idx])
#         try:
#             a_t_minusN = previousArray[idx]
#         except IndexError:
#             a_t_minusN = np.average(nparray)
#         avg_a_t_minusOne_squared = avg_a_t_minusOne**2
#         a_t_squared = a_t**2
#         a_t_minusN_squared = a_t_minusN**2
#         sigma_squared = sigma_squared_minusOne - a_t_minusOne_squared + (a_t_squared - a_t_minusN_squared)/N
#         sigma_squared_array = np.append(sigma_squared_array, sigma_squared)
#         sigma_squared_minusone = sigma_squared
#     return sigma_squared_array, sigma_squared_minusOne, avg_a_t_minusOne


# ✔ unittested
def getFourerAccelVar12point5Hz(nparray, flag):

    hz = 20
    dt = 1/hz
    n = len(nparray)
    w = np.abs(np.fft.fft(nparray)[:int(n/2)+1]/n)
    freqs = np.fft.fftfreq(n, dt)[:int(n/2)+1]
    # if flag == True:
    #     plt.plot(freqs, w, "g")
    #     plt.draw()
    #     plt.show()
    #of variance
    try:
        p0_98 = w[bisect_left(freqs, 0.98)].item(0)
        p1_22 = w[bisect_left(freqs, 1.22)].item(0)
        p0_73  = w[bisect_left(freqs, 0.73)].item(0)
        if bisect_left(freqs,5) != bisect_left(freqs,7):
            p6  = np.max(w[bisect_left(freqs,5):bisect_left(freqs,7)])
        else:
            p6= w[bisect_left(freqs,7)]
        if bisect_left(freqs,2) != bisect_left(freqs,4):
            p2_5 = np.max(w[bisect_left(freqs,2):bisect_left(freqs,4)])
        else:
            p2_5 = w[bisect_left(freqs,2.5)]
        if p0_73 != 0 and p1_22 != 0:
            p_r = 2*p0_98/(p0_73+p1_22)
        else:
            p_r = 0
    except IndexError:
        print(freqs)
        p_r = 0

    return p_r, p2_5, p6

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = pd.np.radians([lat1, lon1, lat2, lon2])

    a = pd.np.sin((lat2-lat1)/2.0)**2 + \
        pd.np.cos(lat1) * pd.np.cos(lat2) * pd.np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * pd.np.arcsin(np.sqrt(a))
#✔ unittested
def getFourierAccelNorm123456HzAndMaxCoeffs(nparray, flag):
    #TODO:

    #n = length of array (points)
    #N = length of signal (s)
    #the period (aka duration) of the signal x, sampled at dt with N samples is dt*N
    # hz = sampling frequency
    hz = 20
    dt = 1/hz
    n = len(nparray)
    coeffs = np.abs(np.fft.fft(nparray)[:int(n/2)+1]/n)
    # if set correctly, you receive hertz
    freqs = np.fft.fftfreq(n, dt)[:int(n/2)+1]
    # if flag == True:
    #     plt.plot(freqs[1:],coeffs[1:],'b')
    #     plt.draw()
    #     plt.show()


    try:
        hz1_peak_idx =  bisect_left(freqs,1)
        hz1_peak = coeffs[hz1_peak_idx]
        hz2_peak_idx  =  bisect_left(freqs,2)
        hz2_peak= coeffs[hz2_peak_idx]
        hz3_peak_idx  =   bisect_left(freqs,3)
        hz3_peak= coeffs[hz3_peak_idx]
        hz4_peak_idx  =   bisect_left(freqs,4)
        hz4_peak= coeffs[hz4_peak_idx]
        hz5_peak_idx =   bisect_left(freqs,5)
        hz5_peak= coeffs[hz5_peak_idx]
        hz6_peak_idx  =    bisect_left(freqs,6)
        hz6_peak= coeffs[hz6_peak_idx]
    except:
        print("________________________________________________")
        print("EXCEPTION occured in getFourrierAccelNorn12345, line 219-230")
        print("coeffs were")
        print(coeffs)
        print("freqs were")
        print(freqs)
        print("________________________________________________")
        raise
    # Find the peak i the coefficients
    #TODO: Test
    #coeffs[0] contains zero freq term - i.e sum of signal, :20 are max for all modes
    # and thus not good deciding factor
    max_idx = np.argmax(coeffs[1:])

    max_coeff= coeffs[1:][max_idx]
    max_freq = freqs[1:][max_idx]
    return [hz1_peak, hz2_peak, hz3_peak, hz4_peak, hz5_peak, hz6_peak, max_freq, max_coeff]

def getSpeedChange(previous_v, next_v):
    speedChange = next_v- previous_v
    return speedChange

#✔unittested
def getPeakAverageAndRatio(nparray):
    tresholdMargin = 0.85
    treshold = tresholdMargin*np.max(nparray)
    peakLocations = list(np.where(nparray >= treshold)[0])
    peaks =    nparray[peakLocations]
    peakRatio = len(peakLocations)/len(nparray)
    peakLocationPairs = list(chunks(peakLocations, 2))
    # 50ms distance between each accel point
    peakAverage_ms = np.average([pair[1]-pair[0] for pair in peakLocationPairs if len(pair)==2] )*50
    return peakAverage_ms, peakRatio

def transformWGS84ToUMT(stopstxt_df):
    WGS84 = "+init=EPSG:4326"
    EPSG21781 = "+init=EPSG:2056"
    Proj_to_EPS21781= pyproj.Proj(EPSG21781)
    UMTstopsGPS=stopstxt_df[['stop_lon','stop_lat']].apply(lambda row: list(Proj_to_EPS21781(row[0],row[1])), axis =1 ).values
    return  UMTstopsGPS

def getSignalMagnitudeArea(x,y,z):
   """
   :param x: x accel coord
   :param y: y accel coord
   :param z: z accel coord
   :return:
   """
   SMA = (np.sum(x)+np.sum(y)+np.sum(y))/len(x)
   return SMA

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunkSensorDfsToWindowsAndSelectFeatures(data, filter):
    """
    :param data:  list of sensordics
    :return: df with all features and fs
    """
    features = [
        'speed'
        ,'speedChange'
        ,'bearingsChange'
        ,'accuracy'
        ,'dist1'
        ,'dist2'
        ,'dist3'
        ,'dist4'
        ,'dist5'
        ,'min_time1'
        ,'min_time2'
        ,'min_time3'
        ,'min_time4'
        ,'min_time5'
        ,'accel_max'
        ,'accel_min'
        ,'accel_var'
        ,'peakAverage'
        ,'peakRatio'
        ,'binnedPercentage1'
        ,'binnedPercentage2'
        ,'binnedPercentage3'
        ,'binnedPercentage4'
        ,'binnedPercentage5'
        ,'binnedPercentage6'
        ,'binnedPercentage7'
        ,'binnedPercentage8'
        ,'binnedPercentage9'
        ,'fc1hz'
        ,'fc2hz'
        ,'fc3hz'
        ,'fc4hz'
        ,'fc5hz'
        ,'fc6hz'
        ,'fft_max_coefficient_freq'
        ,'fft_max_coefficient'
        ,'fc1hzVar'
        ,'fc2_5hzVar'
        ,'fc5hzVar'
        ,'ARc1'
        ,'ARc2'
        ,'ARc3'
        ,'ARc4'
        ,'ARc5'
        ,'SMA'
        ,'label'
    ]
    #
    # os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\2018 GTFS zvv google transit')
    # stopstxt_df = pd.read_csv('stops.txt')[
    #     [
    #         'stop_id',
    #         'stop_lat',
    #         'stop_lon'
    #     ]
    # ]
    #
    # UMTstopsGPS = transformWGS84ToUMT(stopstxt_df)
    # stoptimestxt_df = pd.read_csv('stop_times.txt')[
    #     [
    #         'departure_time',
    #         'stop_id'
    #     ]
    # ]
    # os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\data\\training_data\\sensor_lab final data')


    allTrainingsData = []
    #Todo: swotch back to beginning
    for vehicle in list(data.keys())[1:]:
        vehicleTrainingsData = []
        #TODO: switch back to from beginning
        for i,df in enumerate(data[vehicle]['gps']):
            # if vehicle == 'car' and i < 2:
            #     continue
            vehicle_time = time.time()
            start_time1 = time.time()
            # sbb1 = sbb_features.returnGISFeaturesUTM(
            #     df[['lat','lng', 'timestamp']],
            #     stopstxt_df,
            #     stoptimestxt_df,
            #     UMTstopsGPS
            # )
            # print("--- get sbb features UTM for %s seconds ---" % (time.time() - start_time1))
            start_time = time.time()
            sbb = sbb_features.returnGISFeatures(
                df[['lat','lng', 'timestamp']],
            )
            accel_df = data[vehicle]['accelerometer'][i]
            prevTimestamp = 0
            prevTimestamp = 0 #bug, doesn't work with just one declaration
            previousSpeed = 0
            chunk_size = 2 #seconds
            previousBearing = getBearing((df['lat'][2],df['lng'][2]),(df['lat'][1],df['lng'][1]))
            previousLat =df['lat'][1]
            previousLon =df['lng'][1]
            trainingsData = []
            try:
                for j,timestamp in enumerate(df['time'].values[:-1]):
                    if j%2 == 0:
                        currentTimestamp = df['time'].values[j+1]
                        accel_chunk = accel_df[(accel_df['time'] > prevTimestamp) & (accel_df['time'] <= currentTimestamp)]
                        prevTimestamp = currentTimestamp
                        if len(accel_chunk) >= 10:
                            trainingsDataPoint = []
                            speed = np.average(df['speed'][[j,j+1]].values)
                            trainingsDataPoint.append(speed)
                            speed_change = getSpeedChange(previousSpeed, speed)
                            trainingsDataPoint.append(speed_change)
                            previousSpeed = speed
                            lat = df['lat'][j]
                            lon = df['lng'][j]
                            bearing = getBearing((lat,lon),(previousLat,previousLon))
                            bearingChange = abs(previousBearing - bearing)
                            trainingsDataPoint.append(bearingChange)
                            accuracy = np.average(df['accuracy'][[j,j+1]].values)
                            trainingsDataPoint.append(accuracy)
                            previousLat = lat
                            previousLon = lon
                            sbb_feature = sbb.values[j]
                            trainingsDataPoint.extend(sbb_feature)
                            try:
                                accel_chunk = normalizeAndApplyFilters(accel_chunk)
                            except ValueError:
                                print(traceback.format_exc())
                                print("__________________________________________________________________________________________________________________")
                                print("exception occured in {}'th df of {} vehicle in the {}'th loop of normalizeAndApplyFilters line 407".format(i,vehicle,j))
                                print("accel_chunk was")
                                print(accel_chunk)
                                print("accel_df was")
                                print('timestamps were')
                                print(' {} - {} '.format(prevTimestamp, currentTimestamp))
                                print("__________________________________________________________________________________________________________________")
                            filteredAccelChunk = accel_chunk[filter].values
                            accel_chunk['variance'] = getVariance(filteredAccelChunk)
                            maxAccel = np.max(filteredAccelChunk)
                            trainingsDataPoint.append(maxAccel)
                            minAccel = np.min(filteredAccelChunk)
                            trainingsDataPoint.append(minAccel)
                            accel_var = np.average(accel_chunk['variance'].values)
                            trainingsDataPoint.append(accel_var)
                            peakAverage, peakRatio = getPeakAverageAndRatio(filteredAccelChunk)
                            trainingsDataPoint.append(peakAverage)
                            trainingsDataPoint.append(peakRatio)
                            percentagesPerBin = getBinnedDistribution(filteredAccelChunk)
                            trainingsDataPoint.extend(percentagesPerBin)
                            try:
                                fourrier_123456hz_coeff_max_freq_max_coeff= getFourierAccelNorm123456HzAndMaxCoeffs(filteredAccelChunk, True if j%1000==0 else False)
                            except:
                                print("__________________________________________________________________________________________________________________")
                                print(traceback.format_exc())
                                print("exception occured in {}'th df of {} vehicle in the {}'th loop of getFourrierAccelNorm123456 line 431".format(i,vehicle,j))
                                print("accel_chunk was")
                                print(accel_chunk)
                                print("accel_df was")
                                print(accel_df)
                                print('timestamps were')
                                print(' {} - {} '.format(prevTimestamp, currentTimestamp))
                                print("__________________________________________________________________________________________________________________")
                            trainingsDataPoint.extend(fourrier_123456hz_coeff_max_freq_max_coeff)
                            fourier_12point56hz_var = getFourerAccelVar12point5Hz(accel_chunk['variance'].values, True if j % 1000 == 0 else False)
                            trainingsDataPoint.extend(fourier_12point56hz_var)
                            try:
                                AR5Coeffs = getARCoefficients(filteredAccelChunk)
                            except:
                                print("__________________________________________________________________________________________________________________")
                                print(traceback.format_exc())
                                print("an EXCEPTION occured in {}'th df of {} vehicle in the {}'th loop of getARCoefficients line 445 ".format(i,vehicle,j))
                                print("accel_chunk was")
                                print(accel_chunk)
                                print("accel_df was")
                                print(accel_df)
                                print('timestamps were')
                                print(' {} - {} '.format(prevTimestamp, currentTimestamp))
                                print("__________________________________________________________________________________________________________________")
                            trainingsDataPoint.extend(AR5Coeffs)
                            SignalMagnitudeArea = getSignalMagnitudeArea(accel_chunk['x'],accel_chunk['y'],accel_chunk['z'])
                            trainingsDataPoint.append(SignalMagnitudeArea)
                            trainingsDataPoint.append(vehicle)
                            trainingsData.append(trainingsDataPoint)
                try:
                    trainingsDataDf= pd.DataFrame(data= trainingsData, columns= features)
                    trainingsDataDf.to_csv(path+'{}_{}_trainingsdata.csv'.format(vehicle,i))
                    print(
                        'INFORMATION: done with {} number {}'.format(vehicle,i)
                    )
                except AssertionError:
                    print("__________________________________________________________________________________________________________________")

                    print("________________________________________________")
                    print("trainingsdata was")
                    print(trainingsData)
                    print("accel_df was")
                    print(accel_df)
                    print("gps df was")
                    print(df)
                    print(traceback.format_exc())
                    print("an EXCEPTION occured in {}'th df of {} vehicle line 470 converting the trainsingsData to a DF".format(i,vehicle))
                    print('timestamps were')
                    print(' {} - {} '.format(prevTimestamp, currentTimestamp))
                    print("__________________________________________________________________________________________________________________")

                    print('features were')
                    print('fourier coeffs')
                    print(fourrier_123456hz_coeff_max_freq_max_coeff)
                    print('fourrier var')
                    print(fourier_12point56hz_var)
                    print('AR5')
                    print(AR5Coeffs)

                    print("__________________________________________________________________________________________________________________")

                    print("________________________________________________")
                    print("__________________________________________________________________________________________________________________")
                vehicleTrainingsData.extend(trainingsData)
            except (ValueError, IndexError, AttributeError, KeyError, TypeError) as e:
                print("__________________________________________________________________________________________________________________")
                print("accel_chunk was")
                print(accel_chunk)
                print(traceback.format_exc())
                if j + 50 < accel_df.shape[0] and j -50 > 0:
                    print('surrounding accel_df rows:')
                    print(accel_df.iloc[j-20:j+20,:])
                print("UNSPECIFIEC EXCEPTION occured in {}'th df of {} vehicle in the {}'th loop of chunksensorDf line 465 ".format(i,vehicle,j))
                print('timestamps were')
                print(' {} - {} '.format(prevTimestamp, currentTimestamp))
                print("__________________________________________________________________________________________________________________")
        vehicletrainingsDf = pd.DataFrame(data= vehicleTrainingsData, columns= features)
        vehicletrainingsDf.to_csv(path+'{}_trainingsdata.csv'.format(vehicle))
        allTrainingsData.extend(vehicleTrainingsData)
        print("__________________________________________________________________________________________________________________")
        print(' Finished with vehicle ' + vehicle)
        print("__________________________________________________________________________________________________________________")
    trainingsDf = pd.DataFrame(data= allTrainingsData, columns= features)
    trainingsDf.to_csv(path+'combined_trainingsdata.csv')

def getBearing(pointA, pointB):
    """
    credits:https://gist.github.com/jeromer/2005586
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#✔unittested
def getBinnedDistribution(nparray):
    max = np.max(nparray)
    bins = np.linspace(0,max+1,10)
    length = len(nparray)
    percentagesPerBin = [nparray[(nparray >= bins[i]) & (nparray < bins[i+1])].size/length for i in range(len(bins)-1)]
    return percentagesPerBin

def mergeSensorDfs(data):
    """
    combines all dataSets from a Sensor, adds their timestamps
    :return: {'vehicle'{'sensor':DataFrame}}
    """
    combinedDic = {}
    for vehicle in data:
        combinedDic[vehicle] = {}
        for sensor in data[vehicle]:
            if len(data[vehicle][sensor])>0:
                highestTimeStamp = 0
                uberDf = pd.DataFrame(columns=[x for x in data[vehicle][sensor][0].columns])
                for i, df in enumerate(data[vehicle][sensor]):
                    if not df.empty:
                        df_new = df.copy()
                        df_new['time'] = df_new['time'] + highestTimeStamp
                        highestTimeStamp += int(df_new.loc[df_new['time'].idxmax()]['time'])
                        uberDf = pd.concat([uberDf,df_new], ignore_index=True)
                combinedDic[vehicle][sensor] = uberDf
                combinedDic[vehicle][sensor]['vehicle'] = vehicle
    return combinedDic

def normalizeAndApplyFilters(accel_chunk):
    normalized_filtered_accel_values = accel_chunk.iloc[:,1:4]
    normalized_filtered_accel_values['norm'] = normalized_filtered_accel_values.apply(
        lambda row: np.linalg.norm(row),
        axis=1
    )

    normalized_filtered_accel_values['5_point_median'] = pd.rolling_mean(
            normalized_filtered_accel_values['norm'],
            5
    )
    try:
        window = scipy.signal.gaussian(M=10, std=6)
        window /= window.sum()
    except ValueError:
        print('EXCEPTION: Valueerror, triggered at Line 532, trying to make the window for gaussian.')
        print( traceback.format_exc())
        print("normalized_filtered_accel_values['norm'] is {}".format(normalized_filtered_accel_values['norm']))
        print("accel chunk is {}".format(accel_chunk))
        raise

    try:
        normalized_filtered_accel_values['gaussian'] = pd.Series(
            np.convolve(
              normalized_filtered_accel_values['norm'], window, mode='same'
            )
        )
    except ValueError:
        print('EXCEPTION: Valueerror, triggered at Line 537, trying to apply gaussian lowpass.')
        print( traceback.format_exc())
        print("normalized_filtered_accel_values['norm'] is {}".format(normalized_filtered_accel_values['norm']))
        print("accel chunk is {}".format(accel_chunk))
        raise

    try:
        normalized_filtered_accel_values['butter_5hz_lowpass']=butter_lowpass_filter(
                normalized_filtered_accel_values['norm'],5,20,order=5
        )
    except ValueError:
        print('EXCEPTION: Valueerror, triggered at Line 540, trying to apply butter lowpass')
        print( traceback.format_exc())
        print(normalized_filtered_accel_values['norm'])
        print(accel_chunk)
        raise

    return normalized_filtered_accel_values

def applyNorm(x,y,z):
    return np.linalg.norm(x,y,z)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(
        order,
        normal_cutoff,
        btype='low',
        analog=False
    )
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def saveAsCSV(data):
    for vehicle in data:
        for sensor in data[vehicle]:
           data[vehicle][sensor].to_csv('.\{}_{}.csv'.format(vehicle,sensor))


def loadFromCsv():
    dfs = {}
    for vehicle in vehicleList:
        dfs[vehicle] = {}
        for sensor in sensorList:
            dfs[vehicle][sensor] = pd.read_csv('{}_{}.csv'.format(vehicle, sensor))
    return dfs



filter = "butter_5hz_lowpass"
dfs = getVehicleAndSensorDfs()
dffeat = chunkSensorDfsToWindowsAndSelectFeatures(dfs,filter)
# nparray = np.array([9,46,34,43,4,7,79,46,234,643,46,7,720,810,630,700,1,6,2,6,7,824,23,4,1,2,4])
# coeffs = getARCoefficients(nparray)
# df1 = dfs['bus']['accelerometer'][0]
# nf_df1 = normalizeAndApplyFilters(df1)
# var = getVariance(nf_df1['butter_5hz_lowpass'])
# plt.plot(range(len(var)),var,'b')
# plt.show()
# hz = getFourerAccelVar1Hz(var)
# sbbFeat = getSBBFeatures()
# coeff = getFourierAccelNorm123456HzAndMaxCoeffs(nf_df1['butter_5hz_lowpass'])

    # saveAsCSV(df3)
    # dfs = loadFromCsv()
    # getSBBFeatures(dfs)
    # saveAsCSV(dfs)
