import traceback
import datetime
import pyproj
import pytz
import geopy.distance
import pandas as pd
import os
import time
import numpy as np
from scipy.spatial import distance

##PARAMS
thirtyDayMonths = [9,6,4,11]
thirtyOneDayMonths = [1,3,5,7,8]
downsampling_factor = 15
columns = ['lat','lng','id1','id2','id3','id4','id5','dist1','dist2','dist3','dist4','dist5']

def unixTimeStampToYYYYMMDDhhmm(timestamp_s):
    zurich = pytz.timezone('CET')
    return zurich.localize(datetime.datetime.fromtimestamp(timestamp_s)).strftime('%Y-%m-%d %H:%M')


def getYYYYMMDD(timestamp_s):
    zurich = pytz.timezone('CET')
    return zurich.localize(
           datetime.datetime.fromtimestamp(timestamp_s)
       ).strftime('%Y-%m-%d')


def getYYYYMMDDhhmmss(timestamp_df_s):
    zurich = pytz.timezone('CET')
    return timestamp_df_s.apply(
        lambda x: zurich.localize(
            datetime.datetime.fromtimestamp(x)
        ).strftime('%Y-%m-%d %H:%M:%S')
    )


def getCorrectStopTime(stopTime_YYYYMMDDhhmmss):
    date, stoptime = stopTime_YYYYMMDDhhmmss.split(' ')
    year,month, day = [int(x) for x in date.split('-')]
    hour, minute, second = [int(x) for x in stoptime.split(':')]
    if hour >= 24:
        hour = (hour - 24)
        if (day < 30 and month in thirtyDayMonths) or \
            (day <31 and month in thirtyOneDayMonths) or \
            (day <28 and month is 2):
                day = day + 1
        else:
            day = 1
            month += 1
    [year, month, day,hour,minute,second] = [
        x.zfill(2) if len(x) < 2 else x for x in [
            str(year),
            str(month),
            str(day),
            str(hour),
            str(minute),
            str(second)
        ]
    ]
    correctedStopTime = year + '-' + month + '-' + day + ' '  + hour + ':' + minute + ':' + second
    return correctedStopTime


def getSchedUnixt(corrected_sched_t):
    return list(
        map(
            lambda s: time.mktime(
                datetime.datetime.strptime(
                    s,
                    "%Y-%m-%d %H:%M:%S"
                ).timetuple()
            ),
            corrected_sched_t
        )
    )


def getSecToDeparture(sched_unix_t, timestamp_df_s, i):
    return [
        abs(x) for x in list(
            map(
                lambda x:
                x - timestamp_df_s.iloc[i],
                sched_unix_t
            )
        )
    ]


def getMatchingStopTimes(stopId, stoptimestxt_df,YYYYMMDD):

    return getSchedUnixt(
        [
        getCorrectStopTime(x) for x in YYYYMMDD + " " + stoptimestxt_df[
                stoptimestxt_df.stop_id == stopId
            ].departure_time
        ]
    )


def get5nearestBusIDAndDistance(lat, lng, stopstxt_df):
    """
    :param lat: latitude of current GPS loc
    :param lon: longitude of current GPS loc
    :param stopstxt_df: dataframe containing GTFS "stops.txt"
    :param distanceSorted_stopstxt_df: df with five closest stops and columns [
        stop_id,
        stop_lat,
        stop_lon,
        dist
        ]
    """

    nearest5BusIdsAndDist_df = pd.DataFrame(columns=columns)
    nearest5BusIdsAndDist_df_row_temp = []

    for i in range(len(lat)):
        if i%downsampling_factor == 0:
            stopstxt_df['dist'] =stopstxt_df.apply(
                lambda row: geopy.distance.vincenty(
                    (row.stop_lat, row.stop_lon),
                    (lat[i],lng[i])
                ).m,
                axis=1
            )
            sortedAndFilteredStops_df = stopstxt_df[['stop_id','dist']].sort_values(by=['dist'])[:5]
            dist = sortedAndFilteredStops_df.dist.values
            ids = sortedAndFilteredStops_df.stop_id.values
            arr =  [ lat[i] , lng[i] ] + ids.tolist() + dist.tolist()
            nearest5BusIdsAndDist_df_row_temp = pd.DataFrame(
                np.array( [ arr ] ), columns=columns
            )
        nearest5BusIdsAndDist_df = nearest5BusIdsAndDist_df.append(nearest5BusIdsAndDist_df_row_temp,ignore_index=True)
    return nearest5BusIdsAndDist_df


def getMin5TimeToDepartureFeatureDf(nearest5BusIdsAndTimes_df, timestamp_df_s, YYYYMMDD, stoptimestxt_df):
    """
    :param nearest5BusIdsAndTimes_df:
    :param averageTimeStampInYYYYMMDDhhmm:
    :param timestamp_df_s:
    :param YYYYMMDD: date of TimeStamp
    :param stoptimestxt_df: dataframe witho stoptimes
    :return: shortest time difference at five clostest Bust stops
    """
    columns = ['min_time1','min_time2','min_time3','min_time4','min_time5']
    min5timesToDeparture_df = pd.DataFrame(columns=columns)
    min5timesToDeparture_df_row_temp = []
    busStopIds_df = nearest5BusIdsAndTimes_df[['id1','id2','id3','id4','id5']]

    for i in range(nearest5BusIdsAndTimes_df.shape[0]):
        if i%downsampling_factor == 0:
            min5TimesAllStops_s = np.empty([0,0])
            for stopId in busStopIds_df.iloc[i,:].tolist():
                sched_stop_t_at_StopId  = getMatchingStopTimes(stopId, stoptimestxt_df, YYYYMMDD)
                #TODO: make more efficient, correct whole Df beforehand, not on the go
                #TODO: also, combine all loops that are used for data processing purposes
                if not 'parent' in stopId:
                    sToDeparture = getSecToDeparture(sched_stop_t_at_StopId, timestamp_df_s, i)
                    if len(sToDeparture) >0:
                        try:
                            if len(sToDeparture) < 5:
                                sToDeparture.extend(
                                    [np.average(sToDeparture) for x in range(5-len(sToDeparture))]
                                )
                            min5indPerStop = np.argpartition(sToDeparture,-5)[-5:]
                            min5TimesPerStop_s = np.array(
                                sToDeparture
                            )[min5indPerStop]
                            min5TimesAllStops_s = np.append(
                                min5TimesAllStops_s,
                                min5TimesPerStop_s
                            )
                        except ValueError:
                            print(traceback.format_exc())
                            print('sched_stop_t_at_StopId')
                            print(sched_stop_t_at_StopId)
                            print('sToDeparture')
                            print(sToDeparture)
                            print('mind5indPerStop')
                            print(min5indPerStop)
                            print(min5indAllStops)
                        ##debug
                        # if i%downsampling_factor == 0:
                        #     print('min 5 times per stop')
                        #     print(min5TimesPerStop_s)
                        #     print('min5timesallstops')
                        #     print(min5TimesAllStops_s)
                        # ###debug
                else:
                    min5TimesAllStops_s = np.append(min5TimesAllStops_s, 0)
            if 0 in min5TimesAllStops_s:
                min5TimesAllStops_s[
                    min5TimesAllStops_s == 0
                    ] = np.average(
                    min5TimesAllStops_s[min5TimesAllStops_s != 0]
                )
            min5indAllStops = np.argpartition(
                min5TimesAllStops_s, -5
            )[-5:]
            min5TimesAllStops_s = np.array(
                min5TimesAllStops_s
            )[min5indAllStops]
            min5timesToDeparture_df_row_temp = pd.DataFrame(
                [min5TimesAllStops_s], columns=columns
            )
        min5timesToDeparture_df = min5timesToDeparture_df.append(
            min5timesToDeparture_df_row_temp,
            ignore_index=True
        )
    # print(min5timesToDeparture_df)
    return min5timesToDeparture_df


def getTimeToDepartureUTM(nearest5Ids, timestamp_df_s, YYYYMMDD, stoptimestxt_df):
    """
    :param nearest5BusIdsAndTimes_df:
    :param averageTimeStampInYYYYMMDDhhmm:
    :param timestamp_df_s:
    :param YYYYMMDD: date of TimeStamp
    :param stoptimestxt_df: dataframe witho stoptimes
    :return: shortest time difference at five clostest Bust stops
    """
    min5timesToDeparture= []

    for i in range(len(nearest5Ids)):
        if i%downsampling_factor == 0:
            min5TimesAllStops_s = np.empty([0,0])
            for stopId in nearest5Ids[i]:
                scheduled_stoptimes  = getMatchingStopTimes(stopId, stoptimestxt_df)
                scheduled_t_YYYYMMDDhhmmss =  + " " + scheduled_stoptimes
                if not 'parent' in stopId:
                    corrected_scheduled_t = [
                        getCorrectStopTime(x) for x in scheduled_t_YYYYMMDDhhmmss
                    ]
                    scheduled_unix_t = getSchedUnixt(corrected_scheduled_t)
                    sToDeparture = getSecToDeparture(scheduled_unix_t, timestamp_df_s, i)
                    if len(sToDeparture) >0:
                        try:
                            min5indPerStop = np.argpartition(sToDeparture,-5)[-5:]
                            min5TimesPerStop_s = np.array(
                                sToDeparture
                            )[min5indPerStop]
                            min5TimesAllStops_s = np.append(
                                min5TimesAllStops_s,
                                min5TimesPerStop_s
                            )
                        except ValueError:
                            print(traceback.format_exc())
                            min5TimesAllStops_s = np.append(
                                min5TimesAllStops_s,
                                sToDeparture
                            )

                        ##debug
                        # if i%downsampling_factor == 0:
                        #     print('min 5 times per stop')
                        #     print(min5TimesPerStop_s)
                        #     print('min5timesallstops')
                        #     print(min5TimesAllStops_s)
                        # ###debug
                else:
                    min5TimesAllStops_s = np.append(min5TimesAllStops_s, 0)
            if 0 in min5TimesAllStops_s:
                min5TimesAllStops_s[
                    min5TimesAllStops_s == 0
                ] = np.average(
                    min5TimesAllStops_s[min5TimesAllStops_s != 0]
                )
            try:
                min5indAllStops = np.argpartition(
                    min5TimesAllStops_s, -5
                )[-5:]
                min5TimesAllStops_s = np.array(
                    min5TimesAllStops_s
                )[min5indAllStops]
            except ValueError:
                min5TimesAllStops_s = list(min5TimesAllStops_s).extend(
                    [
                        np.average(min5TimesAllStops_s) for x in range(5-len(min5TimesAllStops_s))
                    ]
                )
                min5indAllStops = np.argpartition(
                    min5TimesAllStops_s, -5
                )[-5:]
                min5TimesAllStops_s = np.array(
                    min5TimesAllStops_s
                )[min5indAllStops]
            min5timesToDeparture.append(min5TimesAllStops_s)
    # print(min5timesToDeparture_df)
    return min5timesToDeparture


def setcwd():
    os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\data\\training_data\\sensor_lab final data')


##Todo: check unit of distances, by looking at GPS lat lon dist
def vectorizedUTMNearest5BusIdsAndDist(UMTmeasurement, UMTstopsGPS, stopstxtdf):

    #distance calculatiosn are correct
    distances = distance.cdist(UMTmeasurement,UMTstopsGPS)
    #indices are the same for every point
    minInds = [np.argpartition(x,-5)[-5:] for x in distances]
    nearest5BusIds = [stopstxtdf['stop_id'].values[x] for x in minInds]
    nearest5BusDistance = [distances[i][minInds[i]] for i in range(len(minInds))]

    return nearest5BusIds, nearest5BusDistance

def transformWGS84ToUMT(df):
    WGS84 = "+init=EPSG:4326"
    EPSG21781 = "+init=EPSG:2056"
    Proj_to_EPSG21781= pyproj.Proj(EPSG21781)
    UMTmeasurementGPS= df[['lng','lat']].apply(lambda row: list(Proj_to_EPSG21781(row[0],row[1])),axis=1).values
    return UMTmeasurementGPS

def returnGISFeaturesUTM(df, stopstxt_df, stoptimestxt_df, UMTstopsGPS):
    """
    :param df:  data[['lat','lng', 'timestamp']],
    :return:
    """
    timestamp_df_s = df['timestamp'] / 1000
    # nearest5BusIdsAndDist_df = get5nearestBusIDAndDistance(
    #     df['lat'],
    #     df['lng'],
    #     stopstxt_df
    # )
    UMTmeasurement= transformWGS84ToUMT(df)
    start = time.time()
    nearest5Ids, nearest5Dists = vectorizedUTMNearest5BusIdsAndDist(UMTmeasurement, UMTstopsGPS, stopstxt_df)
    print(" get5NearestBusIDsAndDistance took {} seconds".format((start-time.time())))
    YYYYMMDD = getYYYYMMDD(timestamp_df_s.iloc[0])
    min5TimesToDeparture_s_df = getTimeToDepartureUTM(nearest5Ids, timestamp_df_s, YYYYMMDD, stoptimestxt_df)
    GISFeatures = nearest5Dists.extend(min5TimesToDeparture_s_df)
    # GISFeatures_df = pd.concat([nearest5BusIdsAndDist_df[['dist1','dist2','dist3','dist4','dist5']],min5TimesToDeparture_s_df], axis=1)
    return GISFeatures


def returnGISFeatures(df):
    """
    :param df:  data[['lat','lng', 'timestamp']],
    :return:
    """
    os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\2018 GTFS zvv google transit')
    stopstxt_df = pd.read_csv('stops.txt')[
        [
            'stop_id',
            'stop_lat',
            'stop_lon'
        ]
    ]

    stoptimestxt_df = pd.read_csv('stop_times.txt')[
        [
            'departure_time',
            'stop_id'
        ]
    ]

    timestamp_df_s = df['timestamp'] / 1000

    nearest5BusIdsAndDist_df = get5nearestBusIDAndDistance(
        df['lat'],
        df['lng'],
        stopstxt_df
    )

    YYYYMMDD = getYYYYMMDD(timestamp_df_s.iloc[0])
    min5TimesToDeparture_s_df = getMin5TimeToDepartureFeatureDf(nearest5BusIdsAndDist_df, timestamp_df_s, YYYYMMDD, stoptimestxt_df)
    GISFeatures_df = pd.concat([nearest5BusIdsAndDist_df[['dist1','dist2','dist3','dist4','dist5']],min5TimesToDeparture_s_df], axis=1)
    return GISFeatures_df

# #### Unit Testing ####
# def testSetup():
#     os.chdir('C:\\Users\\lucas\\OneDrive\\masters_thesis\\2018 GTFS zvv google transit')
#     stopstxt_df = pd.read_csv('stops.txt')[
#         [
#             'stop_id',
#             'stop_lat',
#             'stop_lon'
#         ]
#     ]
#
#     stoptimestxt_df = pd.read_csv('stop_times.txt')[
#         [
#             'departure_time',
#             'stop_id'
#         ]
#     ]
#
#     timestamps = [
#     1517423894,
#     1517423894,
#     1517423895,
#     1517423896,
#     1517423897,
#     1517423898,
#     1517423899,
#     1517423900,
#     1517423901,
#     1517423902,
#     1517423903,
#     1517423904,
#     1517423905,
#     1517423906,
#     1517423895,
#     1517423896,
#     1517423897,
#     1517423898,
#     1517423899,
#     1517423900,
#     1517423901,
#     1517423902,
#     1517423903,
#     1517423904,
#     1517423905,
#     1517423906
#     ]
#
#     lng=[4.7547,4.767,5.4567547,6.7676,7.467,8.46576,7.45623,4.5467,7.4525,8.24356,7.2345,6.74357,5.234,4.7547,4.767,5.4567547,6.7676,7.467,8.46576,7.45623,4.5467,7.4525,8.24356,7.2345,6.74357,5.2345]
#     lat=[46.345,44.34,45.3456,46.56,45.3457,45.34,43.452,42.2345,45.6572,44.5463,43.76547,45.367,45.2345,46.345,44.34,45.3456,46.56,45.3457,45.34,43.452,42.2345,45.6572,44.5463,43.76547,45.367,45.2345]
#     gps_df = pd.DataFrame({'lat':lat, 'lng':lng, 'timestamp_ms':timestamps})
#     return stopstxt_df, stoptimestxt_df, gps_df

# if __name__ == '__main__':
#    stopstxt_df, stoptimestxt_df, gps_df = testSetup()
#    gis = returnGISFeatures(gps_df)
#    print(gis)

