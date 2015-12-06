import pandas as pd
import numpy as np
import os
import json
import pprint
import matplotlib.pylab as pylab
from pylab import plot,show
import numpy as np
import cPickle as pickle
import time
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import math
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold, RFE

''' Business Attributes '''
   
attrs = [
         u'attributes_Accepts Credit Cards', 
         u'attributes_Accepts Insurance',
         u'attributes_Ages Allowed', 
         u'attributes_Alcohol',
         u'attributes_Ambience', 
         u'attributes_Attire', 
         u'attributes_BYOB',
         u'attributes_BYOB/Corkage', 
         u'attributes_By Appointment Only',
         u'attributes_Caters', 
         u'attributes_Coat Check',
         u'attributes_Corkage', 
         u'attributes_Delivery',
         u'attributes_Dietary Restrictions', 
         u'attributes_Dogs Allowed',
         u'attributes_Drive-Thru', 
         u'attributes_Good For Dancing', 
         u'attributes_Good For Groups',
         u'attributes_Good For Kids', 
         u'attributes_Hair Types Specialized In', 
         u'attributes_Happy Hour',
         u'attributes_Has TV', 
         u'attributes_Music',
         u'attributes_Noise Level', 
         u'attributes_Open 24 Hours',
         u'attributes_Order at Counter', 
         u'attributes_Outdoor Seating',
         u'attributes_Parking', 
         u'attributes_Payment Types',
         u'attributes_Price Range', 
         u'attributes_Smoking',
         u'attributes_Take-out', 
         u'attributes_Takes Reservations',
         u'attributes_Waiter Service', 
         u'attributes_Wheelchair Accessible',
         u'attributes_Wi-Fi', 
         u'categories',
         u'hours_Friday', 
         u'hours_Monday', 
         u'hours_Saturday',
         u'hours_Sunday', 
         u'hours_Thursday', 
         u'hours_Tuesday',
         u'hours_Wednesday',
        ]

def getHourAttr(old, new):
    if old[u'close'] == new[u'close'] and old[u'open'] == new[u'open']:
        return 1
    else:
        return -1

def getParkingAttr(old, new):
    parking_keys = [u'garage', u'street', u'validated', u'lot', u'valet']
    '''if np.isnan(old['attributes_Parking']) or math.isnan(new['attributes_Parking']):
        return -1
    '''
    for key in parking_keys:
        try:
            if old['attributes_Parking'][key]!= new['attributes_Parking'][key]:
                return -1
        except KeyError, e:
            '''print 'I got a KeyError - reason "%s"' % str(e)
            print "Old: ", old
            print "New: ", new
            break'''
            return -1
    return 1

def getCategoryAttr(old, new):
    #print old, new
    oldCategories = set(old.encode('utf8').split(','))
    newCategories = set(new.encode('utf8').split(','))
    L = len(oldCategories.intersection(newCategories))
    return L
    
def getVector(old, new):
    res = []
    old_count, new_count = 0, 0
    for attr in attrs:
            
        if pd.isnull(old[attr]) and pd.isnull(new[attr]):
            res.append(1)
            continue
            
        elif pd.isnull(old[attr]) or pd.isnull(new[attr]):
            res.append(-1)
            continue
            
        if 'hours_' in attr:
            res.append(getHourAttr(old[attr], new[attr]))
            continue
            
        if attr == u'categories':
            res.append(getCategoryAttr(old[attr], new[attr]))
            continue
            
        if attr == u'attributes_Parking':
            res.append(getParkingAttr(old, new))
            continue

        elif old[attr] == new[attr]:
            res.append(1)
        else:
            res.append(-1)        
    return res

import datetime
''' Helper for Date in Pandas'''
def todate(d):
    return datetime.datetime.strptime(d, '%Y-%m-%d')


''' Calculate mean_after - mean_before : returns +1 if mean improves -1 if mean reduces'''
def calcAverage(df):
    #print "Printing DF", df
    df_before = df[df['before'] == True]
    df_after = df[df['before'] == False]
    
    #print "before", df_before
    #print "after", df_after
    
    l1 = len(df_before['stars'])
    l2 = len(df_after['stars'])

    if l1 == 0 or l2 == 0:
        return -1
    else:
        a = sum(df_before['stars']) / (len(df_before['stars'])*1.0)
        b = sum(df_after['stars']) / (len(df_after['stars'])*1.0)
        if a - b > 0:
            return -1
        else:
            return 1
        
def createTrainingData(cluster=0, neighbour=u'Anthem', date_range=60, sampleAll=False, maxsamples=10000):
    # Use cluster 0,1 for training
    X, Y = [], []

    diff = datetime.timedelta(days=date_range)
    pd_business_cluster =  pd_lasvegas[pd_lasvegas['cluster'] == cluster]
    pd_old_business = pd_business_cluster[pd_business_cluster['new'] == False]
    #pd_old_business_Y = pd_business_cluster[["business_id", "review_date", "stars"]] # This is to simplify Y calculation
    pd_new_business = pd_business_cluster[pd_business_cluster['new'] == True]
    gb_new_business = pd_new_business.groupby('business_id')
    gb_old_business = pd_old_business.groupby('business_id')

    
    #print "Calculating vector for neighbour ", neighbour
    start_time = time.time()
    for new_business_id, new_business_details in gb_new_business:
        start_date = todate(pd_new_business.iloc[0]['start_date'])
        new_business = new_business_details.iloc[0]
        
        ''' Calculating Y for Current New Business '''
        # Getting all old business reviews between start_date -diff to start_date + diff
        pd_old_business_tempY = pd_old_business[pd_old_business.review_date.apply(todate) <= start_date + diff]
        pd_old_business_tempY = pd_old_business_tempY[pd_old_business_tempY.review_date.apply(todate) >= start_date - diff]
        
        # Labelling All Reviews Before Start Date as True
        pd_old_business_tempY['before'] = (pd_old_business_tempY['review_date'].apply(todate) < start_date)
        
        #print pd_old_business_tempY.groupby('business_id')
        gb_old_business_tempY = pd_old_business_tempY.groupby(['business_id'])
        temp_y = gb_old_business_tempY.apply(calcAverage).values.tolist()
        
        ''' Calculating X for Current New Business '''
        temp_x = []
        for old_business_id, old_business_details in gb_old_business_tempY:
            old_business = old_business_details.iloc[0]
            #print old_business
            temp_x.append(getVector(old_business, new_business))
            
            
        X.extend(temp_x)
        Y.extend(temp_y)
        
        if sampleAll == False and len(X) > maxsamples:
            break

    end_time = time.time()
    #print "Time taken for neighbour ", neighbour, " is ", end_time - start_time
    #print "Completed calculating vector for neighbour ", neighbour
    return X, Y

from functools import partial

def getFeatures(flags):
    new_attrs = []
    for idx, flag in enumerate(flags):
        if flag:
            new_attrs.append( attrs[idx] )
    print new_attrs

logistic = linear_model.LogisticRegression()
clf = SVC(kernel='rbf')

result = {}
for k in range(6, 18):
    result[k] = 0
    pd_lasvegas = pickle.load( open('pd_lasvegas_'+str(k)+'.pkl', 'rb'))
    pd_lasvegas = pd_lasvegas[np.isfinite(pd_lasvegas[u'attributes_Price Range'])]
    pd_lasvegas.rename(columns={'date_x': 'review_date'}, inplace=True)
    pd_lasvegas.rename(columns={'date_y': 'start_date'}, inplace=True)
    pd_lasvegas.rename(columns={'stars_y': 'stars'}, inplace=True)
    pd_lasvegas.reset_index(level=0, inplace=True)
    pd_old_business_vegas = pd_lasvegas[pd_lasvegas['new'] == False]
    pd_new_business_vegas = pd_lasvegas[pd_lasvegas['new'] == True]
    for c in range(k):
        X, Y = createTrainingData(cluster=c, sampleAll=False, maxsamples=20000)
        X_np, Y_np = np.array(X), np.array(Y)
        scores = cross_validation.cross_val_score(logistic, X_np, Y_np, cv=5)
        result[k] += scores.mean()
        print "For k =", k , " and cluster ", c, " of size ", len(X), " score is ", scores.mean()
    result[k] /= k

for k in result:
    print "Cluster ", k, " Score: ", result[k]
