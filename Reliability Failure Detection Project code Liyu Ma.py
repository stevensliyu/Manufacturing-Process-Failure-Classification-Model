#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:14:29 2017

@author: lma
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, recall_score, precision_score

"""import srst process data"""

srst=pd.read_csv('xxx/e2e_table.csv')
def f1(data):
    if data['HI_IDD_SIG_900_ID'] > 0.05 or data['HI_IDD_SIG_900_MD'] > 0.05 or data['HI_IDD_SIG_900_OD'] > 0.05 or data['HI_IDD_RNG_900_ID'] > 12.5 or data['HI_IDD_RNG_900_MD'] > 12.5 or data['HI_IDD_RNG_900_OD'] > 12.5:
        val=1
    else:
        val=0
    return val
srst['val']=srst.apply(f1,axis=1)
srst1=srst[srst['val']==1][['HDDSN','LHD']]
srst1 = srst1.drop_duplicates(subset=['HDDSN','LHD'])

"""import function process data"""

function=pd.read_csv('xxx/e2e_quasi.csv')
def f2(data):
    if data['HI_IDD_SIG_300_ID'] > 0.05 or data['HI_IDD_SIG_300_MD'] > 0.05 or data['HI_IDD_SIG_300_OD'] > 0.05 or data['HI_IDD_RNG_300_ID'] > 12.5 or data['HI_IDD_RNG_300_MD'] > 12.5 or data['HI_IDD_RNG_300_OD'] > 12.5:
        val=1
    else:
        val=0
    return val
function['val']=function.apply(f2,axis=1)

"""select data according customer's requirement"""

df=function[function['TESTCODEC']=='VL6X717']
Date_maxes = df.groupby(['HDDSN', 'LHD']).ENDDATE.transform(max)
df1 = df[df.ENDDATE == Date_maxes]

del df1['val']

""" get failures's Manufacturig ID in SRST process and Funcion Process"""

function1=function[function['val']==1][['HDDSN','LHD']]
function1 = function1.drop_duplicates(subset=['HDDSN','LHD'])
function1.shape
frames = [srst1, function1]
failure = pd.concat(frames)

failure = failure.drop_duplicates(subset=['HDDSN','LHD'])

""" get Non repeated final data for failure and passer according their manufacturing ID"""

keys = ['HDDSN', 'LHD']
i1 = df1.set_index(keys).index
i2 = failure.set_index(keys).index
df2=df1[~i1.isin(i2)]
df3=df1[i1.isin(i2)]
df2['Judge']=0
df3['Judge']=1
frames1=[df2, df3]
data = pd.concat(frames1)

data['ID']=data['HDDSN'] + data['LHD'].map(str)

data.set_index('ID',inplace=True)


""" Select Process Features and Label """


data1=data[['cold_resistance_2_op2451',
'uqasym_3_op2451',
'amplitude_4_op2451',
'uqasym_sig_7_op2451',
'hot_resistance_8_op2451',
'amplitude_sig_14_op2451',
'write_coil_res_16_op2451',
'zero_rng_40_op2451',
'amplitude_rng_41_op2451',
'uqasym_rng_42_op2451',
'kink_md52_82_op2451',
'idema_asymmetry_92_op2451',
'kink_b1_238_op2451',
'kink_b2_239_op2451',
'kink_b3_240_op2451',
'test_time_241_op2451',
'rhot_sig_1007_op2451',
'rcold_sig_1008_op2451',
'rcoil_sig_1009_op2451',
'asym_sig_1015_op2451',
'zero_sig_1017_op2451',
'asym_rng_1021_op2451',
'neg_zero_1043_op2451',
'neg_amp_1044_op2451',
'pos_zero_1045_op2451',
'pos_amp_1046_op2451',
'zero_1047_op2451',
'tfc_dcr_res_1075_op2451',
'tfc_dcr_sig_1077_op2451',
'tck_bhjump_1092_op2451',
'tck_bhj_mag_1093_op2451',
'tck_hyst_1094_op2451',
'tck_hystmag_1095_op2451',
'tck_md52_fld_1098_op2451',
'channel_1124_op2451',
'tfc_amp_1127_op2451',
'tfc_asym_1128_op2451',
'tfc_amp_sig_1129_op2451',
'tfc_asym_sig_1130_op2451',
'tfc_amp_rng_1131_op2451',
'tfc_asym_rng_1132_op2451',
'amp_over_rc_1133_op2451',
'raw_amplitude_1134_op2451',
'delta_tfc_amp_1140_op2451',
'delta_tfc_asym_1141_op2451',
'ecs_1234_op2451',
'ecs_sig_1235_op2451',
'tfc_amp_rng_n_1407_op2451',
'adn_sig_n_1435_op2451',
'adn_ws_n_1436_op2451',
'adn_rng_n_1437_op2451',
'adn_mean_n_1438_op2451',
'adn_max_n_1439_op2451',
'adn_min_n_1440_op2451',
'adn_sig_1441_op2451',
'adn_ws_1442_op2451',
'adn_rng_1443_op2451',
'adn_mean_1444_op2451',
'adn_max_1445_op2451',
'adn_min_1446_op2451',
'adn_vb_1447_op2451',
'adn_gc_1448_op2451',
'adn_hwss_1449_op2451',
'adn_ch_1450_op2451',
'adn_skew_1451_op2451',
'adn_kurt_1452_op2451',
'adn_mode_1453_op2451',
'adn_match_1454_op2451',
'adn2_sig_n_1455_op2451',
'adn2_ws_n_1456_op2451',
'adn2_rng_n_1457_op2451',
'adn2_mean_n_1458_op2451',
'adn2_max_n_1459_op2451',
'adn2_min_n_1460_op2451',
'adn2_sig_1461_op2451',
'adn2_ws_1462_op2451',
'adn2_rng_1463_op2451',
'adn2_mean_1464_op2451',
'adn2_max_1465_op2451',
'adn2_min_1466_op2451',
'adn2_vb_1467_op2451',
'adn2_gc_1468_op2451',
'adn2_hwss_1469_op2451',
'adn2_ch_1470_op2451',
'adn2_skew_1471_op2451',
'adn2_kurt_1472_op2451',
'adn2_mode_1473_op2451',
'adn2_match_1474_op2451',
'mxhfrngn2_n_1476_op2451',
'delta_rng_n_1478_op2451',
'deltatfcasymrng_1479_op2451',
'ad_qsnr_1486_op2451',
'ad_qsnr2_1487_op2451',
'ad_rng_qsnr_1490_op2451',
'ad_rng_qsnr2_1491_op2451',
'amp_rng_n_1492_op2451',
'md52_amp_1493_op2451',
'delta_ad_rng_n_1500_op2451',
'deltatfcamprng_1506_op2451',
'deltatfcampsig_1507_op2451',
'deltatfcasymsig_1508_op2451',
'delta_ad_skew_n_1601_op2451',
'delta_ad_kurt_n_1602_op2451',
'tc_amp_1692_op2451',
'tc_asym_1693_op2451',
'tc_asymuq_1694_op2451',
'tc_bjmp_1695_op2451',
'tc_ampave_1701_op2451',
'tc_amprng_1702_op2451',
'tc_ampsig_1703_op2451',
'tc_asymave_1709_op2451',
'tc_asymrng_1710_op2451',
'tc_asymsig_1711_op2451',
'tc_bjmpmax_1729_op2451',
'tc_bjmpmin_1730_op2451',
'tc_bjmpsig_1731_op2451',
'tc_amp_20mw_1732_op2451',
'tc_asym_20mw_1733_op2451',
'tc_asymuq_20mw_1734_op2451',
'tc_bjmp_20mw_1735_op2451',
'tc_ampave_20mw_1741_op2451',
'tc_amprng_20mw_1742_op2451',
'tc_ampsig_20mw_1743_op2451',
'tc_asymave_20mw_1749_op2451',
'tc_asymrng_20mw_1750_op2451',
'tc_asymsig_20mw_1751_op2451',
'tc_bjmpmax_20mw_1769_op2451',
'tc_bjmpmin_20mw_1770_op2451',
'tc_bjmpsig_20mw_1771_op2451',
'tc_amp_40mw_1772_op2451',
'tc_asym_40mw_1773_op2451',
'tc_asymuq_40mw_1774_op2451',
'tc_bjmp_40mw_1775_op2451',
'tc_ampave_40mw_1781_op2451',
'tc_amprng_40mw_1782_op2451',
'tc_ampsig_40mw_1783_op2451',
'tc_asymave_40mw_1789_op2451',
'tc_asymrng_40mw_1790_op2451',
'tc_asymsig_40mw_1791_op2451',
'tc_bjmpmax_40mw_1809_op2451',
'tc_bjmpmin_40mw_1810_op2451',
'tc_bjmpsig_40mw_1811_op2451',
'kink_amp_1825_op2451',
'kink_asym_1826_op2451',
'rms_sig_ss2_1845_op2451',
'rms_sig_n_ss2_1846_op2451',
'median_ss2_1847_op2451',
'median_n_ss2_1848_op2451',
'sig_ss2_1849_op2451',
'sig_n_ss2_1850_op2451',
'min_ss2_1851_op2451',
'min_n_ss2_1852_op2451',
'max_ss2_1853_op2451',
'max_n_ss2_1854_op2451',
'rng_ss2_1855_op2451',
'rng_n_ss2_1856_op2451',
'skew_ss2_1857_op2451',
'kurt_ss2_1858_op2451',
'sm_min_ss2_1859_op2451',
'sm_min_n_ss2_1860_op2451',
'sm_max_ss2_1861_op2451',
'sm_max_n_ss2_1862_op2451',
'sm_rng_ss2_1863_op2451',
'sm_rng_n_ss2_1864_op2451',
'rng_avg_ss2_1865_op2451',
'rng_avg_n_ss2_1866_op2451',
'rng_sig_ss2_1867_op2451',
'rng_sig_n_ss2_1868_op2451',
'rng_max_ss2_1869_op2451',
'rng_max_n_ss2_1870_op2451',
'rng_fld_ss2_1871_op2451',
'avg_avg_ss2_1872_op2451',
'avg_avg_n_ss2_1873_op2451',
'avg_min_ss2_1874_op2451',
'avg_min_n_ss2_1875_op2451',
'min_fld_ss2_1876_op2451',
'avg_max_ss2_1877_op2451',
'avg_max_n_ss2_1878_op2451',
'max_fld_ss2_1879_op2451',
'avg_rng_ss2_1880_op2451',
'avg_rng_n_ss2_1881_op2451',
'avg_mod_ss2_1882_op2451',
'rms_sig_ss_1883_op2451',
'rms_sig_n_ss_1884_op2451',
'median_ss_1885_op2451',
'median_n_ss_1886_op2451',
'sig_ss_1887_op2451',
'sig_n_ss_1888_op2451',
'min_ss_1889_op2451',
'min_n_ss_1890_op2451',
'max_ss_1891_op2451',
'max_n_ss_1892_op2451',
'rng_ss_1893_op2451',
'rng_n_ss_1894_op2451',
'skew_ss_1895_op2451',
'kurt_ss_1896_op2451',
'sm_min_ss_1897_op2451',
'sm_min_n_ss_1898_op2451',
'sm_max_ss_1899_op2451',
'sm_max_n_ss_1900_op2451',
'sm_rng_ss_1901_op2451',
'sm_rng_n_ss_1902_op2451',
'rng_avg_ss_1903_op2451',
'rng_avg_n_ss_1904_op2451',
'rng_sig_ss_1905_op2451',
'rng_sig_n_ss_1906_op2451',
'rng_max_ss_1907_op2451',
'rng_max_n_ss_1908_op2451',
'rng_fld_ss_1909_op2451',
'avg_avg_ss_1910_op2451',
'avg_avg_n_ss_1911_op2451',
'avg_min_ss_1912_op2451',
'avg_min_n_ss_1913_op2451',
'min_fld_ss_1914_op2451',
'avg_max_ss_1915_op2451',
'avg_max_n_ss_1916_op2451',
'max_fld_ss_1917_op2451',
'avg_rng_ss_1918_op2451',
'avg_rng_n_ss_1919_op2451',
'avg_mod_ss_1920_op2451',
'std_sigma_ss2_1921_op2451',
'std_sigma_n_ss2_1922_op2451',
'ave_ss2_1923_op2451',
'ave_n_ss2_1924_op2451',
'std_sigma_ss_1925_op2451',
'std_sigma_n_ss_1926_op2451',
'ave_ss_1927_op2451',
'ave_n_ss_1928_op2451',
'proberetryprobe_5555_op2451',
'proberetrycount_5556_op2451',
'proberetrycode_5557_op2451',
'Judge'
]]

""" delete missing values """
data2=data1.dropna()

""" delete columns with single value """
nunique=data2.apply(pd.Series.nunique)
cols_to_drop=nunique[nunique==1].index
data3=data2.drop(cols_to_drop,axis=1)

data3.shape

""" delete columns with two values, testing time feature """

data4=data3.drop(['test_time_241_op2451',
                 'adn_vb_1447_op2451',
                 'adn2_vb_1467_op2451',
                 'adn2_match_1474_op2451',
                 'proberetryprobe_5555_op2451',
                 'proberetrycount_5556_op2451',
                 'proberetrycode_5557_op2451'],axis=1)



#data4_x=data4.loc[:,data4.columns!='Judge']

"""correlation checking"""

import matplotlib.pyplot as plt
cor=np.corrcoef(data4_x.T)
type(cor)
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(cor, vmin=0.85, vmax=1,cmap=plt.cm.Spectral_r)


#data4.to_csv('xxx/data4.csv')

""" Features given by running VIF algorithm in R studio, this step is to remove multicollinearity """

data5=data4[[
'hot_resistance_8_op2451',
'amplitude_sig_14_op2451',
'write_coil_res_16_op2451',
'zero_rng_40_op2451',
'uqasym_rng_42_op2451',
'idema_asymmetry_92_op2451',
'kink_b2_239_op2451',
'kink_b3_240_op2451',
'rhot_sig_1007_op2451',
'rcold_sig_1008_op2451',
'rcoil_sig_1009_op2451',
'neg_zero_1043_op2451',
'tfc_dcr_res_1075_op2451',
'tfc_dcr_sig_1077_op2451',
'tck_bhjump_1092_op2451',
'tck_bhj_mag_1093_op2451',
'tck_hyst_1094_op2451',
'tck_hystmag_1095_op2451',
'tck_md52_fld_1098_op2451',
'tfc_amp_sig_1129_op2451',
'tfc_asym_rng_1132_op2451',
'delta_tfc_amp_1140_op2451',
'delta_tfc_asym_1141_op2451',
'ecs_1234_op2451',
'ecs_sig_1235_op2451',
'tfc_amp_rng_n_1407_op2451',
'adn_min_n_1440_op2451',
'adn_mean_1444_op2451',
'adn_max_1445_op2451',
'adn_hwss_1449_op2451',
'adn_skew_1451_op2451',
'adn_kurt_1452_op2451',
'adn_mode_1453_op2451',
'adn_match_1454_op2451',
'adn2_ws_n_1456_op2451',
'adn2_mean_n_1458_op2451',
'adn2_max_1465_op2451',
'adn2_min_1466_op2451',
'adn2_gc_1468_op2451',
'adn2_skew_1471_op2451',
'adn2_kurt_1472_op2451',
'adn2_mode_1473_op2451',
'ad_qsnr_1486_op2451',
'amp_rng_n_1492_op2451',
'tc_amprng_1702_op2451',
'tc_asymrng_1710_op2451',
'tc_bjmpmax_1729_op2451',
'tc_bjmpmin_1730_op2451',
'tc_bjmpsig_1731_op2451',
'tc_bjmp_20mw_1735_op2451',
'tc_amprng_20mw_1742_op2451',
'tc_asymrng_20mw_1750_op2451',
'tc_bjmpmax_20mw_1769_op2451',
'tc_bjmpmin_20mw_1770_op2451',
'tc_amp_40mw_1772_op2451',
'tc_asymuq_40mw_1774_op2451',
'tc_amprng_40mw_1782_op2451',
'tc_asymrng_40mw_1790_op2451',
'tc_bjmpmax_40mw_1809_op2451',
'tc_bjmpmin_40mw_1810_op2451',
'tc_bjmpsig_40mw_1811_op2451',
'median_ss2_1847_op2451',
'min_n_ss2_1852_op2451',
'skew_ss2_1857_op2451',
'kurt_ss2_1858_op2451',
'rng_avg_ss2_1865_op2451',
'rng_max_n_ss2_1870_op2451',
'rng_fld_ss2_1871_op2451',
'min_fld_ss2_1876_op2451',
'max_fld_ss2_1879_op2451',
'avg_mod_ss2_1882_op2451',
'min_n_ss_1890_op2451',
'skew_ss_1895_op2451',
'kurt_ss_1896_op2451',
'rng_avg_ss_1903_op2451',
'rng_sig_ss_1905_op2451',
'rng_max_n_ss_1908_op2451',
'rng_fld_ss_1909_op2451',
'min_fld_ss_1914_op2451',
'max_fld_ss_1917_op2451',
'avg_mod_ss_1920_op2451',
'Judge'
]]


# Anomaly Detection with Isolation Forest because high demensions of the data set

data6 = data5.drop('Judge',axis=1)
train, test= sklearn.cross_validation.train_test_split(data5, train_size = 0.6, random_state=42)

train['judge'] = np.where(train['Judge']==1, -1,1)
test['judge']=np.where(test['Judge']==1,-1,1)

X_train = train.drop(['judge','Judge'],axis=1)
X_test = test.drop(['judge','Judge'],axis=1)

y1_train = train['judge'].values
y1_test = test['judge'].values


from sklearn.ensemble import IsolationForest
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, precision_score,f1_score, precision_recall_curve

# Unsupervised Learning/Isolation Forest

clf = IsolationForest(n_estimators = 200, max_samples = 18680, contamination=0.05)

clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

clf_recall_test = recall_score(y1_test, y_pred_test)
clf_precision_test = precision_score(y1_test, y_pred_test)
#clf_acc_test = clf.score(test_temp_x, y_test)
clf_f1_test = f1_score(y1_test, y_pred_test)
    
#print ('Accuracy: %f \n' % clf.score(test_temp_x, y_test))
#print ("AUC: %f \n" % clf_auc)
print ("Recall Score: %f \n" % clf_recall_test)
print ("Precision Score: %f \n" % clf_precision_test)
print ("Confusion matrix:")
print (pd.crosstab(y1_test, y_pred_test))


# Local Outlier Detection

from sklearn import preprocessing
from sklearn.decomposition import PCA

scaler = preprocessing.StandardScaler(with_std=False)
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)

"""
sklearn_pca = PCA(.98)
sklearn_pca.fit(X_train)
X_train_pca = sklearn_pca.transform(X_train)
X_test_pca  = sklearn_pca.transform(X_test)

"""
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, leaf_size=30)

clf.fit(X_train)
y_pred_train = clf.fit_predict(X_train)
y_pred_test = clf.fit_predict(X_test)

clf_recall_test = recall_score(y1_test, y_pred_test)
clf_precision_test = precision_score(y1_test, y_pred_test)
#clf_acc_test = clf.score(test_temp_x, y_test)
clf_f1_test = f1_score(y1_test, y_pred_test)
    
#print ('Accuracy: %f \n' % clf.score(test_temp_x, y_test))
#print ("AUC: %f \n" % clf_auc)
print ("Recall Score: %f \n" % clf_recall_test)
print ("Precision Score: %f \n" % clf_precision_test)
print ("Confusion matrix:")
print (pd.crosstab(y1_test, y_pred_test))

#Unsupervised Learning/OneClassSvm

pass_ = data5[data5['Judge'] == 0]
failure_ = data5[data5['Judge'] == 1]

pass_train, pass_test= sklearn.cross_validation.train_test_split(pass_, train_size = 0.8, random_state=42)

test_svm = pd.concat((failure_,pass_test),axis=0)


pass_train['judge'] = np.where(pass_train['Judge']==1, -1,1)
test_svm['judge']=np.where(test_svm['Judge']==1,-1,1)

X_train = pass_train.drop(['judge','Judge'],axis=1)
X_test = test_svm.drop(['judge','Judge'],axis=1)

y1_train = pass_train['judge']
y1_test = test_svm['judge']


from sklearn import svm

clf = svm.OneClassSVM(nu=0.8, kernel='poly',gamma=0.5)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

clf_recall_test = recall_score(y1_test, y_pred_test)
clf_precision_test = precision_score(y1_test, y_pred_test)
#clf_acc_test = clf.score(test_temp_x, y_test)
clf_f1_test = f1_score(y1_test, y_pred_test)
    
#print ('Accuracy: %f \n' % clf.score(test_temp_x, y_test))
#print ("AUC: %f \n" % clf_auc)
print ("Recall Score: %f \n" % clf_recall_test)
print ("Precision Score: %f \n" % clf_precision_test)
print ("Confusion matrix:")
print (pd.crosstab(y1_test, y_pred_test))



#Supervised Learning Method

data6= data5.reset_index(drop=True)

""" Class Frequency Visualization"""
import matplotlib.pyplot as plt

count_classes =pd.value_counts(data5['Judge'], sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title("Manufacturing Process Class Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

"""Train/Test Data Splitting"""
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
train, test = sklearn.cross_validation.train_test_split(data6, train_size = 0.6, random_state=42)

X=train.loc[:,train.columns != 'Judge'].values
Y=train['Judge'].values

X_=test.loc[:,test.columns != 'Judge'].values
Y_=test['Judge'].values

X1=train.loc[:,train.columns != 'Judge']
Y1=train['Judge']

X_1=test.loc[:,test.columns != 'Judge']
Y_1=test['Judge']

"""feature selection"""

from sklearn.ensemble import ExtraTreesClassifier
rf = ExtraTreesClassifier()
rf.fit(X1, Y1)

from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(rf, prefit=True)

X_new=model.transform(X1)

feature_idx = model.get_support()
feature_name = X1.columns[feature_idx]


X2= X1[feature_name]
X_2= X_1[feature_name]

X3= X1[feature_name].values
X_3= X_1[feature_name].values

""" 
visulize failure vs pass on (PCA1,PCA2)

from sklearn import preprocessing
from sklearn.decomposition import PCA


data6_x = data6.loc[:, data6.columns !='Judge']

Label= data6['Judge']

scaler=preprocessing.StandardScaler(with_std=True).fit_transform(data6_x.values)
pca =PCA(n_components=2)
pca.fit(scaler)

X_pca = pca.fit_transform(scaler)

colors =np.array([x for x in 'bgrcmyk'])
color=colors[Label.values]

plt.figure(figsize=(25,10))

plt.scatter(X_pca[:,0],X_pca[:,1], c=color)

plt.show()

"""

""" resampling method"""

TRAIN=pd.concat((X2,Y1),axis=1)

training_0 = TRAIN[TRAIN['Judge']==0]
training_1 = TRAIN[TRAIN['Judge']==1]
combined_training = pd.concat([training_1]*50,ignore_index=True).append(training_0)

X_train = combined_training.loc[:,combined_training.columns != 'Judge']
Y_train = combined_training['Judge']

""" Model Matrix """


# Base Logistic Regression
clf = LogisticRegression(penalty='l2',C=1)
#clf=AdaBoostClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(25,10),activation='relu')
clf.fit(X2,Y1)
clf_predict=clf.predict(X_2)

clf_prob = clf.predict_proba(X_2)[:,1]

clf_fpr, clf_tpr, clf_thresholds = roc_curve(Y_1, clf_prob)
clf_auc =  auc(clf_fpr, clf_tpr)
clf_recall = recall_score(Y_1, clf_predict)

clf_precision = precision_score(Y_1, clf_predict)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc
print "Recall Score: %f \n" % clf_recall
print "Precision Score: %f \n" % clf_precision
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict)

# Base Adaboosting Method

#clf = LogisticRegression(penalty='l2',C=1)
clf=AdaBoostClassifier(n_estimators=10)
#clf = MLPClassifier(hidden_layer_sizes=(25,10),activation='relu')
clf.fit(X2,Y1)
clf_predict1=clf.predict(X_2)

clf_prob1 = clf.predict_proba(X_2)[:,1]

clf_fpr1, clf_tpr1, clf_thresholds1 = roc_curve(Y_1, clf_prob1)
clf_auc1 =  auc(clf_fpr1, clf_tpr1)
clf_recall1 = recall_score(Y_1, clf_predict1)

clf_precision1 = precision_score(Y_1, clf_predict1)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc1
print "Recall Score: %f \n" % clf_recall1
print "Precision Score: %f \n" % clf_precision1
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict1)

# Base model Neural Network
clf = MLPClassifier(hidden_layer_sizes=(25,2),activation='logistic')
clf.fit(X2,Y1)
clf_predict2=clf.predict(X_2)

clf_prob2= clf.predict_proba(X_2)[:,1]

clf_fpr2, clf_tpr2, clf_thresholds2 = roc_curve(Y_1, clf_prob2)
clf_auc2 =  auc(clf_fpr2, clf_tpr2)
clf_recall2 = recall_score(Y_1, clf_predict2)

clf_precision2 = precision_score(Y_1, clf_predict2)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc2
print "Recall Score: %f \n" % clf_recall2
print "Precision Score: %f \n" % clf_precision2
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict2)

# Weighted class 1 model + LR

clf = LogisticRegression(penalty='l2',C=1)
clf.fit(X_train,Y_train)
clf_predict3=clf.predict(X_2)

clf_prob3 = clf.predict_proba(X_2)[:,1]

clf_fpr3, clf_tpr3, clf_thresholds3 = roc_curve(Y_1, clf_prob3)
clf_auc3 =  auc(clf_fpr3, clf_tpr3)
clf_recall3 = recall_score(Y_1, clf_predict3)

clf_precision3 = precision_score(Y_1, clf_predict3)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc3
print "Recall Score: %f \n" % clf_recall3
print "Precision Score: %f \n" % clf_precision3
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict3)

# Weighted class 1 + Ada
clf=AdaBoostClassifier(n_estimators=10)
#clf = MLPClassifier(hidden_layer_sizes=(25,10),activation='relu')
clf.fit(X_train,Y_train)
clf_predict4=clf.predict(X_2)

clf_prob4= clf.predict_proba(X_2)[:,1]

clf_fpr4, clf_tpr4, clf_thresholds4 = roc_curve(Y_1, clf_prob4)
clf_auc4 =  auc(clf_fpr4, clf_tpr4)
clf_recall4 = recall_score(Y_1, clf_predict4)

clf_precision4 = precision_score(Y_1, clf_predict4)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc4
print "Recall Score: %f \n" % clf_recall4
print "Precision Score: %f \n" % clf_precision4
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict4)


# Weighted class 1 + NN

clf = MLPClassifier(hidden_layer_sizes=(25,2),activation='logistic')
clf.fit(X_train,Y_train)
clf_predict5=clf.predict(X_2)

clf_prob5= clf.predict_proba(X_2)[:,1]

clf_fpr5, clf_tpr5, clf_thresholds5 = roc_curve(Y_1, clf_prob5)
clf_auc5 =  auc(clf_fpr5, clf_tpr5)
clf_recall5 = recall_score(Y_1, clf_predict5)

clf_precision5 = precision_score(Y_1, clf_predict5)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc5
print "Recall Score: %f \n" % clf_recall5
print "Precision Score: %f \n" % clf_precision5
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict5)



# ROC curve for above model

import matplotlib.pyplot as plt

plt.figure(0,figsize=(20,8))


#plt.plot(clf_fpr,clf_tpr, label = " Base Logistic Regression model, auc = "+str(clf_auc))
#plt.plot(clf_fpr1,clf_tpr1, label = " Base Adaboosting model, auc = "+str(clf_auc1))
#plt.plot(clf_fpr2,clf_tpr2, label = " Base Nerual Network model, auc = "+str(clf_auc2))
plt.plot(clf_fpr3,clf_tpr3, label = " Base LR model + Weighted class 1, auc = "+str(clf_auc3))
#plt.plot(clf_fpr4,clf_tpr4, label = " Base Adaboosting model + Weighted class 1, auc = "+str(clf_auc4))
#plt.plot(clf_fpr5,clf_tpr5, label = " Base Nerual Network model + Weighted class 1, auc = "+str(clf_auc5))
plt.plot(clf_fpr6,clf_tpr6, label = " LR + Ada + NN + Weighted class 1 + Voting Method, auc = "+str(clf_auc6))
plt.plot(clf_fpr13,clf_tpr13, label = " LR + Ada + NN + Weighted class 1 + StackEnsembling, auc = "+str(clf_auc13))
plt.legend(loc=0)

predict3=pd.DataFrame(clf_predict3)
predict4=pd.DataFrame(clf_predict4)
predict5=pd.DataFrame(clf_predict5)

# Y_1, predict3, predict4, predict5 are outputed for "&" voting

Y_1.to_csv('xxx/real.csv')
predict3.to_csv('xxx/LRweighted.csv')
predict4.to_csv('xxx/Adaweighted.csv')
predict5.to_csv('xxx/NNweighted.csv')

# majority voting method

from sklearn.ensemble import VotingClassifier


clf1= LogisticRegression(penalty='l2',C=1)
clf2= AdaBoostClassifier(n_estimators=10)
clf3= MLPClassifier(hidden_layer_sizes=(25,2),activation='logistic')


eclf = VotingClassifier(estimators=[('lg',clf1),('Ada',clf2),('NN',clf3)],weights=[10,1,1],voting='soft')
eclf.fit(X_train,Y_train)

clf_predict6=eclf.predict(X_2)

clf_prob6 = eclf.predict_proba(X_2)[:,1]

clf_fpr6, clf_tpr6, clf_thresholds6= roc_curve(Y_1, clf_prob6)
clf_auc6 =  auc(clf_fpr6, clf_tpr6)
clf_recall6 = recall_score(Y_1, clf_predict6)
clf_precision6 = precision_score(Y_1, clf_predict6)

print "Accuracy: %f \n" % eclf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc6
print "Recall Score: %f \n" % clf_recall6
print "Precision Score: %f \n" % clf_precision6
print "Confusion matrix:" 
print pd.crosstab(Y_1, clf_predict6)


# Stack Ensemble method

skf=StratifiedKFold(n_splits=3)
S=pd.DataFrame(data=None,columns=['predict_lr','predict_Ad','predict_NN'], index=X2.index)


for train_index, test_index in skf.split(X3,Y):
    
     print("Train:", len(train_index), "Test:", len(test_index))
     
     x_train,x_test = X[train_index], X[test_index]
     y_train, y_test= Y[train_index], Y[test_index]
     
     x_training_0 = x_train[np.where(y_train==0)]
     x_training_1 = x_train[np.where(y_train==1)]
     y_training_0 = y_train[np.where(y_train==0)]
     y_training_1 = y_train[np.where(y_train==1)]
     print(x_training_0.shape)
     print(y_training_0.shape)
     x1 = np.vstack([x_training_1]*50)
     y1 = np.hstack([y_training_1]*50)
     print(x1.shape)
     print(y1.shape)
     combined_training_x = np.concatenate((x1,x_training_0),axis=0)
     combined_training_y = np.concatenate((y1,y_training_0),axis=0)
     print(combined_training_x.shape)
     print(combined_training_y.shape)
     
     
     clf = LogisticRegression(penalty='l2',C=1)
    
     clf.fit(combined_training_x,combined_training_y)
     clf_predict7=clf.predict(x_test)
     clf_prob7 = clf.predict_proba(x_test)[:,1]

     clf_fpr7, clf_tpr7, clf_thresholds7 = roc_curve(y_test, clf_prob7)
     clf_auc7 =  auc(clf_fpr7, clf_tpr7)
     clf_recall7 = recall_score(y_test, clf_predict7)

     clf_precision7 = precision_score(y_test, clf_predict7)

     print "Accuracy: %f \n" % clf.score(x_test, y_test)
     print "AUC: %f \n" % clf_auc7
     print "Recall Score: %f \n" % clf_recall7
     print "Precision Score: %f \n" % clf_precision7
    
     
     clf1=AdaBoostClassifier(n_estimators=10)
     clf1.fit(combined_training_x,combined_training_y)
     clf_predict8=clf1.predict(x_test)

     clf_prob8 = clf1.predict_proba(x_test)[:,1]

     clf_fpr8, clf_tpr8, clf_thresholds8 = roc_curve(y_test, clf_prob8)
     clf_auc8 =  auc(clf_fpr8, clf_tpr8)
     clf_recall8 = recall_score(y_test, clf_predict8)

     clf_precision8 = precision_score(y_test, clf_predict8)
     print "Accuracy: %f \n" % clf1.score(x_test, y_test)
     print "AUC: %f \n" % clf_auc8
     print "Recall Score: %f \n" % clf_recall8
     print "Precision Score: %f \n" % clf_precision8
    
     
     clf2 = MLPClassifier(hidden_layer_sizes=(25,2),activation='logistic')
     clf2.fit(combined_training_x,combined_training_y)
     clf_predict9=clf2.predict(x_test)
     clf_prob9 = clf2.predict_proba(x_test)[:,1]

     clf_fpr9, clf_tpr9, clf_thresholds9 = roc_curve(y_test, clf_prob9)
     clf_auc9 =  auc(clf_fpr9, clf_tpr9)
     clf_recall9 = recall_score(y_test, clf_predict9)

     clf_precision9 = precision_score(y_test, clf_predict9)
     print "Accuracy: %f \n" % clf2.score(x_test, y_test)
     print "AUC: %f \n" % clf_auc9
     print "Recall Score: %f \n" % clf_recall9
     print "Precision Score: %f \n" % clf_precision9
    
     
     #clf_prob = clf.predict_proba(TEST)[:,1]
     
     
     
     S['predict_lr'].iloc[test_index]= clf_predict7
     S['predict_Ad'].iloc[test_index]= clf_predict8
     S['predict_NN'].iloc[test_index]= clf_predict9
     
    


S1=pd.concat((X2,S),axis=1)


P1=pd.concat((X2,Y1),axis=1)
training_0 = P1[P1['Judge']==0]
training_1 = P1[P1['Judge']==1]
P2 = pd.concat([training_1]*50,ignore_index=True).append(training_0)

P_X = P2.loc[:,P2.columns != 'Judge']
P_Y = P2['Judge']


clf = LogisticRegression(penalty='l2',C=1)
clf.fit(P_X,P_Y)
clf_predict10=clf.predict(X_3)
X_2['predict_lr']=clf_predict10

clf1=AdaBoostClassifier(n_estimators=10)
clf1.fit(P_X,P_Y)
clf_predict11=clf1.predict(X_3)
X_2['predict_Ad']=clf_predict11

clf2 = MLPClassifier(hidden_layer_sizes=(25,2),activation='logistic')
clf2.fit(P_X,P_Y)
clf_predict12=clf2.predict(X_3)
X_2['predict_NN']=clf_predict12

TEST1.columns

#X_S1=S1[['predict_lr','predict_Ad','predict_NN']].values
#X_S1=S1.loc[:,S1.columns != 'Judge']
#Y_S1=S1['Judge']'50=.
#X_S1_T=test[['predict_lr','predict_Ad','predict_NN']].values
#X_S1_T=test.loc[:,test.columns != 'Judge']
#Y_S1_T=test['Judge']




#clf = DecisionTreeClassifier(max_depth=2)
clf=LogisticRegression(penalty='l2',C=0.01)

#clf=SVC(probability=True)
clf.fit(S1,Y1)
clf_predict13=clf.predict(X_2)

clf_prob13 = clf.predict_proba(X_2)[:,1]

clf_fpr13, clf_tpr13, clf_thresholds13 = roc_curve(Y_1, clf_prob13)
clf_auc13 =  auc(clf_fpr13, clf_tpr13)
clf_recall13 = recall_score(Y_1, clf_predict13)

clf_precision13 = precision_score(Y_1, clf_predict13)

print "Accuracy: %f \n" % clf.score(X_2, Y_1)
print "AUC: %f \n" % clf_auc13
print "Recall Score: %f \n" % clf_recall13
print "Precision Score: %f \n" % clf_precision13
print "Confusion matrix:"
print pd.crosstab(Y_1, clf_predict13)

