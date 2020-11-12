# -*- coding: iso-8859-1 -*-

#How to make your own scan parameter file:
#1) On line 10, choose a unique filename.
#2) On lines 15-36, replace the expressions to the right of the assignment statements with the desired values (maintain formatting)
#3) Run this file from the /basic_gp_stuff/ directory.
#4)2019-10-30 clean up
import numpy as np
import epics
from epics import caget
import pickle

filename = 'tfgp_20191014_RBF.pkl'
#dict = {'LTS:H1:CurrentAO':25,'LTS:V1:CurrentAO':25,'LTS:H2:CurrentAO':25,'offst':0,'amp':1.0,'noise':0.01}
values=[0.33288468, 0.25384964, 0.44732893, 4.36670369, 0.284272 ,  0.38187369,  0.33507802 , 0.57132642,  0.22516421, 0.72504281, 0.92558641, 0.2516231,  0.2894494,  1.389410,  0.52856522, 0.29348572]
#values=[0.33288468, 0.25384964, 0.44732893, 1.15, 0.284272 ,  0.38187369,  0.33507802 , 0.57132642,  0.22516421, 0.72504281, 0.92558641, 0.2516231,  0.2894494,  1.389410,  0.52856522, 0.29348572]

devices=['L1:RG2:QM1:CurrentAO', 'L1:RG2:QM2:CurrentAO', 'L1:RG2:QM3:CurrentAO',
       'L1:RG2:QM4:CurrentAO', 'L1:QM3:CurrentAO', 'L1:QM4:CurrentAO',
       'L1:RG2:SC1:VL:CurrentAO', 'L1:RG2:SC2:HZ:CurrentAO',
       'L1:RG2:SC2:VL:CurrentAO', 'L1:RG2:SC3:HZ:CurrentAO',
       'L1:RG2:SC3:VL:CurrentAO', 'L1:SC3:HZ:CurrentAO', 'L1:SC3:VL:CurrentAO',
       'L1:QM5:CurrentAO', 'L1:SC4:HZ:CurrentAO', 'L1:SC4:VL:CurrentAO']

dict={}
size=len(values)
for i in range(size):
    dict[devices[i]]=values[i]
dict['amp']=0.09861835988630561
dict['noise']=0.0004952875307346928

print('save to tfgp...')
print(dict)
f = open(filename,'wb')
pickle.dump(dict,f)
f.close()


#trained from new scanned data on 10-26-2019
values= [0.22078988,0.26574095,0.47829405,1.00406362,0.14711107,0.24826348,0.37638048,0.66085644,0.38513674,0.96105928,0.49242131,0.20863498,0.2242516,0.54836268,0.07952513,0.20008924]
#noise_param_variance 0.00019078093245028939
#amp 0.14740807434028955
dict['amp']=0.14740807434028955
dict['noise']=0.00019078093245028939
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]

f = open('tfgp_20191026_RBF.pkl','wb')
pickle.dump(dict,f)
f.close()



#matern52
#length_scale_param [0.5301943  0.57974298 0.93287674 1.55738394 0.2781559  0.48676285  0.62080341 1.00700028 0.63068022 1.33504816 0.85157653 0.38468695  0.34187084 0.90039229 0.12904501 0.39346024]
#noise_param_variance 0.00015752787526062437
#amp 0.16033151005207602
dict={}
values= [0.5301943,0.57974298,0.93287674,1.55738394,0.2781559,0.48676285,0.62080341,1.00700028,0.63068022,1.33504816,0.85157653,0.38468695,0.34187084,0.90039229,0.12904501,0.39346024]
dict['amp']=0.16033151005207602
dict['noise']=0.00015752787526062437
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]

f = open('tfgp_20191026_matern52.pkl','wb')
pickle.dump(dict,f)
f.close()



#matern32
#$length_scale_param [1.14104442 1.15166326 1.93485995 3.53009842 0.5435236  1.00184452  1.17587559 1.91437433 1.20989785 2.43048938 2.24330837 0.79114557  0.64170614 1.7408325  0.24202489 0.87199787]
#$noise_param_variance 0.00014944309862160857
#$amp 0.3057118422862669
dict={}
values=[1.14104442,1.15166326,1.93485995,3.53009842,0.5435236,1.00184452,1.17587559,1.91437433,1.20989785,2.43048938,2.24330837,0.79114557,0.64170614,1.7408325,0.24202489,0.87199787]
dict['amp']=0.3057118422862669
dict['noise']=0.00014944309862160857
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]

f = open('tfgp_20191026_matern32.pkl','wb')
pickle.dump(dict,f)
f.close()


#trained from new scanned data on 10-26-2019, remove steering magnets

devices1=['L1:RG2:QM1:CurrentAO', 'L1:RG2:QM2:CurrentAO', 'L1:RG2:QM3:CurrentAO',
       'L1:RG2:QM4:CurrentAO', 'L1:QM3:CurrentAO', 'L1:QM4:CurrentAO',
       'L1:RG2:SC1:VL:CurrentAO', 'L1:RG2:SC2:HZ:CurrentAO',
       'L1:RG2:SC2:VL:CurrentAO', 'L1:RG2:SC3:HZ:CurrentAO',
       'L1:RG2:SC3:VL:CurrentAO', 
       'L1:QM5:CurrentAO']

#got data from gp trainer
values = [0.8246469,  0.51113118, 0.52074331, 0.43801243, 0.31416264, 0.57697836,
 0.67361431, 0.51629454, 0.61509133, 0.42665865, 0.29098141, 0.34906042]
rows = len(values)
dict={}
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.19788069543846426
dict['noise']=0.06565608478608254
#dict['noise']=0.00019  #change it manuall to see if it works
f = open('tfgp_20191026_RBF_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#above results are obtained with default gp kernel -- RBF

#matern32 kernel
#length_scale_param [0.28131369 0.31034553 0.60052556 0.64906617 0.21344806 0.33971465
# 0.36233996 0.35201099 0.32054675 0.54919454 0.57507362 0.44512081]
#noise_param_variance 0.05761468336008892
#amp 0.09213672488797832
values = [0.28131369,0.31034553,0.60052556,0.64906617,0.21344806,0.33971465,
 0.36233996,0.35201099,0.32054675,0.54919454,0.57507362,0.44512081]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.09213672488797832
dict['noise']=0.05761468336008892
f = open('tfgp_20191026_matern32_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()


#matern52 kernel
#length_scale_param [0.21638666 0.27405066 0.49069129 0.49208349 0.20101479 0.28311935   0.30456699 0.28351084 0.27222831 0.49656042 0.43469106 0.37042972]
#noise_param_variance 0.05948358250077242
#amp 0.1077475822322336
values=[0.21638666,0.27405066,0.49069129,0.49208349,0.20101479,0.28311935,0.30456699,0.28351084,0.27222831,0.49656042,0.43469106,0.37042972]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.1077475822322336
dict['noise']=0.05948358250077242
f = open('tfgp_20191026_matern52_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()


#combine 10/14 optimization data and 10/26 scan data /home/oxygen/SHANG/ML_APS/linac/L3ChargeOpt/opt-scan-1014-1026.xls
#16 magnets
#RBF
values=[0.24971038,0.24490193,0.45699345,0.97297992,0.17487721,0.26567337,0.37209686,0.65248962,0.30703547,0.90474733,0.75147228,0.2227032,0.25184649,0.55466075,0.11101448,0.21843818]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]
dict['amp']=0.1262021733
dict['noise']=0.0002695605258
f = open('tfgp_20191014_1026_RBF.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern52
values=[0.51214665,0.48110048,0.81999102,1.54183284,0.30177159,0.49942229,0.55309369,0.93705773,0.45985991,1.26207186,0.98024734,0.39016145,0.3532511,0.9193926,0.18280235,0.42117779]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]
dict['amp']=0.1439545448
dict['noise']=0.0002109903963
f = open('tfgp_20191014_1026_matern52.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern32
values=[1.07022477,0.93354679,1.67322035,3.22587812,0.5736346,1.01722377,0.99742879,1.73075448,0.86538688,2.26070738,2.21487763,0.77684175,0.6217432,1.79983342,0.34600119,0.86584808]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]
dict['amp']=0.2954768632
dict['noise']=0.0001978023533
f = open('tfgp_20191014_1026_matern32.pkl','wb')
pickle.dump(dict,f)
f.close()

#12 magnets (no steering magnets)
#RBF
values=[0.63292721,0.34059407,0.53702173,0.61800174,0.31128117,0.55711319,0.29905284,0.52616939,0.47259304,0.46013766,0.4674393,0.44260775]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.1251243992
dict['noise']=0.05296799903
f = open('tfgp_20191014_1026_RBF_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern52
values=[0.30823974,0.28222969,0.55869266,0.71698127,0.28021579,0.3611339,0.32149422,0.41768433,0.32538725,0.57953255,0.54075733,0.52191877]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.08501787965
dict['noise']=0.04858748309
f = open('tfgp_20191014_1026_matern52_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern32
values=[0.3674013,0.31782013,0.66383077,0.88690924,0.30086013,0.42965852,0.37296711,0.47404646,0.38055838,0.68137019,0.65078568,0.60926786]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.08022366695
dict['noise']=0.04663718527
f = open('tfgp_20191014_1026_matern32_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#add gaussian fit for 16 and 12
values=[0.7322836730022275, 0.4680323307269952, 1.0863090994868232, 2.464526634314797, 0.2275870574368614, 0.5218673479359833, 0.5311585257274893, 0.7224843830268624, 0.4620830977859818, 0.718981751508376, 4.039495633138642, 0.6769443976143563, 0.22242167458357542, 0.841175838942137, 1.8080668592247167, 0.24980528710636288]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices[i]]=values[i]
dict['amp']=0.1474239004
dict['noise']=0.0001905386176
f = open('guassian_fit_sqrt2.pkl','wb')
pickle.dump(dict,f)
f.close()

values=[0.7322836730022275, 0.4680323307269952, 1.0863090994868232, 2.464526634314797, 0.2275870574368614, 0.5218673479359833, 0.5311585257274893, 0.7224843830268624, 0.4620830977859818, 0.718981751508376, 4.039495633138642, 0.841175838942137]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.1978806954
dict['noise']=0.06565608479
f = open('guassian_fit_sqrt2_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()


#combine data from 10/14, 10/23, 10/26, 10/31 for 12 magnets (no steering)
#RBF
values=[0.58241933,0.29678635,0.48080855,3.43618106,0.26113353,0.52766097,0.40420108,0.30947095,0.26139892,0.52998378,0.5324202,0.50275636]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.0473160192
dict['noise']=0.04756546522
f = open('tfgp_20191014-1023-1026-1031_RBF_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern32
values=[0.3707367,0.27209057,0.6268962,1.43793187,0.24651184,0.43000364,0.37451929,0.42389007,0.30644531,0.62692922,0.75646539,0.63878361]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.04234193401
dict['noise']=0.0421605206
f = open('tfgp_20191014-1023-1026-1031_matern32_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()

#matern52
values=[0.34757096,0.2554463,0.56457023,1.31867706,0.23404222,0.39203008,0.34480126,0.3805012,0.28252844,0.56077706,0.6526772,0.57273871]
dict={}
rows=len(values)
for i in range(rows):
    dict[devices1[i]]=values[i]
dict['amp']=0.04389968406
dict['noise']=0.04318395429
f = open('tfgp_20191014-1023-1026-1031_matern52_nosteering.pkl','wb')
pickle.dump(dict,f)
f.close()
