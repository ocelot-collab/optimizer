
import os
import numpy as np
import scipy.io as sio


base_path = '/u1/lcls/matlab/data/2018/2018-01/'
quadlist = ['620', '640', '660', '680']
quadlist = sorted(['QUAD_LTU1_' + x + '_BCTRL' for x in quadlist])
gdet = 'GDET_FEE1_241_ENRCHSTBR'
energy = 'BEND_DMP1_400_BDES'

X = np.zeros((0,len(quadlist)+1))

for dir in os.listdir(base_path):
  path = base_path + dir + '/'
  for f in os.listdir(path):
    if f[:3]=='Oce':
      try: 
        rawdat = sio.loadmat(path+f)['data']
      except:
        continue
      if set(quadlist).issubset(set(rawdat.dtype.names)):
        y = rawdat[gdet][0][0]
        es = rawdat[energy][0][0]
        qs = [rawdat[q][0][0] for q in quadlist]
        shps = [x.shape[1] for x in qs]
        if y.shape[1] < 3 or min(shps) != max(shps) or shps[0] < 3:
          continue
        if y.shape[1] > shps[0]:
          y = y[:,:-1]
        new_stack = np.zeros((y.shape[1], X.shape[1])) 
        for i,q in enumerate(quadlist):
          new_stack[:,i] = qs[i] / es
        
        new_stack[:,-1] = y
        X = np.concatenate((X,new_stack),axis=0)

np.savetxt('ltus_enormed.csv', X) 
      

