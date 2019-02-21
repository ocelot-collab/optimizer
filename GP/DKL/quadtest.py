import numpy as np
import matplotlib.pyplot as plt

from dknet import NNRegressor
from dknet.layers import Dense,CovMat,Dropout,Parametrize
from dknet.optimizers import Adam,SciPyMin,SDProp

np.random.seed(1)
data = np.loadtxt('ltus_en.txt')
train_data = data[:1000]
test_full_data = data[1000:]
np.random.shuffle(test_full_data)
test_data = test_full_data[:1000]

which_qs = [2,3]
x_train = train_data[:,which_qs]
y_train = train_data[:,[-1]]
x_test = test_data[:,which_qs]
y_test = test_data[:,[-1]]

layers=[]
n_out = 2
layers.append(Dense(100,activation='lrelu'))
#layers.append(Dropout(0.8))
layers.append(Dense(100,activation='lrelu'))
#layers.append(Dropout(0.8))
#layers.append(Dense(50,activation='lrelu'))
layers.append(Dense(n_out))
layers.append(CovMat(kernel='rbf',alpha_fixed=False))

opt=Adam(1e-4)

gp=NNRegressor(layers,opt=opt,batch_size=50,maxiter=4000,gp=True,verbose=False)
gp.fit(x_train,y_train)

if len(which_qs) > 2 or n_out > 2 or True:
  ytr_pred,std=gp.predict(x_train)
  ytestpred,std = gp.predict(x_test)
  ydumb = np.mean(y_train)

  mse_train = np.mean((ytr_pred - y_train)**2)
  mse_test = np.mean((y_test - ytestpred)**2)
  mse_dumb = np.mean((y_test - ydumb)**2)
  print 'train',np.sqrt(mse_train)
  print 'test',np.sqrt(mse_test)
  print 'dumb',np.sqrt(mse_dumb)

if len(which_qs)==2 and n_out==2:
  get_p = lambda i,p: np.percentile(data[:,which_qs[i]], p)
  r1 = (get_p(0,5), get_p(0,95))
  r2 = (get_p(1,5), get_p(1,95))
  
  full1 = np.linspace(r1[0], r1[1], 1000)[:,np.newaxis]
  full2 = np.linspace(r2[0], r2[1], 1000)[:,np.newaxis]
 
  num_lines = 6 
  sp1 = np.linspace(r1[0], r1[1], num_lines)
  sp2 = np.linspace(r2[0], r2[1], num_lines)

  points = np.zeros((0,2))
  zs = np.zeros((0,2))
  for i in range(num_lines):
    verts1 = np.ones_like(full2) * sp1[i]
    points = np.concatenate((points, np.concatenate((verts1,full2),axis=1)),axis=0)
    horiz2 = np.ones_like(full1) * sp2[i]
    points = np.concatenate((points, np.concatenate((full1,horiz2),axis=1)),axis=0)

    zs = np.concatenate((zs, gp.fast_forward(points[-2*verts1.shape[0]:])),axis=0)

  alldatx = data[:,which_qs]
  alldaty = data[:,-1]
  alldatz = gp.fast_forward(alldatx)

  plt.scatter(points[:,0],points[:,1],lw=0,c=points[:,0] + points[:,1])
  plt.scatter(alldatx[:,0], alldatx[:,1], lw=0, c=alldaty)
  plt.xlim([-.3+r1[0], .3+r1[1]])
  plt.ylim([-.3+r2[0], .3+r2[1]])
  plt.show()     
  plt.close()
  plt.scatter(zs[:,0], zs[:,1], lw=0, c=points[:,0] + points[:,1])
  plt.scatter(alldatz[:,0], alldatz[:,1], lw=0, c=alldaty)
  plt.xlim([-.3+np.min(zs[:,0]),.3+np.max(zs[:,0])])
  plt.ylim([-.3+np.min(zs[:,1]),.3+np.max(zs[:,1])])
  plt.show()     


