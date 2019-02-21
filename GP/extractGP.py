
import OnlineGP
import scipy.io as sio

def getGP(fname):
    try:
        raw_data = sio.loadmat(fname)
    except:
        return 'Bad file.'
    data = raw_data['data']
    
    print 'data = ',data
    
    C = data['C'][0][0]
    alpha = data['alpha'][0][0]
    KB = data['KB'][0][0]
    KBinv = data['KBinv'][0][0]
    w = data['weighted'][0][0]
    cov = data['covar_params'][0][0][0]
    cov = (cov[0],cov[1])
    BV = data['BV'][0][0]
    noise = data['noise_var'][0][0]

    size = BV.shape
    gp = OnlineGP.OGP(size[1],(cov[0], cov[1], noise),maxBV=size[0],weighted=w)
    gp.noise_var = noise
    gp.alpha = alpha
    gp.C = C
    gp.KB = KB
    gp.KBinv = KBinv
    gp.BV = BV

    return gp
