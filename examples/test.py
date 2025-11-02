from pyevolve import Util
from PIL import Image
import numpy as np
import random
import pywt
import math

IMG_WIDTH=608
IMG_HEIGHT=300

w=Image.new("L", (4,4), "white")
pix=w.load()

#for i in range(0,4):
#    for j in range(0,4):
#        pix[i,j] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

b=Image.new("L", (4,4), "black")
b=np.array(b)
orig = Image.open("./data/est.png")
target = np.array(orig)
input = np.array(Image.open("./data/target.jpg"))
tfft = np.fft.fft2(target)
rmse_accum = Util.VectorErrorAccumulator()



efft = np.fft.fft2(input)    
#print abs(efft/len(efft))
#print abs(tfft/len(tfft))
rmse_accum.append(abs(tfft/len(tfft)), abs(efft/len(efft)))
print rmse_accum.getMean()/255

rmse_accum.reset()
rmse_accum.append(target, input)
print rmse_accum.getRMSE()
score = 0
for i in range(0,3):
    ev = input[:,:,i]   
    hist = np.histogram2d(ev.ravel(), target[:,:,i].ravel(), [x for x in xrange(0, 256)])[0]
    nonzeroInd = np.nonzero(hist)        
    nonzero = hist[nonzeroInd]        
    histProb = nonzero/float(608*300)     
    score += -np.sum(np.log2(histProb)*histProb)
max_entropy = 3*math.log(IMG_WIDTH*IMG_HEIGHT, 2)    
print score/max_entropy
    
#cA, (cH, cV, cD) = pywt.dwt2(target, 'haar')
#print cA
#print cD
#cA, (cH, cV, cD) = pywt.dwt2(b, 'haar')
#print cA
#print cD
#print pywt.dwt2(b, 'haar')        
            