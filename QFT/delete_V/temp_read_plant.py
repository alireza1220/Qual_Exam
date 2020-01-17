#from scipy import signal




#!/usr/bin/env python
#from scipy.io import loadmat
#x = loadmat('TFMAIN4.mat')

#print(x['TF'][0][0][0])
#print(type(x))
#file = tables.open_file('TFMAIN4.mat')

#import matlab.engine

#eng = matlab.engine.start_matlab()
#obj = eng.load('tf.mat')


import matlab.engine
eng = matlab.engine.start_matlab()


obj = eng.load('TFMAIN4.mat')
obj2 = obj['TF']

"""

tf = {}
for i in range(np): # number of plants 
  print(i)
  tf[i] = eng.tfreturn(i+1)



eng.quit()
"""

#obj2 = obj['tf']


#Names = eng.getfield(obj, 'tf')
#import matlab.engine
#eng = matlab.engine.start_matlab()
#eng.bode(nargout=0)
#eng.quit()


#obj = eng.load('TFMAIN4.mat')
#obj2 = obj['TF']
#Names = eng.getfield(obj, 'TF')
#Strs = eng.getfield(obj, 'structures')

"""
for o in obj2: 
    print(o)

print(o)
print(type(o))
Strs = eng.getfield(obj, 'TFMAIN4.mat')

ob = obj2['v40']

import matplotlib.pyplot as plt
#plt.plot(ob)

plt.show()



print(ob)
"""