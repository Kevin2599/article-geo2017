"""
Creata GPR data

*This notebook is to reproduce the results from the article. For more
information see the article in the parent directory.*

=> The figure is created in the notebook `gpr-figure.ipynb`.

Warning!
--------
**SciPy 0.19.0 has a memory leak in quadpack. Use another version (newer or
older)**

The problem is already fixed from the scipy-devs, and will be in the next
bugfix-release 0.19.1. Here I used the previous version 0.18.1.

"""

import os
import scipy
import subprocess
import numpy as np

from empymod.model import gpr, tem
from empymod.utils import printstartfinish


# Parameters
# Parameters as in Hunziker et al., 2015

x = np.r_[0.001, np.arange(1, 201)*.02]  # X-coord from 0-4 m, spacing of 20 cm
y = np.zeros(x.size)                     # Y-coord = 0
zsrc = 0.0000001                         # Source depth just slightly below 0
zrec = 0.5                               # Receiver depth at 0.5 m
depth = [0, 1]                           # 1 layer of 1 m thickness between two
                                         #          half-spaces
eperm = [1, 9, 15]                       # El. permit. model
res = [2e14, 200, 20]                    # Resistivity model: air, 1st layer,
                                         #                    2nd layer
f = np.arange(1, 850+1)*1e6              # Frequencies from 1e6 Hz to 850 Hz,
                                         #                  1e6 Hz sampling
t = np.arange(321)/4*1e-9                # Times from 1 ns to 80 ns,
                                         #            4 samples per ns
cf = 250e6                               # Center frequency
verb = 2                                 # Verbosity level

# Collect general input parameters
inp = {'src': [0, 0, zsrc], 'rec': [x, y, zrec], 'depth': depth, 'res': res,
       'cf': cf, 'ab': 11, 'gain': 3, 'epermH': eperm, 'epermV': eperm, 'loop':
       'off', 'verb': verb, 'freqtime': t, 'opt': 'spline', 'ft': 'fft',
       'ftarg': [f[0], f.size, 2048]}
       # FFT: we are padding with zerose to 2048 samples


# Calculate GPR with `empymod` for FHT, QWE, and QUAD and store it in
# `*.npy`-files which are loaded in the `gpr-figures.ipynb`.

# 1. FHT
gprFHT = gpr(ht='fht', htarg=['key_401_2009', 100], **inp)
np.save('data/GPR-FHT', gprFHT)


# 2. QWE
if scipy.__version__ == '0.19.0':
    print('SciPy 0.19.0 has a memory leak in QUAD, use another version!')
gprQWE = gpr(ht='qwe', htarg=[1e-8, 1e-15, '', 200, 200, 60, 1e-6, 160, 4000],
             **inp)
np.save('data/GPR-QWE', gprQWE)


# 3. QUAD
if scipy.__version__ == '0.19.0':
    print('SciPy 0.19.0 has a memory leak in QUAD, use another version!')
gprQUA = gpr(ht='quad', htarg=['', '', 51, '', 160, 500], **inp)
np.save('data/GPR-QUA', gprQUA)


# Calculate GPR with `EMmod`
# To calculate the `EMmod`-result, `EMmod` must be installed and in the
# bash-PATH.

# We use the empymod-utility to measure execution time; get start time
tstart = printstartfinish(verb)

# Change directory
os.chdir('data/GPR')

# Run EMmod
subprocess.run('bash gprloop_twointerface.scr', shell=True,
               stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

# Change back to original directory
os.chdir('../..')

# Read data
fEM = np.zeros((f.size, x.size), dtype=complex)
for i in range(f.size):
    filename = 'data/GPR/gprloop_twointmod_freq'+str(i+1)+'_11.bin'
    tf = open(filename, 'rb')
    temp = np.fromfile(tf)
    fEM[i, :] = temp[x.size*2:x.size*4:2] + 1j*temp[x.size*2+1:x.size*4+1:2]

# Multiply with ricker wavelet
cfc = -(np.r_[0, f[:-1]]/250e6)**2
fwave = cfc*np.exp(cfc)
fEM *= fwave[:, None]

# Do f->t transform
tEM, conv = tem(fEM, x, f, t, 0, 'fft', [f[0], f.size, 2048, None])

# Apply gain; make pure real
tEM *= (1 + np.abs((t*10**9)**3))[:, None]
gprEMmod = tEM.real

# Print execution time (it will show 'empymod', but obviously in this case it
# is `EMmod`)
printstartfinish(verb, tstart)

# Store EMmod
np.save('data/GPR-EMmod', gprEMmod)


# Calculate theoretical arrival times

# Arrival times for direct wave
clight = 299792458
vel = clight/np.sqrt(eperm[1])
arrtime = np.sqrt((zsrc-zrec)**2 + x**2 + y**2)/vel

# Arrival times for reflected wave
arrtimeref = np.sqrt((np.abs(zsrc - depth[1]) + np.abs(zrec - depth[1]))**2 +
                     x**2 + y**2)/vel

# Arrival times for refracted wave in the air
# This only works if ypos = 0
refractang = np.arcsin(vel/clight)
arrtimerefair = (np.abs(zsrc - depth[0])/np.cos(refractang) +
                 np.abs(zrec - depth[0])/np.cos(refractang))/vel
arrtimerefair += (np.abs(x) - np.abs(zsrc - depth[0])*np.tan(refractang) -
                  np.abs(zrec - depth[0])*np.tan(refractang))/clight

np.savez('data/ArrivalTimes.npz', arrtime=arrtime, arrtimeref=arrtimeref,
         arrtimerefair=arrtimerefair, x=x)
