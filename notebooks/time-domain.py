"""
Time-domain impulse response

A comparison between the analytical solution and `empymod` for a time-domain
impulse responses for inline, x-directed source and receivers, for the four
different frequency-to-time methods **QWE**, **FHT**, **FFTLog**, and **FFT**.
"""
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from empymod import dipole

# Style adjustments
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['text.usetex'] = True
rcParams['font.serif'] = 'Computer Modern Roman'
rcParams['font.family'] = 'serif'
rcParams['font.style'] = 'normal'

def ee_xx_impulse(res, off, time):
    """Electric halfspace impulse response to an electric source, xx, inline.

    Wilson, A. J. S., 1997, equation 5.38.
    The equivalent wavefield concept in multichannel transient electromagnetic
    surveying.
    Ph.D., University Of Edinburgh.
    http://hdl.handle.net/1842/7101

    res   : resistivity [Ohm.m]
    off   : offset [m]
    time  : time(s) [s]
    """
    mu_0 = 4e-7*np.pi  # Permeability of free space  [H/m]
    fact = np.sqrt(mu_0**3/(t**5*res*np.pi**3))/8
    return fact*np.exp(-mu_0*off**2/(4*res*time))

# Example 1: Source and receiver at z=0m

# Comparison with analytical solution; put 1 mm below the interface, as they
# would be regarded as in the air by `emmod` otherwise.
src = [0, 0, 0.001]          # Source at origin, slightly below interface
rec = [6000, 0, 0.001]       # Receivers in-line, 0.5m below interface
res = [1e23, 10]             # Resistivity: [air, half-space]
signal = 0                   # Impulse response
t = np.logspace(-2, 2, 101)  # Desired times (s)
inparg = {'src': src, 'rec': rec, 'depth': 0, 'freqtime': t, 'res': res,
          'signal': signal, 'ht': 'fht', 'verb': 1}

# Impulse response
ex = ee_xx_impulse(res[1], rec[0], t)

# Calculation
inparg['signal'] = 0 # signal 0 = impulse
qwe = dipole(**inparg, ft='qwe')
sin = dipole(**inparg, ft='sin', ftarg='key_81_CosSin_2009')
ftl = dipole(**inparg, ft='fftlog')
fft = dipole(**inparg, ft='fft', ftarg=[.00005, 2**20, '', 10])

# Figure
fig, axs = plt.subplots(figsize=(9, 3.25), facecolor = 'w', nrows=1,
                        ncols=2, sharey=True)
fig.subplots_adjust(wspace=.05)
plt.suptitle('Impulse response for a half-space of 10$\,\Omega$\,m at ' +
             '6\,km offset.', y=1.05, fontsize=14)

# Amplitude
plt.sca(axs[0])
plt.title(r'Amplitude')
plt.xlabel('Time (s)')
plt.ylabel(r'Amplitude (V/(s\,m))')
plt.loglog(t, ex, 'k.', label='Analytical')
plt.loglog(t, np.abs(fft), '0.0', ls=':', lw=1)
plt.loglog(t, np.abs(qwe), '0.0', ls='-', lw=1)
plt.loglog(t, np.abs(sin), '0.6', ls=':', lw=1)
plt.loglog(t, np.abs(ftl), '0.6', ls='-', lw=1)
plt.ylim([1e-21, 1e-10])
plt.xlim([t.min(), t.max()-1])  # Only to 99 to hide 10^2-label

# Error
plt.sca(axs[1])
plt.title('Error')
plt.xlabel('Time (s)')
plt.ylabel('Absolute error (V/(s\,m))')
plt.loglog(t, abs(fft-ex), '0.0', ls=':', lw=1, label='562\,ms: FFT')
plt.loglog(t, abs(qwe-ex), '0.0', ls='-', lw=1, label='298\,ms: QWE')
plt.loglog(1e-10, 1e-15, 'k.', label='Analytical')  # Dummy for legend
plt.loglog(t, abs(sin-ex), '0.6', ls=':', lw=1, label='11\,ms: Sine Filter')
plt.loglog(t, abs(ftl-ex), '0.6', ls='-', lw=1, label='6\,ms: FFTLog')
plt.xlim([t.min(), t.max()-1])  # Only to 99 to hide 10^2-label
axs[1].yaxis.set_ticks_position('right')
axs[1].yaxis.set_label_position('right')

# Plot legend
plt.legend(ncol=2, framealpha=1, bbox_to_anchor=(.65, 0.35),
           bbox_transform=plt.gcf().transFigure)

# Save and show plot
plt.savefig('../figures/impulse2.jpg', bbox_inches='tight')
plt.show()
