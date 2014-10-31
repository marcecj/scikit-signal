import argparse

from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.fftpack as fftpack

from . import regalia_mitra

SIG_LEN = 48000

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--fs",
                    dest="fs",
                    default=SIG_LEN,
                    type=int,
                    help="The sampling frequency")
parser.add_argument("-c", "--nchn",
                    dest="nchn",
                    default=2,
                    type=int,
                    help="The number of channels")
parser.add_argument("-b", "--nbands",
                    dest="nbands",
                    default=4,
                    type=int,
                    help="The number of channels")
parser.add_argument("-o", "--filter-order",
                    dest="order",
                    default=7,
                    type=int,
                    help="The number of channels")
parser.add_argument("-m", "--max-edge-freq",
                    dest="fe_max",
                    default=10e3,
                    type=float,
                    help="The maximum edge frequency")
parser.add_argument("-f", "--freqs",
                    dest="freqs",
                    default=[],
                    type=lambda x: [int(f) for f in x.split(",")],
                    help="A list of edge frequencies (overrides --nbands)")
args = parser.parse_args()

# global parameters
fs = args.fs
nchn = args.nchn
my_eps = np.finfo(np.float).eps

# RMFilterBank parameters
nbands = args.nbands
order = args.order
w_co = args.freqs
fe_max = args.fe_max

im_sig = np.array([[1]+[0 for i in range(SIG_LEN-1)]]*nchn)

#
# Test LTISys helper class
#

# parameters
f_order = 5     # filter order
f_edge = 3e3    # edge frequency

b, a = sig.butter(f_order, f_edge*2/SIG_LEN)

ltisys = regalia_mitra.LTISys(b, a, nchn=2)

print("Filter coeffs:")
print("b =", b)
print("a =", a)

print(ltisys.filter(im_sig))

#
# Test helper functions
#

A1, A2 = regalia_mitra.any_to_ap_pair(b, a)
H1, H2 = regalia_mitra.get_power_complementary_filters(A1, A2)

tf1 = [sig.freqz(*np.hsplit(h, 2))[1] for h in (H1, H2)]

print("All-Pass filter coeffs:")
print("A_1 =\n", A1)
print("A_2 =\n", A2)

print("Recalculated filter coeffs:")
print("b_1 =\n", H1[:, 0])
print("a_1 =\n", H1[:, 1])
print("b_2 =\n", H2[:, 0])
print("a_2 =\n", H2[:, 1])

fig1, ax = plt.subplots(tight_layout=True)
ax.autoscale(tight=True)
ax.plot(np.abs(tf1[0]), label='H1')
ax.plot(np.abs(tf1[1]), label='H2')
ax.legend()

#
# Test RMFilterBank class
#

rm_fb = regalia_mitra.RMFilterBank(
    fs, fe_max, nbands=nbands, nchn=nchn, w_co=w_co,
    lowpass_design_func=lambda w: sig.ellip(order, 1e-5, 50, w)
)

bs_sig = rm_fb.analyze(im_sig)
out_sig = rm_fb.synthesize(bs_sig)

bs_spec = fftpack.fft(bs_sig)[..., :SIG_LEN/2+1]
out_spec = fftpack.fft(out_sig)[0, :SIG_LEN/2+1]

tf2 = [(sig.freqz(*np.hsplit(h[0], 2), worN=SIG_LEN//2)[1],
        sig.freqz(*np.hsplit(h[1], 2), worN=SIG_LEN//2)[1],)
       for h in rm_fb.H]

fig2, ax = plt.subplots(3, 1, sharex=True, tight_layout={'pad': 0.15})

ax[0].set_title('Frequency response of the LP and HP filters.')
for i, t in enumerate(tf2):
    ax[0].plot(20*np.log10(np.abs(t[0])+my_eps),
               label="Low-Pass, $f_c=%.3f$ Hz" % rm_fb.edge_freqs[i])
    ax[0].plot(20*np.log10(np.abs(t[1])+my_eps),
               label="High-Pass, $f_c=%.3f$ Hz" % rm_fb.edge_freqs[i])
    ax[0].set_ylim([-100, 0])

ax[1].set_title('Frequency response of the bands.')
for i, os in enumerate(bs_spec):
    ax[1].plot(20*np.log10(np.abs(os)+my_eps).T, label="Band %i" % (i+1))
    ax[1].set_ylim([-100, 0])

ax[2].set_title('Demonstration of the double-complementary property')
ax[2].plot(20*np.log10(np.abs(out_spec.T) + my_eps),
           label="Spectrum of the synthesis output")
ax[2].plot(20*np.log10(np.sum(np.vstack([np.abs(o)**2 for o in bs_spec]),
                              axis=0)/nchn + my_eps),
           label="$\sum_i \left|H_i(z)\\right|^2$")
ax[2].plot(20*np.log10(np.abs(np.vstack(bs_spec).sum(axis=0)/nchn) + my_eps),
           label="$\left|\sum_i H_i(z)\\right|$")

ax[-1].set_xlabel('Frequency f in Hz')
for ax in fig2.axes:
    ax.legend()
    ax.set_ylabel('Magnitude in dB FS')
    ax.set_xlim((0, SIG_LEN/2+1))

plt.show()
