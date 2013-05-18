from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.fftpack as fftpack
import helpers
import rm_filt

# global parameters
fs     = 48000 # sampling rate
nchn   = 2     # number of channels
my_eps = np.finfo(np.float).eps

im_sig = np.array([[1]+[0 for i in range(fs-1)]]*nchn)

#
# Test LTISys helper class
#

# parameters
order  = 5      # filter order
f_edge = 3e3    # edge frequency

b, a = sig.butter(order, f_edge*2/fs)

ltisys = rm_filt.LTISys(b, a)

print "Filter coeffs:"
print "b =", b
print "a =", a

print ltisys.filter(im_sig)

#
# Test helper functions
#

A1, A2 = helpers.any_to_ap_pair(b,a)
H1, H2 = helpers.get_power_complementary_filters(A1, A2)

tf1 = [sig.freqz(*np.hsplit(h, 2))[1] for h in (H1, H2)]

print "All-Pass filter coeffs:"
print "A_1 =\n", A1
print "A_2 =\n", A2

print "Recalculated filter coeffs:"
print "b_1 =\n", H1[:,0]
print "a_1 =\n", H1[:,1]
print "b_2 =\n", H2[:,0]
print "a_2 =\n", H2[:,1]

fig1 = plt.figure()
ax = fig1.add_subplot(111)

ax.plot(np.abs(tf1[0]), label='H1')
ax.plot(np.abs(tf1[1]), label='H2')
ax.legend()

#
# Test RMFilterBank class
#

# parameters
nbands = 4      # number of bands
order  = 7      # filter order
fe_max = 10e3   # maximum edge frequency

rm_fb = rm_filt.RMFilterBank(fe_max, fs, order=order, nbands=nbands, nchn=nchn)

bs_sig   = rm_fb.analyze(im_sig)
out_sig  = rm_fb.synthesize(bs_sig)

bs_spec  = fftpack.fft(bs_sig)[...,:fs/2+1]
out_spec = fftpack.fft(out_sig)[0,:fs/2+1]

tf2 = [(sig.freqz(*np.hsplit(h[0], 2), worN=fs/2)[1],
        sig.freqz(*np.hsplit(h[1], 2), worN=fs/2)[1],)
        for h in rm_fb.H]

fig2 = plt.figure()

ax = fig2.add_subplot(311)
ax.set_title('Frequency response of the LP and HP filters.')
for i, t in enumerate(tf2):
    ax.plot(20*np.log10(np.abs(t[1])+my_eps),
            label="Low-Pass, $f_c=%.0f$ Hz" % rm_fb.edge_freqs[i])
    ax.plot(20*np.log10(np.abs(t[0])+my_eps),
            label="High-Pass $f_c=%.0f$ Hz" % rm_fb.edge_freqs[i])
    ax.set_ylim([-100,0])

ax = fig2.add_subplot(312)
ax.set_title('Frequency response of the bands.')
for i, os in enumerate(bs_spec):
    ax.plot(20*np.log10(np.abs(os)+my_eps).T, label="Band %i" % (i+1))
    ax.set_ylim([-100,0])

ax = fig2.add_subplot(313)
ax.set_title('Demonstration of the double-complementary property')
ax.plot(20*np.log10(np.abs(out_spec.T) + my_eps),
        label="Spectrum of the synthesis output")
ax.plot(20*np.log10(np.sum(np.vstack([np.abs(o)**2 for o in bs_spec]), axis=0)/nchn + my_eps),
        label="$\sum_i \left|H_i(z)\\right|^2$")
ax.plot(20*np.log10(np.abs(np.vstack(bs_spec).sum(axis=0)/nchn) + my_eps),
        label="$\left|\sum_i H_i(z)\\right|$")

for ax in fig2.axes:
    ax.legend()
    ax.set_ylabel('Magnitude in dB FS')
    ax.set_xlabel('Frequency f in Hz')
    ax.set_xlim((0, fs/2+1))

plt.show()
