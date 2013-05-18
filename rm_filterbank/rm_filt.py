import numpy as np
import scipy.signal as sig
import helpers

class LTISys(object):
    """An LTI filter class.

    This is simply a class that implements an LTI system by wrapping
    scipy.signal.lfilter().
    """

    def __init__(self, b, a, nchn=1):
        """The constructor.

        Inputs:
        -------

        b, a:   Coefficients of the filter design.
        nchn:   The number of input channels to be supported.
        """

        a = a.flatten()
        b = b.flatten()

        self.__a = a
        self.__b = b

        Ma, Mb = a.size, b.size

        self.__nchn = nchn

        self.__order = max(Mb, Ma)-1
        self.__states = np.zeros((self.__nchn,self.__order))

    def filter(self, in_sig, axis=-1):
        """Filter a signal.

        This method filters an N-dimensional signal.  See scipy.signal.lfilter
        for more details.

        Inputs:
        -------

        in_sig: The input signal; it must have self.n_chn channels.
        axis:   The axis along which the filter operates (default: -1).

        """

        # workaround a SEGFAULT in scipy.signal.lfilter() when filtering
        # two-dimensional signals
        in_sig = np.atleast_2d(in_sig)
        out_sig = np.zeros(in_sig.shape)
        for i in range(self.__nchn):
            out_sig[i,:], self.__states[i,:] = sig.lfilter(
                b    = self.__b,
                a    = self.__a,
                x    = in_sig[i,:],
                axis = axis,
                zi   = self.__states[i,:]
            )

        return out_sig

    b = property(fget=lambda self: self.__b)
    a = property(fget=lambda self: self.__a)
    n_chn = property(fget=lambda self: self.__nchn)

class RMFilterBank(object):
    """A Class that implements a Regalia-Mitra filter bank.

    This class provides two methods: analyze() and synthesize(), that implement
    the band splitting and its inverse, respectively.  This allows one to split
    a signal into N bands, process these bands separately, and then recombine
    them.
    """

    def __init__(self,
                 max_edge_freq,
                 fs,
                 order,
                 nbands=2,
                 nchn=1,
                 w_co=[],
                 filter_type='ellip'):
        """The Constructor.

        The constructor
        """

        # override nbands if w_co is passed
        if w_co:
            nbands = len(w_co)+1

        self.__nbands = nbands

        self.__AP     = []
        self.__H      = []
        self.__edge_freqs = None
        self.__gen_filter_bank(max_edge_freq, fs, order, nbands, filter_type, w_co)

        # construct the analysis filter tree
        self.__ana_filters = []
        for i in xrange(nbands-1):
            self.__ana_filters.append([])
            for j in xrange(i):
                self.__ana_filters[i].append(LTISys(*np.hsplit(self.__AP[i][0], 2), nchn=nchn))
            for h in self.__AP[i][::-1]:
                self.__ana_filters[i].append(LTISys(*np.hsplit(h, 2), nchn=nchn))

        # construct the synthesis filter tree
        self.__syn_filters = []
        for i in xrange(nbands-1):
            self.__syn_filters.append([])
            for j in xrange(nbands-2-i):
                self.__syn_filters[i].append(LTISys(*np.hsplit(self.__AP[nbands-2-i][1], 2), nchn=nchn))
            for h in self.__AP[nbands-2-i]:
                self.__syn_filters[i].append(LTISys(*np.hsplit(h, 2), nchn=nchn))

    def __gen_filter_bank(self, max_edge_freq, fs, order, nbands, filter_type, w_co=[]):
        """Function to generate AP filters for constructing a Regalia-Mitra filter bank
        with equidistant edge frequencies.  First, low-pass elliptic filters are
        designed.  Secondly, the AP filters from which the LP and it's
        double-complementary HP filter can be derived via butterfly operation are
        calculated.  Thirdly, the double-complementary high-pass filter is calculated.

        Usage:
          [AP, H] = gen_filter_bank(nbands, max_edge_freq, order, fs)

        Input arguments (all required):
          nbands:         Number of bands of the filter bank.
          max_edge_freq:  The highest edge frequency (in Hz).  All edge frequencies
                          are lower or equal to this frequency.
          order:          The order the low-pass filters used to create the all-pass
                          filters are supposed to have.
          fs:             The sampling frequency (in Hz).

        Output arguments:
          AP:     Cell array containing the coefficients of complementary all-pass
                  filters whose sum yields a low-pass and whose difference yields the
                  double-complementary high-pass filter.  The b and a coefficients are
                  in the first and second column, respectively.
          H:      Cell array containing the coefficients of the double complementary
                  low-pass and high-pass filters described above.
        """

        if not w_co:
            # split edge frequencies evenly (though arbitrarily limit edge frequencies to
            # user configurable value)
            w_co = np.linspace(max_edge_freq/nbands, max_edge_freq, nbands)[:-1]/(fs/2.)
        else:
            w_co = np.array(w_co)
            w_co = w_co/(fs/2.)

        self.__edge_freqs = w_co*fs/2

        b = np.zeros((nbands,order+1))
        a = np.zeros((nbands,order+1))

        for i in range(nbands-1):
            # original filter
            #
            # If the pass-band is maximally flat, one could perhaps optimise for
            # performance by approximating the filter by using only lowest necessary LP
            # filter instead of going through all previous filters.

            if filter_type == 'butter':
                b_d, a_d = sig.butter(order, w_co[-1-i], 'low')
            elif filter_type == 'ellip':
                b_d, a_d = sig.ellip(order, 0.0001, 50, w_co[-1-i])
            else:
                raise ValueError('Bad filter type!')

            b[i] = b_d/a_d[0]
            a[i] = a_d/a_d[0]

            A1, A2 = helpers.any_to_ap_pair(b[i],a[i])
            H1, H2 = helpers.get_power_complementary_filters(A1,A2)

            self.__AP.append((A1, A2))
            self.__H.append((H1, H2))

    def analyze(self, in_sig):

        in_sig = np.atleast_2d(in_sig)

        nchn, nsmp = in_sig.shape

        out_sig    = np.zeros((self.__nbands, nchn, nsmp))
        out_sig[0] = in_sig.copy()

        for i, s in enumerate(self.__ana_filters):

            out_sig[i+1] = s[-1].filter(out_sig[i])
            for j, h in enumerate(s[:-1]):
                out_sig[j] = h.filter(out_sig[j])

            # butterfly operation
            out_sig[i:i+2] = np.array((0.5*(out_sig[i+1] - out_sig[i]),
                                       0.5*(out_sig[i+1] + out_sig[i])))

        return out_sig

    def synthesize(self, bs_sig):

        out_sig = bs_sig.copy()

        for i, s in enumerate(self.__syn_filters):
            out_sig[-2:] = np.array((0.5*(out_sig[-1] - out_sig[-2]),
                                     0.5*(out_sig[-1] + out_sig[-2])))
            for j, h in enumerate(s):
                out_sig[j] = h.filter(out_sig[j])

            out_sig[-2] = out_sig[-2:].sum(axis=0)
            out_sig = out_sig[:-1]

        return out_sig[0]

    AP = property(fget=lambda self: self.__AP)
    H  = property(fget=lambda self: self.__H)
    edge_freqs  = property(fget=lambda self: self.__edge_freqs)
