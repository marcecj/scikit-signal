"""This module implements a set of functions and a class for generating
Regalia-Mitra filter banks.

Regalia-Mitra filter banks are a form of tree-structured filter bank that
consist of all-pass filters.  These all-pass filters are derived from a set of
low-pass filter designs which must fulfill certain properties [1]_.  They have
the special property of being doubly-complementary, i.e., simultaneously
all-pass- and power-complementary.  For an N-band filter bank, this means that
the following two equations hold for its output filters::

    |sum[k=0..N]  H_k(z)|   = 1
     sum[k=0..N] |H_k(z)|^2 = 1

Regalia-Mitra filter banks consist of two stages: an analysis and a synthesis
filter bank.  However, due to the all-pass complementary property, it is
possible to simply sum up the output of the analysis stage.  The output will
merely be all-pass filtered.  However, by using the synthesis filter bank, the
overall transfer function of the system remains invariant to parameter changes
(according to [0]_).

This implementation uses the collapsed-tree form (see figures 9 and 10a in
[0]_).

References
----------

.. [0] P. A. Regalia, P. P. Vaidyanathan, M. Renfors, Y. Neuvo, and S. K.
   Mitra, 'Tree-structured complementary filter banks using all-pass sections',
   IEEE Trans. Circuits and Systems, vol. 34, no. 12, pp. 1470-1484, December
   1987.  (downloadable at http://faculty.cua.edu/regalia/)

.. [1] Vaidyanathan, P.P.; Mitra, S.K.; Neuvo, Y., 'A new approach to the
   realization of low-sensitivity IIR digital filters', Acoustics, Speech and
   Signal Processing, IEEE Transactions on , vol.34, no.2, pp.350,361, Apr 1986
"""

from inspect import isfunction

import numpy as np
import scipy.signal as sig


def get_power_complementary_q(P, D):
    """Calculates the non-recursive coefficients Q of a doubly-complementary
    filter Q/D to a filter P/D, with P it's non-recursive coefficients and D
    it's recursive coefficients.  The term "doubly-complementary" means a group
    of filters that is simultaneously all-pass-complementary and
    power-complementary, i.e.::

        |P(z)/D(z)    +  Q(z)/D(z)|   = 1
        |P(z)/D(z)|^2 + |Q(z)/D(z)|^2 = 1.

    Parameters
    ----------
    P : numpy.ndarray
        The non-recursive filter coefficients.
    D : numpy.ndarray
        The recursive filter coefficients.

    Returns
    -------
    Q : numpy.ndarray
        The non-recursive coefficients of the doubly-complementary filter.
    """

    # make sure we get column vectors
    P = np.atleast_2d(P.flatten()).T
    D = np.atleast_2d(D.flatten()).T

    # get doubly complementary filter
    # R(z) = P(z)^2 - z^(-n)*D(z^-1)*D(z)
    P_sq = P.dot(P.T)
    D_sq = D[::-1].dot(D.T)

    # sort P and D by power of z, since the distribution of the powers of z of
    # the squared polynomial in matrix form is:
    #
    # P(z)'*P(z) = 1      z      ...    z^n
    #              z      z^2    ...    z^n+1
    #              ...    ...    ...    ...
    #              z^n    z^n+1  ...    z^2n
    #
    # and similarly for z^-n*D(z^-1)*D (Matlab equivalent: D(end:-1:1)'*D).

    N = D.size
    P_sq_sort = np.zeros((N*2-1,))
    D_sq_sort = np.zeros((N*2-1,))
    for m in range(N):
        for n in range(N):
            P_sq_sort[m+n] += P_sq[m, n]
            D_sq_sort[m+n] += D_sq[m, n]

    Q = np.zeros((P.size,))

    # recursively calculate Q from R
    R = P_sq_sort - D_sq_sort
    Q[0] = np.sqrt(R[0])
    Q[1] = R[1]/(2*Q[0])
    # Q is antisymmetric, so calculate N/2 and generate the rest from that
    for k in range(2, N//2):
        Q[k] = (R[k] - (Q[1:k].T.dot(Q[k-1:0:-1])))/(2*Q[0])

    Q[N/2:] = -Q[N/2-1::-1]

    return Q


def get_power_complementary_filters(A1, A2):
    """Convert a pair of complementary all-pass filters to a
    doubly-complementary low-pass/high-pass pair.  The term
    "doubly-complementary" means a group of filters that is simultaneously
    all-pass-complementary and power-complementary, i.e.::

        |H1(z)    +  H2(z)|   = 1
        |H1(z)|^2 + |H2(z)|^2 = 1,

    where H1 is the original filter from which A1 and A2 were calculated.

    Parameters
    ----------
    A1, A2 : numpy.ndarray
        A pair of complementary all-pass filter coefficients (the b and a
        coefficients must be in the first and second column, respectively).

    Returns
    -------
    H1, H2 : numpy.ndarray
        Arrays containing the coefficients of doubly-complementary high-pass
        and low-pass filters.  The b and a coefficients are in the first and
        second column, respectively.
    """

    # extract b and a coefficients
    A1b, A1a = [a.T for a in np.hsplit(A1, 2)]
    A2b, A2a = [a.T for a in np.hsplit(A2, 2)]

    # calculate and sort coefficients of the doubly-complementary filters

    H1b = 0.5*(A1b.T.dot(A2a) + A1a.T.dot(A2b))
    H1a = A1a.T*A2a
    H2b = 0.5*(A1b.T.dot(A2a) - A1a.T.dot(A2b))

    M, N = H1b.shape
    H1b_sort = np.zeros((M+N, 1))
    H2b_sort = np.zeros((M+N, 1))
    H1a_sort = np.zeros((M+N, 1))
    H2a_sort = np.zeros((M+N, 1))

    for m in range(M):
        for n in range(N):
            H1b_sort[m+n] += H1b[m, n]
            H2b_sort[m+n] += H2b[m, n]
            H1a_sort[m+n] += H1a[m, n]
            H2a_sort[m+n] += H1a[m, n]

    H1 = np.hstack((H1b_sort, H1a_sort))
    H2 = np.hstack((H2b_sort, H2a_sort))

    return (H1, H2)


def any_to_ap_pair(b, a):
    """Converts any filter that satisfies the constraints in [1]_ to a pair of
    doubly-complementary all-pass filters.

    Parameters
    ----------
    b : numpy.ndarray
        The non-recursive filter coefficients.
    a : numpy.ndarray
        The recursive filter coefficients.

    Returns
    -------
    A1, A2 : numpy.ndarray
        Arrays containing the coefficients of complementary all-pass filters
        whose sum yields a low-pass and whose difference yields the
        doubly-complementary high-pass filter.  The b and a coefficients are in
        the first and second column, respectively.
    """

    # make sure we get 1d vectors
    b = b.flatten()
    a = a.flatten()

    P = b/a[0]  # numerator polynomial
    D = a/a[0]  # denominator polynomial

    # get P's doubly-complementary polynomial Q
    Q = get_power_complementary_q(P, D)

    z1 = np.roots(P+Q)
    # z2 = np.roots(P-Q)

    # calculate the all-pass functions
    A2a = np.poly(z1[np.abs(z1) < 1])
    A2b = A2a[::-1]
    A1b = np.poly(z1[np.abs(z1) >= 1])
    A1a = A1b[::-1]

    # normalize a0 to 1
    A2b, A2a = A2b/A2a[0], A2a/A2a[0]
    A1b, A1a = A1b/A1a[0], A1a/A1a[0]

    A1 = np.vstack((A1b, A1a)).T
    A2 = np.vstack((A2b, A2a)).T

    return A1, A2


class LTISys(object):
    """An LTI filter class.

    This is a simple class that implements an LTI system by wrapping
    `scipy.signal.lfilter`.  Its primary purpose is to take care of the filter
    state.
    """

    def __init__(self, b, a, nchn=1):
        """Initialise an LTISys object.

        Parameters
        ----------
        b : numpy.ndarray
            The non-recursive filter coefficients.
        a : numpy.ndarray
            The recursive filter coefficients.
        nchn : int, optional
            The number of input channels to be supported (default: 1).
        """

        a = a.flatten()
        b = b.flatten()

        self.__a = a
        self.__b = b

        self.__nchn = nchn

        order = max(a.size, b.size)-1
        self.__states = np.zeros((self.__nchn, order))

    def filter(self, x, axis=-1):
        """Filter an N-dimensional signal.  See `scipy.signal.lfilter` for more
        details.

        Parameters
        ----------
        x : numpy.ndarray
            The input signal; it must have `self.nchn` channels.
        axis : int, optional
            The axis along which the filter operates (default: -1).

        """

        # Swap the axes here instead of passing the axis argument to
        # scipy.signal.lfilter because in the latter case the shape of the
        # states would not match the shape of the input signal, leading to an
        # error in lfilter.
        x = np.atleast_2d(x).swapaxes(axis, -1)

        if x.shape[0] != self.__nchn:
            raise ValueError('x has the wrong number of channels.')

        # TODO: file a bug report: when zi has the wrong dimensionality,
        # lfilter causes SIGSEG's, SIGABRT's, etc.
        y, self.__states = sig.lfilter(
            b=self.__b,
            a=self.__a,
            x=x,
            zi=self.__states
        )

        return y.swapaxes(axis, -1)

    @property
    def b(self):
        "The non-recursive filter coefficients."
        return self.__b

    @property
    def a(self):
        "The recursive filter coefficients."
        return self.__a

    @property
    def nchn(self):
        "The number of channels of the filter."
        return self.__nchn


class RMFilterBank(object):
    """A class that implements a Regalia-Mitra filter bank.

    This class provides two methods: `analyze()` implements the band splitting,
    and `synthesize()` its inverse.  This allows one to split a signal into N
    bands, process these bands separately, and then recombine them.
    """

    def __init__(self,
                 fs=1.0,
                 max_edge_freq=None,
                 numbands=2,
                 w_co=None,
                 nchn=1,
                 lowpass_design_func=None):
        """Initialise an RMFilterBank object.

        Parameters
        ----------
        fs : float, optional
            The sampling rate in Hz (default: 1.0).
        max_edge_freq : float, optional
            The highest edge frequency of the filter bank (in Hz), that is, the
            edge frequency of the final high-pass.  If None (the default), and
            if `w_co` is None, then it is set to ``fs/2``.
        numbands : int, optional
            The number of frequency bands of the filter bank (default: 2).
        w_co : list-like, optional
            A list of edge frequencies (in Hz).  This overrides `numbands` if
            given.
        nchn : int, optional
            The number of channels the filter bank should support.
        lowpass_design_func : function, optional
            A function that designs a low-pass filter.  It must implement the
            following API::

                b, a = lowpass_design_func(w_e)

            Its sole argument is the edge frequency normalised to ``fs=2``, and
            its return values are the b and a coefficients of the filter.  The
            default is to design an elliptic filter with ``N=7``, ``rp=1e-5``
            and ``rs=50``.
        """

        # override numbands if w_co is passed
        if w_co:
            numbands = len(w_co)+1

        if not max_edge_freq:
            max_edge_freq = fs/2

        self.__numbands = numbands

        self.__AP = []
        self.__H = []
        self.__edge_freqs = None
        self.__gen_filter_bank(fs, max_edge_freq, w_co, numbands,
                               lowpass_design_func)

        # construct the analysis filter tree
        self.__ana_filters = []
        for i in range(numbands-1):
            self.__ana_filters.append([])
            for j in range(i):
                self.__ana_filters[i].append(
                    LTISys(*np.hsplit(self.__AP[i][0], 2), nchn=nchn)
                )
            for h in self.__AP[i][::-1]:
                self.__ana_filters[i].append(
                    LTISys(*np.hsplit(h, 2), nchn=nchn)
                )

        # construct the synthesis filter tree
        self.__syn_filters = []
        for i in range(numbands-1):
            self.__syn_filters.append([])
            for j in range(numbands-2-i):
                self.__syn_filters[i].append(
                    LTISys(*np.hsplit(self.__AP[numbands-2-i][1], 2),
                           nchn=nchn)
                )
            for h in self.__AP[numbands-2-i]:
                self.__syn_filters[i].append(
                    LTISys(*np.hsplit(h, 2), nchn=nchn)
                )

    def __gen_filter_bank(self, fs, max_edge_freq, w_co, numbands,
                          lowpass_design_func):
        """Function to generate AP filters for constructing a Regalia-Mitra
        filter bank with equidistant edge frequencies.  First, low-pass
        elliptic filters are designed.  Secondly, the AP filters from which the
        LP and it's doubly-complementary HP filter can be derived via butterfly
        operation are calculated.  Thirdly, the doubly-complementary high-pass
        filter is calculated.

        Parameters
        ----------
        fs : float
            The sampling frequency (in Hz).
        max_edge_freq : float
            The highest edge frequency of the filter bank (in Hz), that is, the
            edge frequency of the final high-pass.
        numbands : int
            The number of frequency bands of the filter bank.
        w_co : list-like
            A list of edge frequencies (in Hz).  This overrides `numbands` if
            given.
        lowpass_design_func : function, optional
            A function that designs a low-pass filter.  It must implement the
            following API:

                b, a = lowpass_design_func(w_e)

            Its sole argument is the edge frequency normalised to ``fs=1``, and
            its return values are the b and a coefficients of the filter.  The
            default is to design an elliptic filter with ``N=7``, ``rp=1e-5``
            and ``rs=50``.

        Returns
        -------
        AP : numpy.ndarray
            The coefficients of the complementary all-pass filters whose sum
            yields a low-pass and whose difference yields the
            doubly-complementary high-pass filter.  The b and a coefficients
            are in the first and second column, respectively.
        H : numpy.ndarray
            The coefficients of the doubly complementary low-pass and
            high-pass filters described above.
        """

        if not w_co:
            # split edge frequencies evenly (though arbitrarily limit edge
            # frequencies to user configurable value)
            w_co = np.linspace(max_edge_freq/numbands, max_edge_freq,
                               numbands)[:-1]/(fs/2.)
        else:
            w_co = np.array(w_co)
            w_co = w_co/(fs/2.)

        if not lowpass_design_func:
            lowpass_design_func = lambda w: sig.ellip(7, 1e-5, 50, w)
        elif not isfunction(lowpass_design_func):
            raise ValueError('"lowpass_design_func" is not a function.')

        self.__edge_freqs = w_co[::-1]*fs/2

        for i in range(numbands-1):
            # original filter
            #
            # If the pass-band is maximally flat, one could perhaps optimise
            # for performance by approximating the filter by using only lowest
            # necessary LP filter instead of going through all previous
            # filters.

            b_d, a_d = lowpass_design_func(w_co[-1-i])

            A1, A2 = any_to_ap_pair(b_d/a_d[0], a_d/a_d[0])
            H1, H2 = get_power_complementary_filters(A1, A2)

            self.__AP.append((A1, A2))
            self.__H.append((H1, H2))

    def analyze(self, x):
        """Split an input signal into multiple frequency bands using an
        analysis filter bank.

        Parameters
        ----------
        x : numpy.ndarray
            The input signal.

        Returns
        -------
        x_bs : numpy.ndarray
            The band-split signal.
        """

        x = np.atleast_2d(x)

        nchn, nsmp = x.shape

        x_bs = np.zeros((self.__numbands, nchn, nsmp))
        x_bs[0] = x.copy()

        for i, s in enumerate(self.__ana_filters):

            x_bs[i+1] = s[-1].filter(x_bs[i])
            for j, h in enumerate(s[:-1]):
                x_bs[j] = h.filter(x_bs[j])

            # butterfly operation
            x_bs[i:i+2] = np.array((0.5*(x_bs[i+1] - x_bs[i]),
                                    0.5*(x_bs[i+1] + x_bs[i])))

        return x_bs

    def synthesize(self, x_bs):
        """Reconstruct a signal from it's band-split representation using a
        synthesis filter bank.

        Note that due to the all-pass complementary property of the filter
        bank, it is possible to simply sum up the bands.

        Parameters
        ----------
        x_bs : numpy.ndarray
            A band-split signal (filtered output of .analyse()).

        Returns
        -------
        y : numpy.ndarray
            The synthesised output signal.
        """

        y = x_bs.copy()

        for s in self.__syn_filters:
            y[-2:] = np.array((0.5*(y[-1] - y[-2]),
                               0.5*(y[-1] + y[-2])))
            for j, h in enumerate(s):
                y[j] = h.filter(y[j])

            y[-2] = y[-2:].sum(axis=0)
            y = y[:-1]

        return y[0]

    @property
    def AP(self):
        "The all-pass filter pairs that make up the filter bank."
        return self.__AP

    @property
    def H(self):
        """Doubly-complementary low-pass/high-pass pairs, calculated from the
        all-pass pairs.
        """
        return self.__H

    @property
    def edge_freqs(self):
        "The edge frequencies of the filter bank."
        return self.__edge_freqs
