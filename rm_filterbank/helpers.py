import numpy as np

def get_power_complementary_q(P,D):
    """Function to get the non-recursive coefficients Q of a double-complementary
    filter Q/D to a filter P/D with P it's non-recursive coefficients and D it's
    recursive coefficients.  The term "double-complementary" means a group of
    filters whose absolute sum as well as whose sum of absolute powers yields
    unity, i.e.:

            |P(z)/D(z)      +   Q(z)/D(z)|      =   1
      and
            |P(z)/D(z)|^2   +   |Q(z)/D(z)|^2   =   1.

    Parameters:
    -----------

    P, D: The non-recursive and recursive coefficients of the filter,
          respectively.

    Returns:
    --------

    Q: The non-recursive coefficients of the double-complementary filter.
    """

    # make sure we get column vectors
    P = np.atleast_2d(P.flatten()).T
    D = np.atleast_2d(D.flatten()).T

    # get double complementary filter
    # R(z) = P(z)^2 - z^(-n)*D(z^-1)*D(z)
    P_sq = P.dot(P.T)
    D_sq = D[::-1].dot(D.T)

    # sort P and D by power of z, since the distribution of the powers of z of the
    # squared polynomial in matrix form is:
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
    for mm in range(N):
        for nn in range(N):
            P_sq_sort[mm+nn] += P_sq[mm,nn]
            D_sq_sort[mm+nn] += D_sq[mm,nn]

    Q = np.zeros((P.size,))

    # recursively calculate Q from R
    R    = P_sq_sort - D_sq_sort
    Q[0] = np.sqrt(R[0])
    Q[1] = R[1]/(2*Q[0])
    # Q is antisymmetric, so calculate N/2 and generate the rest from that
    for kk in range(2,N/2):
        Q[kk] = (R[kk] - (Q[1:kk].T.dot(Q[kk-1:0:-1])))/(2*Q[0])

    Q[N/2:] = -Q[N/2-1::-1]

    return Q

def get_power_complementary_filters(A1, A2):
    """Function to get a pair of double-complementary filters from a pair of
    complementary all-pass filters.  The term "double-complementary" means a group
    of filters whose abolute sum as well as whose sum of absolute powers yields
    unity, i.e.:

            |H1(z)      +   H2(z)|      =   1
      and
            |H1(z)|^2   +   |H2(z)|^2   =   1,

      where H1 is the original filter from which A1 and A2 were calculated.

    Parameters:
    -----------

    A1, A2: A pair of complementary all-pass filter coefficients (the b and a
            coefficients must be in the first and second column, respectively).

    Returns:
    --------

    H1, H2: Matrices containing the coefficients of double-complementary
            high-pass and low-pass filters.  The b and a coefficients are in the
            first and second column, respectively.
    """

    # extract b and a coefficients
    A1b, A1a = [a.T for a in np.hsplit(A1, 2)]
    A2b, A2a = [a.T for a in np.hsplit(A2, 2)]

    # calculate and sort coefficients of the double-complementary filters

    H1b = 0.5*(A1b.T.dot(A2a) + A1a.T.dot(A2b))
    H1a = A1a.T*A2a
    H2b = 0.5*(A1b.T.dot(A2a) - A1a.T.dot(A2b))

    M, N     = H1b.shape
    H1b_sort = np.zeros((M+N,1))
    H2b_sort = np.zeros((M+N,1))
    H1a_sort = np.zeros((M+N,1))
    H2a_sort = np.zeros((M+N,1))

    for mm in range(M):
        for nn in range(N):
            H1b_sort[mm+nn] += H1b[mm,nn]
            H2b_sort[mm+nn] += H2b[mm,nn]
            H1a_sort[mm+nn] += H1a[mm,nn]
            H2a_sort[mm+nn] += H1a[mm,nn]

    H1 = np.hstack((H1b_sort, H1a_sort))
    H2 = np.hstack((H2b_sort, H2a_sort))

    return (H1, H2)

def any_to_ap_pair(b,a):
    """Function to convert a filter to a pair of double-complementary all-pass
    filters.

    Parameters:
    -----------

      b, a:   Coefficients of the filter design.

    Returns:
    --------

      A1, A2: Matrices containing the coefficients of complementary all-pass
              filters whose sum yields a low-pass and whose difference yields
              the double-complementary high-pass filter.  The b and a
              coefficients are in the first and second column, respectively.
    """

    # make sure we get 1d vectors
    b = b.flatten()
    a = a.flatten()

    P = b/a[0] # numerator polynomial
    D = a/a[0] # denominator polynomial

    # get P's double-complementary polynomial Q
    Q = get_power_complementary_q(P,D)

    z1 = np.roots(P+Q)
    # z2 = np.roots(P-Q)

    # calculate the all-pass functions
    A2a = np.poly(z1[np.abs(z1)<1])
    A2b = A2a[::-1]
    A1b = np.poly(z1[np.abs(z1) >= 1])
    A1a = A1b[::-1]

    # normalize a0 to 1
    A2b, A2a = A2b/A2a[0], A2a/A2a[0]
    A1b, A1a = A1b/A1a[0], A1a/A1a[0]

    A1 = np.vstack((A1b, A1a)).T
    A2 = np.vstack((A2b, A2a)).T

    return A1, A2
