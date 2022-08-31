import numpy
import scipy

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def compute_pca(data, m):
    c_data = data - mcol(data.mean(axis=1))
    C = numpy.dot(c_data, c_data.T)
    U, s, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P


def apply_pca(data, params):
    return numpy.dot(params.T, data)


def compute_lda(data, labels):
    N_1 = labels.sum()
    N = len(labels)
    N_0 = N - N_1

    d_0 = data[:, labels == 0]
    d_1 = data[:, labels == 1]

    mu = data.mean(1)
    mu_0 = mcol(data[:, labels == 0].mean(1))
    mu_1 = mcol(data[:, labels == 1].mean(1))

    C_0 = numpy.dot(d_0 - mu_0, (d_0 - mu_0).T)
    C_1 = numpy.dot(d_1 - mu_1, (d_1 - mu_1).T)
    SW = (C_0 * N_0 + C_1 * N_1) / N

    SB = (numpy.dot(mu_0 - mu, (mu_0 - mu).T) * N_0 + numpy.dot(mu_1 - mu, (mu_1 - mu).T) * N_1) / N

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0: 1]

    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:1]
    return U


def apply_lda(data, V):
    V = mrow(V)
    return numpy.dot(data, V)
