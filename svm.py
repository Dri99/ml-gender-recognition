import time

import numpy
import scipy.optimize
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def train_svm(data, label, C, gamma, kernel_fn, K, pi_t):
    M = data.shape[0]
    N = data.shape[1]
    DTR = data
    Z = label
    Z = (Z - 0.5) * 2

    pi_emp = (label == 1).sum() / N
    Ct = C
    Cf = C
    if pi_t is not None:
        Ct = C * pi_t / pi_emp
        Cf = C * (1 - pi_t) / (1 - pi_emp)

    if kernel_fn is None:
        DTR = numpy.vstack([data, numpy.ones((1, N)) * K])
        H = numpy.dot(DTR.T, DTR)
    else:
        H = kernel_fn(data, data)

    H = mcol(Z) * mrow(Z) * H

    Cs = numpy.zeros(N)
    Cs[label == 1] = Ct
    Cs[label == 0] = Cf

    def JDual(alpha):
        # start = time.time()
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        al = alpha.sum()
        # print("compute_min_dcf took %s" % (time.time() - start))
        return -0.5 * aHa.ravel() + al, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = numpy.dot(mrow(w), DTR)
        loss = numpy.maximum(numpy.zeros(S.shape), -Cs * S * Z).sum()
        return 0.5 * numpy.linalg.norm(w) ** 2 + loss

    bounds = numpy.array([(0, C)] * N)
    bounds[label == 1] = (0, Ct)
    bounds[label == 0] = (0, Cf)

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(  # N,
        LDual,
        numpy.zeros(N),  # N,
        bounds=bounds,
        factr=0.1,
        maxiter=100000,
        maxfun=100000,
    )

    if kernel_fn is None:
        wStar = numpy.dot(DTR, mcol(alphaStar) * mcol(Z))  # M+1 x N * Nx1 -> M+1 x 1
        # jd = JDual(alphaStar)[0]
        # print("JDual: %f" % jd)
        # j = JPrimal(wStar)
        # print("JPrimal: %f" % j)
        # duality_gap = jd - j
        # print("Duality gap: %f" % duality_gap)
        return wStar
    else:
        SV = DTR[:, alphaStar != 0]
        alphaStar = (Z * alphaStar)[alphaStar != 0]
        return SV, alphaStar, kernel_fn


def get_svm_trainer_model(C, gamma=None, kernel_fn=None, K=1, pi_t=None):
    def trainer(DTR, DTL):
        return train_svm(DTR, DTL, C, gamma, kernel_fn, K, pi_t)

    if kernel_fn is None:
        return trainer, score_svm_linear
    else:
        return trainer, score_svm_non_linear


def score_svm_linear(test_data, params):
    w = params[0:test_data.shape[0], :]
    b = params[-1, :]
    return (numpy.dot(w.T, test_data) + b).ravel()


def score_svm_non_linear(test_data, params):
    SV, alpha_z, kernel_fn = params
    return score_svm_non_linear_expanded(SV, test_data, alpha_z, kernel_fn)


def score_svm_non_linear_expanded(SV, test_data, alpha_z, kernel_fn):
    # alpha_z Nt,
    # SV MxNt
    k = kernel_fn(SV, test_data)  # ->NtxNe
    return numpy.dot(mrow(alpha_z), k).ravel()  # Ne,


def get_poly_kernel(d, c, eps):
    def inner_kernel(x1, x2, ):
        return poly_kernel(x1, x2, d, c, eps)

    return inner_kernel


def poly_kernel(x1, x2, d, c, eps):
    XtX = numpy.dot(x1.T, x2) + c  # N1xN2
    return XtX ** d + eps


def get_rbf_kernel(gamma, K):
    def inner_kernel(x1, x2, ):
        return rbf_kernel(x1, x2, gamma, K)

    return inner_kernel


def rbf_kernel(x1, x2, gamma, K):
    distances = mcol((x1 ** 2).sum(0)) + mrow((x2 ** 2).sum(0)) - 2 * numpy.dot(x1.T, x2)
    H = numpy.exp(-gamma * distances) + K
    return H
