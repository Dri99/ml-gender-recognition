import numpy


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def logpdf_GAU_ND(X, mu, C):
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2 * numpy.pi) + 0.5 * numpy.linalg.slogdet(P)[1]
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i + 1]
        res = const - 0.5 * numpy.dot((x - mu).T, numpy.dot(P, (x - mu)))
        Y.append(res)
    return numpy.array(Y).ravel()


def logpdf_GAU_ND_fast(X, mu, C):
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2 * numpy.pi) + 0.5 * numpy.linalg.slogdet(P)[1]
    Xc = X - mu
    Diag = Xc * numpy.dot(P, Xc)
    return const - 0.5 * Diag.sum(axis=0)


def empirical_mean(X):
    return mcol(X.mean(1))


def empirical_epsilon(X):
    Xc = X - empirical_mean(X)
    return numpy.dot(Xc, Xc.T) / X.shape[1]


def empirical_moments(X):
    return empirical_mean(X), empirical_epsilon(X)


def full_cov_evaluation(data, label):
    mu_0 = empirical_mean(data[:, label == 0])
    eps_0 = empirical_epsilon(data[:, label == 0])
    mu_1 = empirical_mean(data[:, label == 1])
    eps_1 = empirical_epsilon(data[:, label == 1])

    # pi = label.sum(1) / label.size(1)
    # t_emp = -numpy.log(pi) + numpy.log((1 - pi))
    return [(mu_0, eps_0), (mu_1, eps_1)]

def diag_cov_evaluation(data, label):
    mu_0 = empirical_mean(data[:, label == 0])
    eps_0 = empirical_epsilon(data[:, label == 0])
    mu_1 = empirical_mean(data[:, label == 1])
    eps_1 = empirical_epsilon(data[:, label == 1])
    eps_0 = eps_0 * numpy.eye(data.shape[0],data.shape[0])
    eps_1 = eps_1 * numpy.eye(data.shape[0], data.shape[0])
    return [(mu_0, eps_0), (mu_1, eps_1)]

def tied_cov_evaluation(data, label):
    N_0 = numpy.array(label == 0).sum()
    N_1 = label.size - N_0
    mu_0 = empirical_mean(data[:, label == 0])
    sig_0 = empirical_epsilon(data[:, label == 0])
    mu_1 = empirical_mean(data[:, label == 1])
    sig_1 = empirical_epsilon(data[:, label == 1])
    sig = (sig_0 * N_0 + sig_1 * N_1) / label.size
    # pi = N_0 / (N_0 + N_1)
    # t = -numpy.log(pi) + numpy.log((1 - pi))
    return [(mu_0, sig), (mu_1, sig)]


def diag_tied_cov_evaluation(data, label):
    tied_params = tied_cov_evaluation(data, label)
    mu_0 = tied_params[0][0]
    mu_1 = tied_params[1][0]
    sig = tied_params[0][1] * numpy.eye(data.shape[0],data.shape[0])
    return [(mu_0, sig), (mu_1, sig)]


def full_cov_log_score(samples, params, ):
    log_den_0 = logpdf_GAU_ND(samples, params[0][0], params[0][1])
    # log_den_0 = numpy.exp(log_den_0)
    log_den_1 = logpdf_GAU_ND(samples, params[1][0], params[1][1])
    # log_den_1 = numpy.exp(log_den_1)
    threshold = 0
    if len(params) > 2:
        threshold = params[2]
    return log_den_1 - log_den_0 - threshold


def full_cov_log_score_fast(samples, params, ):
    log_den_0 = logpdf_GAU_ND_fast(samples, params[0][0], params[0][1])
    # log_den_0 = numpy.exp(log_den_0)
    log_den_1 = logpdf_GAU_ND_fast(samples, params[1][0], params[1][1])
    # log_den_1 = numpy.exp(log_den_1)
    threshold = 0
    if len(params) > 2:
        threshold = params[2]
    return log_den_1 - log_den_0 - threshold
