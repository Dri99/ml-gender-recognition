import numpy
import scipy.optimize
from evaluation import *


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def logreg_obj_wrap(DTR, LTR, l, pi_t=None):
    Z = LTR * 2.0 - 1.0  # 1xN
    M = DTR.shape[0]
    N = LTR.size
    if pi_t is None:
        pi_t = LTR.sum() / N

    def logreg_obj_func(v):
        w = mcol(v[0:M])  # 1xM
        b = v[-1]  # scalar

        regularizer = l / 2 * numpy.linalg.norm(w) ** 2  # scalar
        S = numpy.dot(w.T, DTR) + b  # (1xM)x(MxN) -> 1xN
        CXE = numpy.logaddexp(numpy.zeros(Z.size), -Z * S)  # product term by term
        CXE_0 = CXE[:, LTR == 0]
        CXE_1 = CXE[:, LTR == 1]

        return regularizer + CXE_1.mean() * pi_t + CXE_0.mean() * (1 - pi_t)

    return logreg_obj_func


def log_reg_evaluation(data, label, _lambda, pi_t=None):
    logreg_funct = logreg_obj_wrap(data, label, _lambda, pi_t)
    v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_funct, numpy.zeros(data.shape[0] + 1), approx_grad=True)
    w = v[0:data.shape[0]]
    b = v[-1]
    return w, b


def get_logReg_trainer(_lambda, pi_t=None):
    def internal_fn(in_data, in_label):
        return log_reg_evaluation(in_data, in_label, _lambda, pi_t)

    return internal_fn


def logRegScorer(data, params):
    w = params[0]
    b = params[1]
    return numpy.dot(w.T, data) + b


def lambda_tuning(data, labels, name):
    lambdas = numpy.linspace(-5, 2, 20)
    lambdas = numpy.power(10, lambdas)
    min_dcfs_bal = []
    min_dcfs_unbal = []
    min_dcfs_unbal2 = []
    for l in lambdas:
        start = time.time()
        log_reg_evaluator = get_logReg_trainer(l)
        s, TL = k_fold_score(data, labels, 5, log_reg_evaluator, logRegScorer, seed=seed)
        min_dcfs_bal.append(compute_min_dcf(s, TL, 0.5, 1, 1))
        min_dcfs_unbal.append(compute_min_dcf(s, TL, 0.1, 1, 1))
        min_dcfs_unbal2.append(compute_min_dcf(s, TL, 0.9, 1, 1))
        print("λ took %s" % (time.time() - start))

    min_overall = numpy.array(min_dcfs_bal).min(initial=2)
    plt.plot(lambdas, min_dcfs_bal, label=r'$minDCF (\tilde{π} = 0.5)$')
    plt.plot(lambdas, min_dcfs_unbal, label=r'$minDCF (\tilde{π} = 0.1)$')
    plt.plot(lambdas, min_dcfs_unbal2, label=r'$minDCF (\tilde{π} = 0.9)$')
    plt.title('minDCF per λ hyperparameter')
    plt.xscale('log')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('diagrams/lambda_tune_' + name + '.jpg')
    plt.show()
    return min_overall

def lambda_tuning_quad(data, labels, name):
    lambdas = numpy.linspace(-5, 2, 20)
    lambdas = numpy.power(10, lambdas)
    min_dcfs_bal = []
    min_dcfs_unbal = []
    min_dcfs_unbal2 = []
    for l in lambdas:
        start = time.time()
        log_reg_evaluator = get_quadLogReg_trainer(l)
        s, TL = k_fold_score(data, labels, 5, log_reg_evaluator, quad_log_reg_scorer, seed=seed)
        min_dcfs_bal.append(compute_min_dcf(s, TL, 0.5, 1, 1))
        min_dcfs_unbal.append(compute_min_dcf(s, TL, 0.1, 1, 1))
        min_dcfs_unbal2.append(compute_min_dcf(s, TL, 0.9, 1, 1))
        print("λ took %s.2 s" % (time.time() - start))

    min_overall = numpy.array(min_dcfs_bal).min(initial=2)
    plt.plot(lambdas, min_dcfs_bal, label=r'$minDCF (\tilde{π} = 0.5)$')
    plt.plot(lambdas, min_dcfs_unbal, label=r'$minDCF (\tilde{π} = 0.1)$')
    plt.plot(lambdas, min_dcfs_unbal2, label=r'$minDCF (\tilde{π} = 0.9)$')
    plt.title('minDCF per λ hyperparameter')
    plt.xscale('log')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('diagrams/lambda_tune_quad_' + name + '.jpg')
    plt.show()
    return min_overall


def log_reg_min_dcf(data, label, lambd, pi_tr):
    log_reg_evaluator = get_logReg_trainer(lambd, pi_tr)
    s, TL = k_fold_score(data, label, 5, log_reg_evaluator, logRegScorer, seed=seed)
    min_dcf = compute_min_dcf(s, TL, 0.5, 1, 1)
    print("LogReg lambda: 10e-4  pi_tr: %f pi_app: 0.5\t %f" % (pi_tr, min_dcf))
    min_dcf = compute_min_dcf(s, TL, 0.1, 1, 1)
    print("LogReg lambda: 10e-4  pi_tr: %f pi_app: 0.1\t %f" % (pi_tr, min_dcf))
    min_dcf = compute_min_dcf(s, TL, 0.9, 1, 1)
    print("LogReg lambda: 10e-4  pi_tr: %f pi_app: 0.9\t %f" % (pi_tr, min_dcf))


def quad_log_reg_min_dcf(data, label, lambd, pi_tr):

        log_reg_evaluator = get_quadLogReg_trainer(lambd, pi_tr)
        s, TL = k_fold_score(data, label, 5, log_reg_evaluator, quad_log_reg_scorer, seed=seed)
        min_dcf = compute_min_dcf(s, TL, 0.5, 1, 1)
        print("LogReg lambda: %f  pi_tr: %f pi_app: 0.5\t %f" % (lambd,pi_tr, min_dcf))
        min_dcf = compute_min_dcf(s, TL, 0.1, 1, 1)
        print("LogReg lambda: %f  pi_tr: %f pi_app: 0.1\t %f" % (lambd,pi_tr, min_dcf))
        min_dcf = compute_min_dcf(s, TL, 0.9, 1, 1)
        print("LogReg lambda: %f  pi_tr: %f pi_app: 0.9\t %f" % (lambd,pi_tr, min_dcf))


def phi(data):
    Y = []
    for i in range(data.shape[1]):
        x = mcol(data[:, i])
        M = numpy.hstack((x * x.T, x))
        y = numpy.ravel(M, order='F')
        Y.append(y)
    Y = numpy.array(Y).T
    return Y


def get_quadLogReg_trainer(_lambda, pi_t=None):
    def internal_fn(in_data, in_label):
        return quad_log_reg_evaluation(in_data, in_label, _lambda, pi_t)

    return internal_fn


def quad_log_reg_evaluation(data, label, _lambda, pi_t=None):
    phied = phi(data)
    logreg_funct = logreg_obj_wrap(phied, label, _lambda, pi_t)
    v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_funct, numpy.zeros(phied.shape[0] + 1), approx_grad=True)
    w = v[0:phied.shape[0]]
    c = v[-1]
    A = w[0: data.shape[0] ** 2]
    b = w[data.shape[0] ** 2:]
    A = A.reshape((data.shape[0], data.shape[0]))
    A = A.T
    return (A, b, c)


def quad_log_reg_scorer(data, params):
    A = params[0]
    b = params[1]
    c = params[2]
    xpAx = (data * numpy.dot(A, data)).sum(axis=0)
    return xpAx + numpy.dot(b.T, data) + c
