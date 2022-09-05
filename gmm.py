import numpy
import scipy.special

from mvg import logpdf_GAU_ND_fast, full_cov_evaluation, logpdf_GAU_ND


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def logpdf_GMM(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_fast(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)  # N,


def GMM_EM(X, gmm, psi, delta_t, type=''):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    M = X.shape[0]

    while llOld is None or llNew - llOld > delta_t:
        # while llOld is None or cycles > 0:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_fast(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum(0) / N
        P_int = SJ - SM
        P = numpy.exp(P_int)
        gmmNew = []

        sigmaTied = numpy.zeros((M, M))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (mrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - numpy.dot(mu, mu.T)

            if type == 'tied':
                sigmaTied += Z * Sigma
                gmmNew.append((w, mu))
                continue
            elif type == 'diag':
                Sigma *= numpy.eye(Sigma.shape[0])

            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Ms = s * numpy.eye(s.size)
            covNew = numpy.dot(U, numpy.dot(Ms, U.T))
            gmmNew.append((w, mu, covNew))

        if type == 'tied':
            sigmaTied /= N
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s < psi] = psi
            Ms = s * numpy.eye(s.size)
            sigmaTied = numpy.dot(U, numpy.dot(Ms, U.T))
            gmmNew = list(map(lambda tuple_p: (tuple_p[0], tuple_p[1], sigmaTied), gmmNew))
        gmm = gmmNew

        # sigma_c = numpy.zeros((M, M))
        # for g in range(G):
        #     gamma = P[g, :]
        #     Z = gamma.sum()
        #     sigma_c += Z * gmm[g][2]
        # sigma_c = sigma_c / G
        # for g in range(G):
        #     gmm[g] = (gmm[g][0], gmm[g][1], sigma_c)

        # if llOld is not None and llNew < llOld:
        #    raise "GMM did not optimize"
    # print('diff: %f' % (llNew - llOld))
    # print(llOld)
    # print(llNew)
    # print('Em done')
    return gmm


def GMM_EM_2(DT, gmm, psi, type, diff=1e-6):
    D, N = DT.shape
    to = None
    tn = None

    while to == None or tn - to > diff:
        to = tn
        S, logD = logpdf_GMM(DT, gmm)
        tn = logD.sum() / N
        P = numpy.exp(S - logD)

        newGmm = []
        sigmaTied = numpy.zeros((D, D))
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (mrow(gamma) * DT).sum(1)
            S = numpy.dot(DT, (mrow(gamma) * DT).T)
            w = Z / P.sum()
            mu = mcol(F / Z)
            sigma = (S / Z) - numpy.dot(mu, mu.T)
            if type == 'tied':
                sigmaTied += Z * sigma
                newGmm.append((w, mu))
                continue
            elif type == 'diag':
                sigma *= numpy.eye(sigma.shape[0])
            U, s, _ = numpy.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = numpy.dot(U, mcol(s) * U.T)
            newGmm.append((w, mu, sigma))

        if type == 'tied':
            sigmaTied /= N
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s < psi] = psi
            sigmaTied = numpy.dot(U, mcol(s) * U.T)
            newGmm2 = []
            for i in range(len(newGmm)):
                (w, mu) = newGmm[i]
                newGmm2.append((w, mu, sigmaTied))
            newGmm = newGmm2
        gmm = newGmm
        # print("%", tn)

    # print("EM done")
    return gmm


def lbg_split(X, gmm, alpha, psi, delta_t, type):
    G = len(gmm)
    # print("Producing 2*%d-GMM from %d-GMM" % (G, G))
    newGmm = []
    for g in range(G):
        w = gmm[g][0]
        mu = gmm[g][1]
        Sigma_g = gmm[g][2]
        U, s, _ = numpy.linalg.svd(Sigma_g)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        newGmm = newGmm + [(w / 2, mu + d, Sigma_g), (w / 2, mu - d, Sigma_g)]
    gmm = GMM_EM(X, newGmm, psi, delta_t, type)
    return gmm


def train_lbg_gmm(data, labels, alpha, psi, delta_t, maxComponents, type, singleGmm):
    theta_0 = full_cov_evaluation(data, labels)
    X_0 = data[:, labels == 0]
    X_1 = data[:, labels == 1]
    All_GMMS_0 = {}
    All_GMMS_1 = {}
    GMM_0 = [(1.0, theta_0[0][0], theta_0[0][1])]
    GMM_1 = [(1.0, theta_0[1][0], theta_0[1][1])]
    components = 1
    GMM_0 = GMM_EM(X_0, GMM_0, psi, delta_t, type)
    GMM_1 = GMM_EM(X_1, GMM_1, psi, delta_t, type)
    All_GMMS_0[1] = GMM_0
    All_GMMS_1[1] = GMM_1
    while components < maxComponents:
        GMM_0 = lbg_split(X_0, GMM_0, alpha, psi, delta_t, type)
        GMM_1 = lbg_split(X_1, GMM_1, alpha, psi, delta_t, type)
        components = components * 2
        All_GMMS_0[components] = GMM_0
        All_GMMS_1[components] = GMM_1
    if singleGmm:
        return All_GMMS_0[maxComponents], All_GMMS_1[maxComponents]
    else:
        return All_GMMS_0, All_GMMS_1


def get_gmm_full_cov_trainer(alpha=0.1, psi=0.01, delta_t=1e-6, maxComponents=16, singleGmm=False):
    def inner_trainer(data, labels):
        return train_lbg_gmm(data, labels, alpha, psi, delta_t, maxComponents, '', singleGmm)

    return inner_trainer


def get_gmm_tied_cov_trainer(alpha=0.1, psi=0.01, delta_t=1e-6, maxComponents=16, singleGmm=False):
    def inner_trainer(data, labels):
        return train_lbg_gmm(data, labels, alpha, psi, delta_t, maxComponents, 'tied', singleGmm)

    return inner_trainer


def get_gmm_diag_cov_trainer(alpha=0.1, psi=0.01, delta_t=1e-6, maxComponents=16, singleGmm=False):
    def inner_trainer(data, labels):
        return train_lbg_gmm(data, labels, alpha, psi, delta_t, maxComponents, 'diag', singleGmm)

    return inner_trainer


def score_gmm(test_data, params):
    GMM_0, GMM_1 = params
    log_den_0 = logpdf_GMM(test_data, GMM_0)
    log_den_1 = logpdf_GMM(test_data, GMM_1)
    return log_den_1 - log_den_0


def score_multiple_gmm(test_data, params):
    All_GMMS_0, All_GMMS_1 = params
    Scores = {}
    for G in sorted(All_GMMS_0):
        single_gmm_tuple = (All_GMMS_0[G], All_GMMS_1[G])
        s = score_gmm(test_data, single_gmm_tuple)
        Scores[G] = s.ravel()
    return Scores
