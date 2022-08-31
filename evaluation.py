import time

import numpy
import matplotlib.pyplot as plt

import mvg
from gmm import get_gmm_full_cov_trainer, score_multiple_gmm, get_gmm_tied_cov_trainer, get_gmm_diag_cov_trainer
from svm import get_svm_trainer_model, get_poly_kernel, get_rbf_kernel

# ii32 = numpy.iinfo(numpy.int32)
# seed = numpy.int32(numpy.random.rand() * ii32.max)
seed = 223134132


def k_fold_score(D, L, K, model_builder, model_scorer, seed=seed, gmm_G=None):
    folds = range(D.shape[1])
    folds = numpy.remainder(folds, K)
    folds = numpy.sort(folds)
    stacked = numpy.vstack([D, L]).T
    numpy.random.seed(seed)
    stacked = numpy.random.permutation(stacked).T
    fold_D = stacked[0:D.shape[0], :]
    fold_L = stacked[D.shape[0], :]
    S = numpy.array([])

    if gmm_G is not None:
        S = {}
        g = 1
        while g <= gmm_G:
            S[g] = numpy.array([])
            g = g * 2
    for i in range(K):
        DTR = fold_D[:, folds != i]
        DTE = fold_D[:, folds == i]
        LTR = fold_L[folds != i,]
        LTE = fold_L[folds == i,]

        model_parameter = model_builder(DTR, LTR)

        scores = model_scorer(DTE, model_parameter, )
        if gmm_G is None:
            S = numpy.concatenate((S, scores))
        else:
            for G, S_ in S.items():
                S[G] = numpy.concatenate((S_, scores[G]))

    return S, fold_L


import GMM_giulia

import GMM_git


def k_fold_giulia_gmm(D, L, K, seed=seed):
    folds = range(D.shape[1])
    folds = numpy.remainder(folds, K)
    folds = numpy.sort(folds)
    stacked = numpy.vstack([D, L]).T
    numpy.random.seed(seed)
    stacked = numpy.random.permutation(stacked).T
    fold_D = stacked[0:D.shape[0], :]
    fold_L = stacked[D.shape[0], :]
    fold_L = fold_L.astype(int)
    S = numpy.array([])

    for i in range(K):
        DTR = fold_D[:, folds != i]
        DTE = fold_D[:, folds == i]
        LTR = fold_L[folds != i,]
        LTE = fold_L[folds == i,]

        trainer = GMM_git.GMM()

        trainer = trainer.trainClassifier(DTR, LTR, 16)

    return


def plot_roc_curve(llrs, labels, line_title):
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    fpr = numpy.zeros(thresholds.size)
    tpr = numpy.zeros(thresholds.size)
    # fnr = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        pred = numpy.int32(llrs > t)
        Conf = numpy.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Conf[i, j] = ((pred == i) * (labels == j)).sum()
        tpr[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1])
        fpr[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])
        # fnr[idx] = Conf[0, 1] / (Conf[0, 1] + Conf[1, 1])

    plt.plot(fpr, tpr, label=line_title)
    plt.legend()
    plt.title('ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


def assign_labels(scores, pi, c_fn, c_fp, th=None):
    if th is None:
        th = -numpy.log(pi * c_fn) + numpy.log((1 - pi) * c_fp)
    predicted = scores > th
    return numpy.int32(predicted)


def compute_confusion_matrix(pred, label):
    CM = numpy.zeros((2, 2))
    CM[0, 0] = ((pred == 0) * (label == 0)).sum()
    CM[0, 1] = ((pred == 0) * (label == 1)).sum()
    CM[1, 0] = ((pred == 1) * (label == 0)).sum()
    CM[1, 1] = ((pred == 1) * (label == 1)).sum()
    # for i in range(2):
    #     for j in range(2):
    #         CM[i, j] = ((pred == i) * (label == j)).sum()
    return CM


def compute_emp_bayes(CM, pi, c_fn, c_fp, ):
    fpr = CM[1, 0] / (CM[1, 0] + CM[0, 0])
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    return pi * c_fn * fnr + (1 - pi) * c_fp * fpr


def compute_normalized_emp_bayes(CM, pi, c_fn, c_fp):
    empBayes = compute_emp_bayes(CM, pi, c_fn, c_fp)
    return empBayes / min(pi * c_fn, (1 - pi) * c_fp)


def compute_act_dcf(scores, labels, pi, c_fn, c_fp, th=None):
    predicted = assign_labels(scores, pi, c_fn, c_fp, th)
    CM = compute_confusion_matrix(predicted, labels)
    dcf = compute_normalized_emp_bayes(CM, pi, c_fn, c_fp)
    return dcf


def compute_min_dcf(scores, labels, pi, c_fn, c_fp):
    t = numpy.array(scores)
    t.sort()
    t = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcf_list = []
    for _th in t:
        dcf_list.append(compute_act_dcf(scores, labels, pi, c_fn, c_fp, th=_th))

    return numpy.array(dcf_list).min(initial=2)


def bayes_error_plot_y(p_array, scores, labels, min_cost=False):
    y = []

    for i, p in enumerate(p_array):
        pi = 1.0 / (1.0 + numpy.exp(-p))
        if min_cost:
            y.append(compute_min_dcf(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_dcf(scores, labels, pi, 1, 1))

    return numpy.array(y)


def bayes_error_plot(scores, labels, title):
    p_s = numpy.linspace(-3, 3, 50)
    plt.plot(p_s, bayes_error_plot_y(p_s, scores, labels, min_cost=False), label='actualDCF' + title)
    plt.plot(p_s, bayes_error_plot_y(p_s, scores, labels, min_cost=True), label='minDCF' + title)
    plt.xlabel(r'log$\frac{\tilde{π}}{1-\tilde{π}}$')
    # r'log$\frac{$\tilde{π}$}{1-$\tilde{π}$}$'
    plt.legend()
    plt.grid()


def min_dcf_mvg_models(raw, zD, gD, pca_d, L, K, trainer, type='Full Cov'):
    s, TL = k_fold_score(raw, L, K, trainer, mvg.full_cov_log_score_fast, seed=seed)
    zs, zTL = k_fold_score(zD, L, K, trainer, mvg.full_cov_log_score_fast, seed=seed)
    gs, gTL = k_fold_score(gD, L, K, trainer, mvg.full_cov_log_score_fast, seed=seed)
    pcas, pcaTL = k_fold_score(pca_d, L, K, trainer, mvg.full_cov_log_score_fast, seed=seed)
    print("%", type)
    print("Raw")
    print(compute_min_dcf(s, TL, 0.5, 1, 1))
    print(compute_min_dcf(s, TL, 0.1, 1, 1))
    print("Z")
    print(compute_min_dcf(zs, zTL, 0.5, 1, 1))
    print(compute_min_dcf(zs, zTL, 0.1, 1, 1))
    print("Gauss")
    print(compute_min_dcf(gs, gTL, 0.5, 1, 1))
    print(compute_min_dcf(gs, gTL, 0.1, 1, 1))
    print("PCA")
    print(compute_min_dcf(pcas, pcaTL, 0.5, 1, 1))
    print(compute_min_dcf(gs, gTL, 0.1, 1, 1))


def plot_C_optimize_linear_svm(data, label):
    Cs = numpy.linspace(-3, 1, 50)
    Cs = numpy.power(10, Cs)
    line_1 = []
    line_2 = []
    for i, C in enumerate(Cs):
        start = time.time()
        linear_svm_tr, linear_svm_scorer = get_svm_trainer_model(C, K=1)
        l_svm_sc, l_svm_l = k_fold_score(data, label, 5, linear_svm_tr, linear_svm_scorer, seed=seed)
        print("k_fold_score took %d s" % (time.time() - start))
        line_1.append(compute_min_dcf(l_svm_sc, l_svm_l, 0.5, 1, 1))
        line_2.append(compute_min_dcf(l_svm_sc, l_svm_l, 0.1, 1, 1))
    plt.plot(Cs, line_1, label=r'$minDCF (\tilde{π} = 0.5)$')
    plt.plot(Cs, line_2, label=r'$minDCF (\tilde{π} = 0.1)$')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('diagrams/C_tune_lin_svm(2).jpg')
    plt.show()


def plot_c_eps_optimize_quadratic_svm(data, label, C):
    cs = numpy.linspace(-2, +2, 40)
    cs = numpy.power(10, cs)
    lines = [[], [], [], []]

    for i, c in enumerate(cs):
        for j, eps in enumerate([0, 0.1, 1, 10]):
            kernel_fn = get_poly_kernel(2, c, eps)
            start = time.time()
            quad_svm_tr, quad_svm_scorer = get_svm_trainer_model(C, kernel_fn=kernel_fn)
            svm_sc, svm_l = k_fold_score(data, label, 5, quad_svm_tr, quad_svm_scorer, seed=seed)
            print("k_fold_score took %d s" % (time.time() - start))
            start = time.time()
            lines[j].append(compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1))
            print("compute_min_dcf took %d s" % (time.time() - start))
            print("minDcf (pi=0.5): %f at c=%f eps=%f" % (lines[j][i], c, eps))
            print("%d over %d" % (i, cs.size))
    plt.plot(cs, lines[0], label=r'$minDCF ξ = 0 $')
    plt.plot(cs, lines[1], label=r'$minDCF ξ = 0.1 $')
    plt.plot(cs, lines[2], label=r'$minDCF ξ = 1 $')
    plt.plot(cs, lines[3], label=r'$minDCF ξ = 10 $')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('c')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('diagrams/c_eps_tune_quad_svm.jpg')
    plt.show()


def plot_C_optimize_quadratic_svm(data, label, c, eps):
    Cs = numpy.linspace(-1, +2, 40)
    Cs = numpy.power(10, Cs)[::-1]
    line_1 = []
    line_2 = []
    kernel_fn = get_poly_kernel(2, c, eps)
    for i, C in enumerate(Cs):
        start = time.time()
        quad_svm_tr, quad_svm_scorer = get_svm_trainer_model(C, kernel_fn=kernel_fn)
        svm_sc, svm_l = k_fold_score(data, label, 4, quad_svm_tr, quad_svm_scorer, seed=seed)
        print("k_fold_score took %d s" % (time.time() - start))
        start = time.time()
        line_1.insert(0, compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1))
        line_2.insert(0, compute_min_dcf(svm_sc, svm_l, 0.1, 1, 1))
        print("compute_min_dcf took %d s" % (time.time() - start))
        print("minDcf (pi=0.5): %f" % line_1[0])
        print("%d over %d" % (i, Cs.size))

    Cs = Cs[::-1]
    plt.plot(Cs, line_1, label=r'$minDCF (\tilde{π} = 0.5)$')
    plt.plot(Cs, line_2, label=r'$minDCF (\tilde{π} = 0.1)$')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('diagrams/C_tune_quad_svm.jpg')
    plt.show()


def plot_gamma_optimize_rbf_svm(data, label):
    gammas = numpy.linspace(-3, 0, 30)
    gammas = numpy.power(10, gammas)
    lines = [[], [], [], []]

    for i, gamma in enumerate(gammas):
        for j, eps in enumerate([0, 0.1, 1, 10]):
            kernel_fn = get_rbf_kernel(gamma, eps)
            start = time.time()
            rbk_svm_tr, rbf_svm_scorer = get_svm_trainer_model(1, kernel_fn=kernel_fn)
            svm_sc, svm_l = k_fold_score(data, label, 4, rbk_svm_tr, rbf_svm_scorer, seed=seed)
            print("k_fold_score took %d s" % (time.time() - start))
            start = time.time()
            lines[j].append(compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1))
            print("compute_min_dcf took %d s" % (time.time() - start))
            print("minDcf (pi=0.5): %f" % lines[j][i])
            print("%d over %d" % (i, gammas.size))
    plt.plot(gammas, lines[0], label=r'$minDCF ξ = 0$')
    plt.plot(gammas, lines[1], label=r'$minDCF ξ = 0.1$')
    plt.plot(gammas, lines[2], label=r'$minDCF ξ = 1$')
    plt.plot(gammas, lines[3], label=r'$minDCF ξ = 10$')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('gamma')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('diagrams/gamma_tune_rbf_svm.jpg')
    plt.show()


def plot_C_optimize_rbf_svm(data, label):
    Cs = numpy.linspace(-1, 2, 30)
    Cs = numpy.power(10, Cs)[::-1]
    lines = [[], [], []]

    for i, C in enumerate(Cs):
        for j, gamma in enumerate([0.01, 0.1, 10]):
            kernel_fn = get_rbf_kernel(gamma, 1)
            start = time.time()
            rbk_svm_tr, rbf_svm_scorer = get_svm_trainer_model(C, kernel_fn=kernel_fn)
            svm_sc, svm_l = k_fold_score(data, label, 4, rbk_svm_tr, rbf_svm_scorer, seed=seed)
            print("k_fold_score took %d s" % (time.time() - start))
            start = time.time()
            # lines[j].append(compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1))
            lines[j].insert(0, compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1))
            print("compute_min_dcf took %d s" % (time.time() - start))
            print("minDcf (pi=0.5): %f at C: %f" % (lines[j][i], C))
            print("%d over %d" % (i, Cs.size))

    Cs = Cs[::-1]
    plt.plot(Cs, lines[0], label=r'$minDCF γ = 0.01$')
    plt.plot(Cs, lines[1], label=r'$minDCF γ = 0.1$')
    plt.plot(Cs, lines[2], label=r'$minDCF γ = 10$')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('diagrams/C_gamma_tune_rbf_svm_fine.jpg')
    plt.show()


def plot_gmm_full(data, gData, label, maxComponents):
    trainer = get_gmm_diag_cov_trainer(psi=0.01, maxComponents=maxComponents)
    scores, sc_label = k_fold_score(data, label, 5, trainer, score_multiple_gmm, gmm_G=maxComponents)
    minDCFs = {}
    for G, scores_ in scores.items():
        minDcf = compute_min_dcf(scores_, sc_label, 0.5, 1, 1)
        minDCFs[G] = minDcf
    Gs = []
    for g in minDCFs.keys():
        Gs.append(str(g))
    x = numpy.arange(len(Gs))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x + width / 2, minDCFs.values(), width=0.35, label=r'$minDCF (\tilde{π} = 0.5)$ -Raw ')

    scores, sc_label = k_fold_score(gData, label, 5, trainer, score_multiple_gmm, gmm_G=maxComponents)
    minDCFs = {}
    for G, scores_ in scores.items():
        minDcf = compute_min_dcf(scores_, sc_label, 0.5, 1, 1)
        minDCFs[G] = minDcf
    Gs = []
    for g in minDCFs.keys():
        Gs.append(str(g))
    rects2 = ax.bar(x - width / 2, minDCFs.values(), width=0.35, label=r'$minDCF (\tilde{π} = 0.5)$ - Z-Normalised')

    ax.grid(True, axis='y')
    ax.set_ylabel('minDCF')
    ax.set_xlabel('GMM Components')
    ax.set_xticks(x, Gs)
    ax.legend()
    ax.bar_label(rects1, labels=None, padding=3)
    ax.bar_label(rects2, labels=None, padding=3)
    fig.tight_layout()
    plt.savefig('diagrams/gmm_result.jpg')
    plt.show()
