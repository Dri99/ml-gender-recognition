import time

import numpy.linalg

from reduction import *
from preprocess import *
from mvg import *
from logReg import *
from evaluation import *
from threadpoolctl import threadpool_limits


# start = time.time()
# print("compute_min_dcf took %s" % (time.time() - start))


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


if __name__ == '__main__':
    D, L = load('../Gender_Detection/Train.txt')

    # plot_data(D, L, 'diagrams/raw_')
    raw = D
    z_params = get_z_params(D)
    zD = apply_z_center(D, z_params)
    # plot_data(D, L, 'diagrams/raw_')

    # plot_data(zD, L, 'diagrams/centered_')

    gD = gaussianify_slow(zD, zD)

    pca_params = compute_pca(zD, 11)
    pca_d = apply_pca(zD, pca_params)
    # corr_heatmaps(raw, L, "raw")
    # corr_heatmaps(pca_d, L, "pca")

    # lda_params = compute_lda(zD, L)
    # lda_D = evaluete_lda(zD, lda_params)
    # plot_data(gD, L, 'diagrams/gaussianized_')

    # min_dcf_mvg_models(raw, zD, gD, pca_d, L, 5, full_cov_evaluation, 'Full Cov')
    # min_dcf_mvg_models(raw, zD, gD, pca_d, L, 5, tied_cov_evaluation, 'Tied')
    # min_dcf_mvg_models(raw, zD, gD, pca_d, L, 5, diag_cov_evaluation, 'Diag')
    # min_dcf_mvg_models(raw, zD, gD, pca_d, L, 5, diag_tied_cov_evaluation, 'Diag-Tied')

    # corr_heatmaps(raw, L, "raw")
    # corr_heatmaps(zD, L, "centered")
    # corr_heatmaps(gD, L, "guassianized")

    # lambda_tuning(zD, L, 'z')
    # lambda_tuning_quad(zD,L, "z_logReg")
    # lambda_tuning(pca_d, L, 'g')
    # print("Raw")
    # log_reg_min_dcf(raw, L, 1e-4, 0.5)
    # print("Z normalised")
    # log_reg_min_dcf(zD, L, 1e-4, 0.5)
    # print("PCA")
    # log_reg_min_dcf(pca_d, L, 1e-4, 0.5)
    #
    # print("Raw")
    # quad_log_reg_min_dcf(raw, L, 1e-4, 0.5)
    # print("Z normalised")
    # quad_log_reg_min_dcf(zD, L, 1e-4, 0.5)
    # print("PCA")
    # quad_log_reg_min_dcf(pca_d, L, 1e-4, 0.5)

    threadpool_limits(limits=7)
    # plot_C_optimize_linear_svm(zD, L)
    # plot_c_eps_optimize_quadratic_svm(zD, L, 1)
    plot_C_optimize_quadratic_svm(zD, L, 0.2, 0)
    # plot_C_optimize_rbf_svm(zD, L)

    # Maybe old
    ##################
    # kernel_fn = get_poly_kernel(1, 1)
    # rbk_svm_tr, rbf_svm_scorer = get_svm_trainer_model(5, kernel_fn=kernel_fn)
    # svm_sc, svm_l = k_fold_score(zD, L, 5, rbk_svm_tr, rbf_svm_scorer, seed=seed)
    # print("minDcf (pi=0.5): %f " % (compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1)))
    #
    # kernel_fn = get_rbf_kernel(0.01, 1)
    # rbk_svm_tr, rbf_svm_scorer = get_svm_trainer_model(30, kernel_fn=kernel_fn)
    # svm_sc, svm_l = k_fold_score(zD, L, 5, rbk_svm_tr, rbf_svm_scorer, seed=seed)
    # print("minDcf (pi=0.5): %f " % (compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1)))
    ##################
    ##
    # plot_gmm_full(raw, zD, L, 32)

    # plot_roc_curve(s, TL, 'fullCov')
    # plot_roc_curve(LR_s, LR_TL, 'logReg')
    # bayes_error_plot(s, TL, '-tiedCov')
    # bayes_error_plot(LR_s, LR_TL, '-logReg')
    # plt.savefig('diagrams/MVG_LogReg_dcf_2.jpg')
    # plt.show()
    #
    # linear_svm_tr, linear_svm_scorer = get_svm_trainer_model(1, K=1)
    # start = time.time()
    # l_svm_sc, l_svm_l = k_fold_score(zD, L, 5, linear_svm_tr, linear_svm_scorer)
    # min_dcf = compute_min_dcf(l_svm_sc, l_svm_l, 0.5, 1, 1)
    # print("Linear svm C: 1 pi_app: 0.5\t %f" % (min_dcf))
    # bayes_error_plot(s, TL, '-tiedCov')
    # bayes_error_plot(LR_s, LR_TL, '-logReg')
    # bayes_error_plot(l_svm_sc, l_svm_l, '-svm')
    # plt.savefig('diagrams/all_dcf.jpg')
    # plt.show()
    # print("k_fold_score took %d s" % (time.time() - start))
    # bayes_error_plot(l_svm_sc, l_svm_l, '-L_svm')
    # plt.savefig('diagrams/l_svm_dcf_3.jpg')
    # plt.show()

    # log_trainer = get_logReg_trainer(10 ** -4, 0.5)
    # params = log_trainer(raw, L)
    # DT, LT = load('../Pulsar_Detection/Test.txt')
    # Scores = logRegScorer(DT, params)
    # params = tied_cov_evaluation(raw, L)
    # scores_tied = full_cov_log_score_fast(DT, params)
    #
    # bayes_error_plot(Scores, LT, '-final_LogReg')
    # bayes_error_plot(scores_tied, LT, '-final_TiedCov')
    # plt.savefig('diagrams/all_dcf.jpg')
    # plt.show()
    # print(compute_act_dcf(Scores, LT, 0.5, 1, 1))
    # print(compute_act_dcf(scores_tied, LT, 0.5, 1, 1))
    print("end")
