import time

import numpy.linalg

from gmm import score_gmm
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
    # lin_svm_tr, lin_svm_scorer = get_svm_trainer_model(1)
    # svm_sc, svm_l = k_fold_score(zD, L, 5, lin_svm_tr, lin_svm_scorer)
    # print_model_performance(svm_l, svm_sc, model_name='Linear SVM')

    # plot_c_eps_optimize_quadratic_svm(zD, L, 1)
    # plot_C_optimize_quadratic_svm(zD, L, 0.2, 0)

    # Optimisation of RBF SVM
    # plot_C_optimize_rbf_svm(zD, L)
    # best With C = 30 gamma = 0.01 K = 1
    # kernel_fn = get_rbf_kernel(0.01, 1)
    # rbk_svm_tr, rbf_svm_scorer = get_svm_trainer_model(30, kernel_fn=kernel_fn)
    # svm_sc, svm_l = k_fold_score(zD, L, 5, rbk_svm_tr, rbf_svm_scorer)
    # print("minDcf (pi=0.5): %f " % (compute_min_dcf(svm_sc, svm_l, 0.5, 1, 1)))

    # Optimisation of GMM
    # plot_gmm_full(raw, zD, L, 32)
    # gmm_S, gmm_L = k_fold_score(raw, L, 5, get_gmm_tied_cov_trainer(maxComponents=8, singleGmm=True), score_gmm)
    # print_model_performance(gmm_L, gmm_S, model_name='GMM G=8')

    # Check the best model
    # plot_roc_curve(gmm_S, gmm_L, 'GMM')
    # plot_roc_curve(svm_sc, svm_l, 'linSVM')
    # bayes_error_plot(gmm_S, gmm_L, '-gmmTied')
    # bayes_error_plot(svm_sc, svm_l, '-RbfSvm')
    # plt.savefig('diagrams/error_plot_TiedGmm_Rbf.jpg')
    # plt.show()

    # GMM Calibration
    # gmm_S, gmm_SC, gmm_L = validate_score_calibration(gmm_S, gmm_L)
    # bayes_error_plot(gmm_S, gmm_L, '-gmmUncalibrated')
    # bayes_error_plot(gmm_SC, gmm_L, '-gmmCalibrated')
    # plt.savefig('diagrams/error_plot_score_calibration.jpg')
    # plt.show()

    # Experimental results

    DT, LT = load('../Gender_Detection/Test.txt')
    z_params = get_z_params(D)
    zDT = apply_z_center(DT, z_params)
    gDT = gaussianify_slow(zD, zDT)
    pca_params = compute_pca(zD, 11)
    pcaDT = apply_pca(zDT, pca_params)

    validate_mvg_models(raw, zD, gD, pca_d, L, full_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Full cov testing')
    validate_mvg_models(raw, zD, gD, pca_d, L, tied_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Tied cov testing')
    validate_mvg_models(raw, zD, gD, pca_d, L, diag_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Diag cov testing')
    validate_mvg_models(raw, zD, gD, pca_d, L, diag_tied_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Diag-tied testing')
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
