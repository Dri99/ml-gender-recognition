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
    # print("Raw 1e-5")
    # log_reg_min_dcf(raw, L, 1e-5, 0.5)
    # print("Raw")
    # log_reg_min_dcf(raw, L, 1e-4, 0.5)
    # print("Z normalised")
    # log_reg_min_dcf(zD, L, 1e-4, 0.5)
    # print("PCA")
    # log_reg_min_dcf(pca_d, L, 1e-4, 0.5)
    # print("Gaussian")
    # log_reg_min_dcf(gD, L, 1e-4, 0.5)

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
    # kernel_fn = get_poly_kernel(2, 0.2, 0)
    # quad_svm_tr, quad_svm_scorer = get_svm_trainer_model(1.5, kernel_fn=kernel_fn)
    # svm_sc, svm_l = k_fold_score(zD, L, 5, quad_svm_tr, quad_svm_scorer)
    # print_model_performance(L, zs=svm_sc, zL=svm_l, model_name='QuadSVM')

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
    # plot_roc_curve(svm_sc, svm_l, 'RbfSVM')
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

    # MVG
    # validate_mvg_models(raw, zD, gD, pca_d, L, full_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Full cov testing')
    # validate_mvg_models(raw, zD, gD, pca_d, L, tied_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Tied cov testing')
    # validate_mvg_models(raw, zD, gD, pca_d, L, diag_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Diag cov testing')
    # validate_mvg_models(raw, zD, gD, pca_d, L, diag_tied_cov_evaluation, DT, zDT, gDT, pcaDT, LT, 'Diag-tied testing')

    # Logistic Regression
    # trainer = get_logReg_trainer(1e-4, 0.5)
    # params = trainer(raw, L)
    # print_model_performance(LT, logRegScorer(DT, params), model_name='LogReg')
    # params = trainer(zD, L)
    # s_logReg = logRegScorer(zDT, params)
    # print_model_performance(LT, zs=s_logReg, zL=LT, model_name='LogReg')
    # params = trainer(pca_d, L)
    # print_model_performance(LT, pcas=logRegScorer(pcaDT, params), pcaL=LT, model_name='LogReg')

    # QuadLogReg
    # trainer = get_quadLogReg_trainer(1e-4, 0.5)
    # params = trainer(raw, L)
    # print_model_performance(LT, quad_log_reg_scorer(DT, params), model_name='QuadLogReg')
    # params = trainer(zD, L)
    # print_model_performance(LT, zs=quad_log_reg_scorer(zDT, params), zL=LT, model_name='QuadLogReg')
    # params = trainer(pca_d, L)
    # print_model_performance(LT, pcas=quad_log_reg_scorer(pcaDT, params), pcaL=LT, model_name='QuadLogReg')

    # LinSVM
    # lin_svm_tr, lin_svm_scorer = get_svm_trainer_model(1)
    # params = lin_svm_tr(zD, L)
    # s = lin_svm_scorer(zDT, params)
    # print_model_performance(LT, zs=s, zL=LT, model_name='LinSVM')

    # QuadSVM
    # kernel_fn = get_poly_kernel(2, 0.2, 0)
    # quad_svm_tr, quad_svm_scorer = get_svm_trainer_model(1.5, kernel_fn=kernel_fn)
    # params = quad_svm_tr(zD,L)
    # svm_sc = quad_svm_scorer(zDT, params)
    # print_model_performance(L, zs=svm_sc, zL=LT, model_name='QuadSVM')

    # RBF SVM
    # kernel_fn = get_rbf_kernel(0.01, 1)
    # rbf_svm_tr, rbf_svm_scorer = get_svm_trainer_model(30, kernel_fn=kernel_fn)
    # params = rbf_svm_tr(zD, L)
    # svm_sc = rbf_svm_scorer(zDT, params)
    # print_model_performance(L, zs=svm_sc, zL=LT, model_name='RbfSVM')

    # Calibration of svm
    # Dcal, Lcal = k_fold_score(zD, L, 5, rbf_svm_tr, rbf_svm_scorer)
    # cal_params = evaluate_calibration_param(Dcal, Lcal)
    # svm_s_cal = compute_score_calibration(svm_sc, cal_params)
    # bayes_error_plot(svm_sc, LT, '-uncalibrated')
    # bayes_error_plot(svm_s_cal, LT, '-calibrated')
    # plt.savefig('diagrams/error_plot_score_calibration.jpg')
    # plt.show()

    # GMM
    # gmm_trainer = get_gmm_tied_cov_trainer(maxComponents=8, singleGmm=True)
    # params = gmm_trainer(D, L)
    # gmm_S = score_gmm(DT, params)
    # print_model_performance(LT, gmm_S, model_name='GMM G=8')
    #
    # bayes_error_plot(s_logReg, LT, '-final_LogReg')
    # bayes_error_plot(svm_s_cal, LT, '-final_RbfSvm')
    # bayes_error_plot(gmm_S, LT, '-final_GMM')
    # plt.savefig('diagrams/all_dcf.jpg')
    # plt.show()
    # print_model_performance(LT, s_logReg, model_name='LogReg', actual=True)
    # print_model_performance(LT, svm_s_cal, model_name='rbf svm', actual=True)
    # print_model_performance(LT, gmm_S, model_name='Gmm', actual=True)
    print("end")
