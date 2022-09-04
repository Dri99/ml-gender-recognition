% Full Cov
Raw
0.048
0.12533333333333332
Z
0.048
0.12533333333333332
Gauss
0.05966666666666666
0.18766666666666668
PCA
0.094
0.18766666666666668
% Tied
Raw
0.04733333333333333
0.12166666666666667
Z
0.04733333333333333
0.12166666666666667
Gauss
0.06066666666666667
0.18000000000000002
PCA
0.09466666666666666
0.18000000000000002
% Diag
Raw
0.5640000000000001
0.8273333333333333
Z
0.5640000000000001
0.8273333333333333
Gauss
0.5413333333333333
0.8100000000000002
PCA
0.10366666666666666
0.8100000000000002
% Diag-Tied
Raw
0.567
0.8243333333333334
Z
0.567
0.8243333333333334
Gauss
0.5403333333333333
0.8056666666666668
PCA
0.10400000000000001
0.8056666666666668

linear logreg
raw
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.5	 0.047333
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.1	 0.127000
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.9	 0.125667
Z normalised
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.5	 0.046000
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.1	 0.136333
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.9	 0.129667
PCA
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.5	 0.095667
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.1	 0.266333
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.9	 0.218667



quadratic logreg
raw
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.5	 0.120667
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.1	 0.323667
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.9	 0.333333
Z normalised
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.5	 0.052333
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.1	 0.150667
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.9	 0.139000
pca
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.5	 0.092667
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.1	 0.256000
LogReg lambda: 0.000100  pi_tr: 0.500000 pi_app: 0.9	 0.224667


Linear svm C: 1 
minDcf (pi=0.5): 0.046333 
minDcf (pi=0.1): 0.128667 
minDcf (pi=0.9): 0.129000 
Quadratic svm:

Rbf svm C = 30 gamma=0.01
K = 1 not optimised
minDcf (pi=0.5): 0.045000 

GMM G = 8

# Experimental result
Full cov testing
Raw
minDcf (pi=0.5): 0.053000
minDcf (pi=0.1): 0.134500
minDcf (pi=0.9): 0.137500
Z
minDcf (pi=0.5): 0.053000
minDcf (pi=0.1): 0.134500
minDcf (pi=0.9): 0.137500
Gauss
minDcf (pi=0.5): 0.984000
minDcf (pi=0.1): 0.997000
minDcf (pi=0.9): 2.000000
PCA
minDcf (pi=0.5): 0.113000
minDcf (pi=0.1): 0.279500
minDcf (pi=0.9): 0.255000
Tied cov testing
Raw
minDcf (pi=0.5): 0.050500
minDcf (pi=0.1): 0.132500
minDcf (pi=0.9): 0.135000
Z
minDcf (pi=0.5): 0.050500
minDcf (pi=0.1): 0.132500
minDcf (pi=0.9): 0.135000
Gauss
minDcf (pi=0.5): 0.983500
minDcf (pi=0.1): 0.993000
minDcf (pi=0.9): 2.000000
PCA
minDcf (pi=0.5): 0.111500
minDcf (pi=0.1): 0.278000
minDcf (pi=0.9): 0.255500
Diag cov testing
Raw
minDcf (pi=0.5): 0.570000
minDcf (pi=0.1): 0.810000
minDcf (pi=0.9): 0.882000
Z
minDcf (pi=0.5): 0.570000
minDcf (pi=0.1): 0.810000
minDcf (pi=0.9): 0.882000
Gauss
minDcf (pi=0.5): 0.992500
minDcf (pi=0.1): 0.999000
minDcf (pi=0.9): 2.000000
PCA
minDcf (pi=0.5): 0.116000
minDcf (pi=0.1): 0.293500
minDcf (pi=0.9): 0.262500
Diag-tied testing
Raw
minDcf (pi=0.5): 0.570000
minDcf (pi=0.1): 0.808500
minDcf (pi=0.9): 0.879500
Z
minDcf (pi=0.5): 0.570000
minDcf (pi=0.1): 0.808500
minDcf (pi=0.9): 0.879500
Gauss
minDcf (pi=0.5): 0.992500
minDcf (pi=0.1): 0.999000
minDcf (pi=0.9): 2.000000
PCA
minDcf (pi=0.5): 0.115500
minDcf (pi=0.1): 0.285000
minDcf (pi=0.9): 0.266500
