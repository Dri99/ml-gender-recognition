# Full Cov

### Raw
minDcf (pi=0.5): 0.048000
minDcf (pi=0.1): 0.125333
minDcf (pi=0.9): 0.124000

### Z
minDcf (pi=0.5): 0.048000
minDcf (pi=0.1): 0.125333
minDcf (pi=0.9): 0.124000

### Gauss
minDcf (pi=0.5): 0.059667
minDcf (pi=0.1): 0.187667
minDcf (pi=0.9): 0.174667

### PCA
minDcf (pi=0.5): 0.094000
minDcf (pi=0.1): 0.264667
minDcf (pi=0.9): 0.221333
# Tied

### Raw
minDcf (pi=0.5): 0.047333
minDcf (pi=0.1): 0.121667
minDcf (pi=0.9): 0.125333

### Z
minDcf (pi=0.5): 0.047333
minDcf (pi=0.1): 0.121667
minDcf (pi=0.9): 0.125333

### Gauss
minDcf (pi=0.5): 0.060667
minDcf (pi=0.1): 0.180000
minDcf (pi=0.9): 0.166333

### PCA
minDcf (pi=0.5): 0.094667
minDcf (pi=0.1): 0.263667
minDcf (pi=0.9): 0.222000
# Diag

### Raw
minDcf (pi=0.5): 0.564000
minDcf (pi=0.1): 0.827333
minDcf (pi=0.9): 0.848333

### Z
minDcf (pi=0.5): 0.564000
minDcf (pi=0.1): 0.827333
minDcf (pi=0.9): 0.848333

### Gauss
minDcf (pi=0.5): 0.541333
minDcf (pi=0.1): 0.810000
minDcf (pi=0.9): 0.825000

### PCA
minDcf (pi=0.5): 0.103667
minDcf (pi=0.1): 0.266333
minDcf (pi=0.9): 0.243333
# Diag-Tied

### Raw
minDcf (pi=0.5): 0.567000
minDcf (pi=0.1): 0.824333
minDcf (pi=0.9): 0.848667

### Z
minDcf (pi=0.5): 0.567000
minDcf (pi=0.1): 0.824333
minDcf (pi=0.9): 0.848667

### Gauss
minDcf (pi=0.5): 0.540333
minDcf (pi=0.1): 0.805667
minDcf (pi=0.9): 0.816333

### PCA
minDcf (pi=0.5): 0.104000
minDcf (pi=0.1): 0.269667
minDcf (pi=0.9): 0.234667

## linear logreg lambda = 1e-4

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

Gaussian
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.5	 0.055333
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.1	 0.162333
LogReg lambda: 10e-4  pi_tr: 0.500000 pi_app: 0.9	 0.160667

## lambda = 1e-5
LogReg lambda: 10e-5  pi_tr: 0.500000 pi_app: 0.5	 0.047333
LogReg lambda: 10e-5  pi_tr: 0.500000 pi_app: 0.1	 0.127000
LogReg lambda: 10e-5  pi_tr: 0.500000 pi_app: 0.9	 0.125667

## quadratic logreg
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


## Linear svm 
C: 1
z
minDcf (pi=0.5): 0.046333 
minDcf (pi=0.1): 0.128667 
minDcf (pi=0.9): 0.129000 

## Quadratic svm 
c=0.2 eps=0 C=1.5
### Z
minDcf (pi=0.5): 0.053000
minDcf (pi=0.1): 0.148667
minDcf (pi=0.9): 0.144333


# RBF SMV
### Z
Rbf svm C = 30 gamma=0.01
K = 1 not optimised
minDcf (pi=0.5): 0.045000 

## GMM
GMM G = 8


# Experimental result
## Full cov testing

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
## Tied cov testing

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
## Diag cov testing

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
## Diag-tied testing

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


# LogReg lamda = 1e-4
Raw
minDcf (pi=0.5): 0.052500
minDcf (pi=0.1): 0.135000
minDcf (pi=0.9): 0.133000

Z
minDcf (pi=0.5): 0.050500
minDcf (pi=0.1): 0.144000
minDcf (pi=0.9): 0.135000

PCA
minDcf (pi=0.5): 0.111000
minDcf (pi=0.1): 0.278000
minDcf (pi=0.9): 0.260500

# logreg lambda = 1e-5

# QuadLogReg

### Raw
minDcf (pi=0.5): 0.108500
minDcf (pi=0.1): 0.333000
minDcf (pi=0.9): 0.273000

### Z
minDcf (pi=0.5): 0.051500
minDcf (pi=0.1): 0.144500
minDcf (pi=0.9): 0.138000


### PCA
minDcf (pi=0.5): 0.109500
minDcf (pi=0.1): 0.252500
minDcf (pi=0.9): 0.253500

# LinSVM
### Z
minDcf (pi=0.5): 0.051000
minDcf (pi=0.1): 0.141000
minDcf (pi=0.9): 0.131000

# QuadSVM
### Z
minDcf (pi=0.5): 0.056500
minDcf (pi=0.1): 0.130500
minDcf (pi=0.9): 0.150000

# RbfSVM
### Z
minDcf (pi=0.5): 0.052000
minDcf (pi=0.1): 0.127000
minDcf (pi=0.9): 0.130000

# GMM G=8
### Raw
minDcf (pi=0.5): 0.030500
minDcf (pi=0.1): 0.087000
minDcf (pi=0.9): 0.085000

# Final not calib
# LogReg
minDcf (pi=0.5): 0.054000
minDcf (pi=0.1): 0.162500
minDcf (pi=0.9): 0.166500
# rbf svm
minDcf (pi=0.5): 0.054500
minDcf (pi=0.1): 0.291000
minDcf (pi=0.9): 0.339500
# Gmm
minDcf (pi=0.5): 0.031500
minDcf (pi=0.1): 0.096000
minDcf (pi=0.9): 0.091500

# Final calibrated 
# LogReg
minDcf (pi=0.5): 0.054000
minDcf (pi=0.1): 0.162500
minDcf (pi=0.9): 0.166500
# rbf svm
minDcf (pi=0.5): 0.053000
minDcf (pi=0.1): 0.148000
minDcf (pi=0.9): 0.137500
# Gmm
minDcf (pi=0.5): 0.031500
minDcf (pi=0.1): 0.096000
minDcf (pi=0.9): 0.091500