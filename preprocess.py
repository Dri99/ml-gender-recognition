import time

import numpy
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import rankdata, norm


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                # name = line.split(',')[-1].strip()
                label = int(line.split(',')[-1])
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def plot_data(data, label, name):
    M=data.shape[0]
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    Df = data[:, label == 0]
    Dt = data[:, label == 1]

    hFea = {
        0: 'Feature 0',
        1: 'Feature 1',
        2: 'Feature 2',
        3: 'Feature 3',
        4: 'Feature 4',
        5: 'Feature 5',
        6: 'Feature 6',
        7: 'Feature 7',
        8: 'Feature 8',
        9: 'Feature 9',
        10: 'Feature 10',
        11: 'Feature 11',
    }

    for dIdx in range(M):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(Df[dIdx, :], bins=20, density=True, alpha=0.4, label='Male')
        plt.hist(Dt[dIdx, :], bins=20, density=True, alpha=0.4, label='Female')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(name + ('hist_%d.jpg' % dIdx))
    plt.show()


def get_z_params(train_data, ):
    data = train_data - mcol(train_data.mean(1))
    dev = numpy.sqrt(mcol(data.var(1)))
    return train_data.mean(1), dev


def apply_z_center(data, params):
    mu, dev = params
    return (data - mcol(mu)) / mcol(dev)


def z_center(data):
    data = data - mcol(data.mean(1))
    var = mcol(data.var(1))
    return data / var


def gaussianify_old(data):
    return numpy.array([norm.ppf((rankdata(x_i) + 1) / (len(x_i) + 2)) for x_i in data])


def gaussianify(train_data, test_data):
    M = test_data.shape[0]
    N = test_data.shape[1]
    mega_matrix = []
    for sample in test_data.T:
        hdata = numpy.hstack((train_data, mcol(sample)))
        mega_matrix.append(hdata)

    mega_matrix = numpy.vstack(mega_matrix)

    r = rankdata(mega_matrix, axis=1) + 1
    test = r[:, -1]
    test_ranks = test.reshape((N, M)).T / (N + 2)
    return norm.ppf(test_ranks)


def gaussianify_slow(train_data, test_data):
    N = test_data.shape[1]
    test_ranks = []
    for sample in test_data.T:
        test_ranks.append(rank_fn_fast(train_data, sample))

    test_ranks = numpy.array(test_ranks).T / (N + 2)
    return norm.ppf(test_ranks)


def gaussianify3(train_data, test_data):
    M = test_data.shape[0]
    N = test_data.shape[1]
    mega_train_data = numpy.vstack([train_data] * N)
    test_col = test_data.T.reshape((M * N, 1))
    ranks = mega_train_data < test_col
    ranks = (ranks.sum(axis=1) + 1)  # MNx1
    ranks = ranks.reshape((N, M))  # NxM
    ranks = ranks.T  # MxN
    test_ranks = ranks / (N + 2)
    return norm.ppf(test_ranks)


def rank_fn(train_data, sample):
    hdata = numpy.hstack((train_data, mcol(sample)))
    r = rankdata(hdata, axis=0) + 1
    return r[:, -1]


def rank_fn_fast(train_data, sample):
    sample = mcol(sample)
    rank = numpy.array(train_data < sample)
    rank = rank.sum(axis=1) + 1
    return rank


def corr_heatmaps(data, label, name):
    C_full = numpy.abs(numpy.corrcoef(data))
    C_0 = numpy.abs(numpy.corrcoef(data[:, label == 0]))
    C_1 = numpy.abs(numpy.corrcoef(data[:, label == 1]))
    second = sb.color_palette("flare", as_cmap=True)
    third = sb.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    plt.title("Full dataset")
    sb.heatmap(C_full, cmap="YlGnBu", )
    plt.tight_layout()
    plt.savefig('diagrams/heat_map_full_' + name+ '.jpg')
    plt.show()
    plt.title("Male")
    sb.heatmap(C_0, cmap=second, )
    plt.tight_layout()
    plt.savefig('diagrams/heat_map_0_' + name + '.jpg')
    plt.show()
    plt.title("Female")
    sb.heatmap(C_1, cmap=third, )
    plt.tight_layout()
    plt.savefig('diagrams/heat_map_1_' + name + '.jpg')
    plt.show()
