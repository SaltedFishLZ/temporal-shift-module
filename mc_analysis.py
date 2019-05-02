import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

    # f = open("{}.shapes.pkl".format(DATASET), "rb")
    # shapes = pickle.load(f)
    # f.close()
    # (lengths, heights, widths, channels) = zip(*shapes)
    # # print(lengths)
    # plt.hist(lengths, density=True, bins=10)
    # plt.show()


def analyze_chances(filename):
    f = open(filename, "rb")
    idaccs = pickle.load(f)
    f.close()

    # [Chance][K]
    idaccs = np.array(idaccs)
    # [K][Chance]
    idaccs = np.swapaxes(idaccs, 0, 1)

    K = idaccs.shape[0]
    means = np.empty(K)
    vars = np.empty(K)
    for _k in range(K):
        means[_k] = np.mean(idaccs[_k])
        vars[_k] = np.var(idaccs[_k])

    print("MEANS : ", means[_k])
    print("VARS : ", vars[_k])

    return((means, vars))

def analyze_imporvement(filename):
    f = open(filename, "rb")
    mcaccs = pickle.load(f)
    f.close()

    # [Chance][K]
    mcaccs = np.array(mcaccs)
    # [K][Chance]
    mcaccs = np.swapaxes(mcaccs, 0, 1)
    K = mcaccs.shape[0]
    N = mcaccs.shape[1]

    chances = np.array(range(N)) + 1
    for _k in range(K):
        plt.plot(chances, mcaccs[_k], label='Top-{}'.format(_k))
    plt.legend(loc='best')
    plt.xlabel('Chances')
    plt.ylabel('Accuracy')    
    plt.show()

if __name__ == "__main__":

    config = "TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e25"
    
    idacc_name = config + ".idacc"
    analyze_chances(idacc_name)
      
    mcacc_name = config + ".mcacc"
    analyze_imporvement(mcacc_name)

    
