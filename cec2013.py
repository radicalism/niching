###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################
from scipy.spatial.distance import pdist, squareform

# from scipy.spatial import distance
import numpy as np
import math
from functions import *
from cfunction import *
from CF1 import *
from CF2 import *
from CF3 import *
from CF4 import *


class CEC2013(object):
    func_idx = -1 #func_idx
    functions = {
        1: five_uneven_peak_trap,   #name of benchmark
        2: equal_maxima,
        3: uneven_decreasing_maxima,
        4: himmelblau,
        5: six_hump_camel_back,
        6: shubert,
        7: vincent,
        8: shubert,
        9: vincent,
        10: modified_rastrigin_all,
        11: CF1,
        12: CF2,
        13: CF3,
        14: CF3,
        15: CF4,
        16: CF3,
        17: CF4,
        18: CF3,
        19: CF4,
        20: CF4,
    }
    func_name = None  #func_name
    opt = [
        200.0,
        1.0,
        1.0,
        200.0,
        1.031628453489877,
        186.7309088310239,
        1.0,
        2709.093505572820,
        1.0,
        -2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    __rho_ = [
        0.01,
        0.01,
        0.01,
        0.01,
        0.5,
        0.5,
        0.2,
        0.5,
        0.2,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]
    nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
    budget = [
        50000,
        50000,
        50000,
        50000,
        50000,
        200000,
        200000,
        400000,
        400000,
        200000,
        200000,
        200000,
        200000,
        400000,
        400000,
        400000,
        400000,
        400000,
        400000,
        400000,
    ]
    dimensions = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

    def __init__(self, idx):
        assert idx > 0 and idx <= 20
        self.func_idx = idx
        if self.func_idx > 0 and self.func_idx < 11:
            self.func_name = self.functions[self.func_idx]
        else:
            self.func_name = self.functions[self.func_idx](self.get_dimension())

    def evaluate(self, x):
        x_ = np.asarray(x)
        #print(np.asarray(x_).shape)
        #assert np.asarray(x_).shape[1] == self.get_dimension()
        if self.func_idx > 0 and self.func_idx < 11:
            return self.func_name(x_)
        else:
            return self.func_name.evaluate(np.asarray(x_))

    def get_lbound(self):
        result = None
        if self.func_idx == 1 or self.func_idx == 2 or self.func_idx == 3:
            result = [0]*self.get_dimension()
        elif self.func_idx == 4:
            result = [-6]*self.get_dimension()
        elif self.func_idx == 5:
            result = [-1.9, -1.1]
        elif self.func_idx == 6 or self.func_idx == 8:
            result = [-10]*self.get_dimension()
        elif self.func_idx == 7 or self.func_idx == 9:
            result = [0.25]*self.get_dimension()
        elif self.func_idx == 10:
            result = [0]*self.get_dimension()
        elif self.func_idx > 10:
            result = self.func_name.get_lbound()
        return result

    def get_ubound(self):
        result = None
        if self.func_idx == 1:
            result = [30]*self.get_dimension()
        elif self.func_idx == 2 or self.func_idx == 3:
            result = [1]*self.get_dimension()
        elif self.func_idx == 4:
            result = [6]*self.get_dimension()
        elif self.func_idx == 5:
            result = [1.9, 1.1]
        elif self.func_idx == 6 or self.func_idx == 8:
            result = [10]*self.get_dimension()
        elif self.func_idx == 7 or self.func_idx == 9:
            result = [10]*self.get_dimension()
        elif self.func_idx == 10:
            result = [1]*self.get_dimension()
        elif self.func_idx > 10:
            result = self.func_name.get_ubound()
        return result

    def get_fitness_goptima(self):
        return self.opt[self.func_idx - 1]

    def get_dimension(self):
        return self.dimensions[self.func_idx - 1]

    def get_no_goptima(self):
        return self.nopt[self.func_idx - 1]

    def get_rho(self):
        return self.__rho_[self.func_idx - 1]

    def get_maxfes(self):
        return self.budget[self.func_idx - 1]

    def get_info(self):
        return {
            "fbest": self.get_fitness_goptima(),
            "dimension": self.get_dimension(),
            "nogoptima": self.get_no_goptima(),
            "maxfes": self.get_maxfes(),
            "rho": self.get_rho(),
        }


def how_many_goptima(pop, spopfits,f:CEC2013, accuracy):
    # pop: NP, D
    #print(f'pop[0].type={type(pop[0])}\npop[0].shape={pop[0].shape}')


    # find seeds in the temp population (indices!)
    seeds_idx = find_seeds_indices(pop, f.get_rho()) #rho： 判断两个个体是否处于同一niche的半径 返回的是sorted_pop中unique的解在sorted_pop中的下标集合

    count = 0
    goidx = []
    for idx in seeds_idx:
        # evaluate seed
        seed_fitness = spopfits[idx]  # f.evaluate(sorted_pop[idx])

        # |F_seed - F_goptimum| <= accuracy
        if math.fabs(seed_fitness - f.get_fitness_goptima()) <= accuracy:
            count = count + 1
            goidx.append(idx)

        # save time
        if count == f.get_no_goptima():
            break

    # gather seeds
    seeds = pop[goidx]

    return count, seeds


def find_seeds_indices(sorted_pop, radius): #返回pop中unique个体的下标的集合
    seeds = []
    seeds_idx = []
    # Determine the species seeds: iterate through sorted population
    for i, x in enumerate(sorted_pop):
        found = False
        # Iterate seeds
        for j, sx in enumerate(seeds):
            # Calculate distance from seeds
            dist = math.sqrt(sum((x - sx) ** 2))

            # If the Euclidean distance is less than the radius
            if dist <= radius:
                found = True
                break
        if not found:
            seeds.append(x)
            seeds_idx.append(int(i))

    return seeds_idx
