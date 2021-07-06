import random
import numpy as np


def crossover_pmx(parent_1, parent_2, **kwargs):
    points_num = len(parent_1)
    flag = False
    while flag == False:
        cut_ix = np.random.choice(points_num + 1, 2, replace=False)
        min_ix = np.min(cut_ix)
        max_ix = np.max(cut_ix)
        part = parent_1[min_ix:max_ix]
        if len(part) != len(parent_1):
            flag = True
    parent_1[min_ix:max_ix] = parent_2[min_ix:max_ix]
    parent_2[min_ix:max_ix] = part
    return parent_1, parent_2


# discrete crossover
def crossover_one_point(parent_1, parent_2, **kwargs):
    elem_cross_prob = kwargs['elem_cross_prob']
    for i in range(len(parent_1)):
        if i not in [12, 13, 14]:
            if random.random() < elem_cross_prob:
                dop = parent_1[i]
                parent_1[i] = parent_2[i]
                parent_2[i] = dop
    return parent_1, parent_2


# returns one child (!)
def crossover_blend(parent_1, parent_2, **kwargs):
    alpha = kwargs['alpha']
    child = []
    u = random.random()
    gamma = (1 - 2 * alpha) * u - alpha
    for i in range(len(parent_1)):
        child.append((1 - gamma) * parent_1[i] + gamma * parent_2[i])
    if random.random() > 0.5:
        child[12:15] = parent_1[12:15]
    else:
        child[12:15] = parent_2[12:15]
    return child


def crossover(crossover_type='crossover_one_point'):
    if crossover_type == 'crossover_pmx':
        return crossover_pmx
    if crossover_type == 'crossover_one_point':
        return crossover_one_point
    if crossover_type == 'blend_crossover':
        return crossover_blend
