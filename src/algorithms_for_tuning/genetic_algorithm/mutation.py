import random
import numpy as np


def mutation_one_param(individ, high_decor=1e5,
                       high_n=8, high_spb=1e2,
                       low_spm=-1e2, elem_mutation_prob=0.1
                       ):
    for i in range(len(individ)):
        if random.random() <= elem_mutation_prob:
            if i in [0]:
                individ[i] = np.random.uniform(low=0, high=high_decor, size=1)[0]
            if i in [15]:
                individ[i] = np.random.uniform(low=0, high=1, size=1)[0]
            if i in [1, 4, 7, 10, 11]:
                individ[i] = np.random.randint(low=0, high=high_n, size=1)[0]
            if i in [2, 3]:
                individ[i] = np.random.uniform(low=1e-3, high=high_spb, size=1)[0]
            if i in [5, 6, 8, 9]:
                individ[i] = np.random.uniform(low=low_spm, high=-1e-3, size=1)[0]
    return individ


def positioning_mutation(individ, elem_mutation_prob=0.1):
    for i in range(len(individ)):
        set_1 = set([i])
        if random.random() <= elem_mutation_prob:
            if i in [0, 15]:
                set_2 = set([0, 15])
                ix = np.random.choice(list(set_2.difference(set_1)))
                tmp = individ[ix]
                individ[ix] = individ[i]
                individ[i] = tmp
            elif i in [1, 4, 7, 10, 11]:
                set_2 = set([1, 4, 7, 10, 11])
                ix = np.random.choice(list(set_2.difference(set_1)))
                tmp = individ[ix]
                individ[ix] = individ[i]
                individ[i] = tmp
            elif i in [2, 3]:
                set_2 = set([2, 3])
                ix = np.random.choice(list(set_2.difference(set_1)))
                tmp = individ[ix]
                individ[ix] = individ[i]
                individ[i] = tmp
            elif i in [5, 6, 8, 9]:
                set_2 = set([5, 6, 8, 9])
                ix = np.random.choice(list(set_2.difference(set_1)))
                tmp = individ[ix]
                individ[ix] = individ[i]
                individ[i] = tmp
        return individ


def mutation_combined(individ, elem_mutation_prob=0.1):
    if random.random() <= individ[14]:  # TODO: check 14th position
        return mutation_one_param(individ, elem_mutation_prob)
    else:
        return positioning_mutation(individ, elem_mutation_prob)
    pass


def mutation_psm(individ, elem_mutation_prob):
    for i in range(len(individ)):
        if random.random() < elem_mutation_prob:
            if i == 0:
                individ[i] = individ[15]
            elif i == 15:
                individ[i] = individ[0]
            elif i in [1, 4, 7, 10]:
                j = np.random.choice([1, 4, 7, 10])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
            elif i in [2, 3]:
                j = np.random.choice([2, 3])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
            # elif i in [12, 13, 14]:
            #     j = np.random.choice([12, 13, 14])
            #     tmp = individ[i]
            #     individ[i] = individ[j]
            #     individ[j] = tmp
            elif i in [5, 6, 8, 9]:
                j = np.random.choice([5, 6, 8, 9])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
    return individ


def mutation(mutation_type='mutation_one_param'):
    if mutation_type == 'mutation_one_param':
        return mutation_one_param
    if mutation_type == 'combined':
        return mutation_combined
    if mutation_type == 'psm':
        return mutation_psm
    if mutation_type == "positioning_mutation":
        return positioning_mutation
