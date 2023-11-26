import random
import numpy as np
from typing import List


def mutation_one_param(
        individ: List[float],
        low_spb: float,
        high_spb: float,
        low_spm: float,
        high_spm: float,
        low_n: int,
        high_n: int,
        low_back: int,
        high_back: int,
        low_decor: float,
        high_decor: float,
        elem_mutation_prob: float = 0.1,
):
    """
    One-point mutation

    Checking the probability of mutation for each of the elements

    Parameters
    ----------
    individ: List[float]
        Individual to be processed
    low_spb: float
        The lower possible bound for sparsity regularizer of back topics
    high_spb: float
        The higher possible bound for sparsity regularizer of back topics
    low_spm: float
        The lower possible bound for sparsity regularizer of specific topics
    high_spm: float
        The higher possible bound for sparsity regularizer of specific topics
    low_n: int
        The lower possible bound for amount of iterations between stages
    high_n: int
        The higher possible bound for amount of iterations between stages
    low_back:
        The lower possible bound for amount of back topics
    high_back:
        The higher possible bound for amount of back topics


    Returns
    ----------
    Updated individuals with exchanged chromosome parts
    """
    for i in range(len(individ)):
        if random.random() <= elem_mutation_prob:
            if i in [2, 3]:
                individ[i] = np.random.uniform(low=low_spb, high=high_spb, size=1)[0]
            for i in [5, 6, 8, 9]:
                individ[i] = np.random.uniform(low=low_spm, high=high_spm, size=1)[0]
            for i in [1, 4, 7, 10]:
                individ[i] = float(np.random.randint(low=low_n, high=high_n, size=1)[0])
            for i in [11]:
                individ[i] = float(
                    np.random.randint(low=low_back, high=high_back, size=1)[0]
                )
            for i in [0, 15]:
                individ[i] = np.random.uniform(low=low_decor, high=high_decor, size=1)[
                    0
                ]
    return individ


def positioning_mutation(individ, elem_mutation_prob=0.1, **kwargs):
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


def mutation_combined(individ, elem_mutation_prob=0.1, **kwargs):
    if random.random() <= individ[14]:  # TODO: check 14th position
        return mutation_one_param(individ, elem_mutation_prob)
    else:
        return positioning_mutation(individ, elem_mutation_prob)
    pass


def mutation_psm_new(individ, elem_mutation_prob, **kwargs):
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


def mutation_psm(individ, elem_mutation_prob=0.1, **kwargs):
    for i in range(len(individ)):
        if random.random() < elem_mutation_prob:
            if i == 0:
                individ[i] = np.random.uniform(low=1, high=100, size=1)[0]
            elif i in [1, 4, 7]:
                j = np.random.choice([1, 4, 7])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
            elif i in [2, 5]:
                j = np.random.choice([2, 5])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
            elif i in [3, 6]:
                j = np.random.choice([2, 5])
                tmp = individ[i]
                individ[i] = individ[j]
                individ[j] = tmp
    return individ


def mutation(mutation_type="mutation_one_param"):
    if mutation_type == "mutation_one_param":
        return mutation_one_param
    if mutation_type == "combined":
        return mutation_combined
    if mutation_type == "psm":
        return mutation_psm
    if mutation_type == "positioning_mutation":
        return positioning_mutation
