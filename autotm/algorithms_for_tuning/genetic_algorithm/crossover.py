import random
import numpy as np
from typing import List, Tuple, Callable


def crossover_pmx(parent_1: List[float], parent_2: List[float], **kwargs) -> Tuple[List[float], List[float]]:
    """
    Pmx crossover

    Exchange chromosome parts

    Parameters
    ----------
    parent_1: List[float]
        The first individual to be processed
    parent_2: List[float]
        The second individual to be processed

    Returns
    ----------
    Updated individuals with exchanged chromosome parts
    """
    points_num = len(parent_1)
    flag = False
    while not flag:
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
def crossover_one_point(parent_1: List[float], parent_2: List[float], **kwargs) -> Tuple[List[float], List[float]]:
    """
    One-point crossover

    Exchange points between chromosomes

    Parameters
    ----------
    parent_1: List[float]
        The first individual to be processed
    parent_2: List[float]
        The second individual to be processed

    Returns
    ----------
    Updated individuals with exchanged chromosome parts
    """
    elem_cross_prob = kwargs["elem_cross_prob"]
    for i in range(len(parent_1)):
        # removed mutation preservation
        if random.random() < elem_cross_prob:
            dop = parent_1[i]
            parent_1[i] = parent_2[i]
            parent_2[i] = dop
    return parent_1, parent_2


def crossover_blend_new(parent_1: List[float], parent_2: List[float], **kwargs) -> Tuple[List[float], List[float]]:
    """
    Blend crossover

    Making combination of parents solution with coefficient

    Parameters
    ----------
    parent_1: List[float]
        The first individual to be processed
    parent_2: List[float]
        The second individual to be processed
    alpha: float
        Blending coefficient

    Returns
    ----------
    Updated individuals with exchanged chromosome parts
    """
    alpha = kwargs["alpha"]
    child_1 = []
    child_2 = []
    u = random.random()
    gamma = (1.0 + 2.0 * alpha) * u - alpha  # fixed (1. + 2. * alpha) * u - alpha
    for i in range(len(parent_1)):
        child_1.append((1.0 - gamma) * parent_1[i] + gamma * parent_2[i])
        child_2.append(gamma * parent_1[i] + (1.0 - gamma) * parent_2[i])

    # TODO: reconsider this
    child_1[12:15] = parent_1[12:15]
    child_2[12:15] = parent_2[12:15]
    return child_1, child_2


def crossover_blend(parent_1: List[float], parent_2: List[float], **kwargs) -> Tuple[List[float], List[float]]:
    """
    Blend crossover

    Making combination of parents solution with coefficient

    Parameters
    ----------
    parent_1: List[float]
        The first individual to be processed
    parent_2: List[float]
        The second individual to be processed
    alpha: float
        Blending coefficient

    Returns
    ----------
    Updated individuals with exchanged chromosome parts
    """
    alpha = kwargs["alpha"]
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



def crossover(crossover_type: str = "crossover_one_point") -> Callable:

    """
    Crossover function

    Parameters
    ----------
    crossover_type : str, default="crossover_one_point"
        Crossover to be used in the genetic algorithm
    """
    if crossover_type == "crossover_pmx":
        return crossover_pmx
    if crossover_type == "crossover_one_point":
        return crossover_one_point
    if crossover_type == "blend_crossover":
        return crossover_blend
