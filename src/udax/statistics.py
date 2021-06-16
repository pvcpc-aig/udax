"""
Several math functions used in probability and statistical computations.
The module name was chosen to separate it from the python standard math
module.
"""
from math import factorial


def fact(N):
    """
    A shortened alias for Python's `math.factorial` for the sake of
    completion.

    :param N
        The number whose factorial to compute.
    
    :return
        <int:factorial>
    """
    return factorial(N)


def perm(N, R):
    """
    The permutation function P(N,R) = N!/(N-R)!

    :param N
        Total elements.
    
    :param R
        Number to choose.
    
    :return
        <int:permutations>
    """
    result = 1
    while N > R:
        result *= N
        N -= 1
    return result


def comb(N, R):
    """
    The combination function C(N,R) = N!/[(N-R)!R!]

    :param N
        Total elements.
    
    :param R
        Number to choose.
    
    :return
        <int:combinations>
    """
    result = 1
    Min = R
    Max = N - R
    if R > (N >> 1):
        Min = N - R
        Max = R
    return perm(N, Max) // fact(Min)


def f_score(recall, precision, beta=1):
    """
    Computes the F-score given the recall, precision,
    and an optional beta. The default beta will compute
    the harmonic mean of the recall and precision without
    bias toward either.

    :param recall
        The recall ratio, [0, 1]
    
    :param precision
        The precision ratio, [0, 1]
    
    :param beta
        The weight of precision with respect to recall.
    
    :return
        <float:f-score>
    """
    if precision == 0 or recall == 0:
        return 0
    beta_2 = beta * beta
    return (1 + beta_2) * precision * recall / (beta_2 * precision + recall)
