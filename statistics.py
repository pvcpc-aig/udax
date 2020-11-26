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
