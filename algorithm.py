"""
Implementations of miscellaneous algorithms that may or may not be
grouped into a different module for organization later.
"""
from collections.abc import Sequence


def wlcsubsequence(A, B, weight_f=lambda x: x * x, traceback=True):
    """
    Computes the weighted longest common, discontiguous subsequence in both
    sequences `A`, `B`. The elements in `A` and `B` must be comparable
    with these comparison functions: 
    https://docs.python.org/3/reference/datamodel.html#object.__lt__

    :param A
        The first sequence of elements.
    
    :param B
        The second sequence of elements.
    
    :param weight_f = lambda x: x * x
        A mathematically increasing, and preferrably concave up, function
        used to compute the score increment for consecutive matches in 
        the two sequences. The default value is a quadratic function, but
        an identity function may be used to simulate the more standard
        longest common subsequence algorithm. The specialization is
        aliased by `lcssubsequence()` below.

    :param traceback = True
        Whether to gather a list of all longest subsequences in the
        resulting computation.
    
    :return
        if traceback:
            (<list:subsequences>, <int:longest_size>)
        else:
            <int:longest_size>
    """
    # verification
    if not isinstance(A, Sequence) or not isinstance(B, Sequence):
        raise ValueError("Sequences A, B must be of Sequence type")
    if not callable(weight_f):
        raise ValueError("weight_f must be a mathematically concave up callable function")

    La = len(A)
    Lb = len(B)

    # tables: W will hold traces for longest sub**strings**,
    # C is the weighted LCS table.
    W = [ [ 0 for _ in range(Lb + 1) ] for _ in range(La + 1) ]
    C = [ [ 0 for _ in range(Lb + 1) ] for _ in range(La + 1) ]

    for i, a in enumerate(A):
        Ti = i + 1
        for j, b in enumerate(B):
            Tj = j + 1
            if a == b:
                k = W[Ti-1][Tj-1]
                W[Ti][Tj] = k + 1
                C[Ti][Tj] = C[Ti-1][Tj-1] + weight_f(k + 1) - weight_f(k)
            else:
                C[Ti][Tj] = max(
                    C[Ti-1][Tj],
                    C[Ti][Tj-1])
                W[Ti][Tj] = 0

    score = C[La][Lb]

    def _traceback(Ti, Tj):
        nonlocal C, A, B
        i, j = Ti-1, Tj-1

        # if empty of visited, return emptiness
        if C[Ti][Tj] == 0:
            return []
        
        top_left, top, left, current = (
            C[Ti-1][Tj-1],
            C[Ti-1][Tj],
            C[Ti][Tj-1],
            C[Ti][Tj])
        
        # mark visited
        C[Ti][Tj] = 0
        if A[i] == B[j]:
            elem = A[i]
            prior = _traceback(Ti-1, Tj-1)
            if len(prior) == 0:
                return [ [elem] ]
            else:
                return [ [ *x, elem ] for x in prior ]
        elif top > left:
            return _traceback(Ti-1, Tj)
        elif left > top:
            return _traceback(Ti, Tj-1)
        else:
            top_branch = _traceback(Ti-1, Tj)
            left_branch = _traceback(Ti, Tj-1)
            return [ *top_branch, *left_branch ]
    
    if traceback:
        return _traceback(La, Lb), score
    return score


def lcsubsequence(A, B, traceback=True):
    """
    Computes the longest common, discontiguous subsequence in both
    sequences `A`, `B`. The elements in `A` and `B` must be comparable
    with these comparison functions: 
    https://docs.python.org/3/reference/datamodel.html#object.__lt__

    This is a specialization of the `wlcsubsequence()` function, defined
    above, with the weight function set to the identity function, i.e.,
    f(x) = x.

    :param A
        The first sequence of elements.
    
    :param B
        The second sequence of elements.

    :param traceback = True
        Whether to gather a list of all longest subsequences in the
        resulting computation.
    
    :return
        if traceback:
            (<list:subsequences>, <int:longest_size>)
        else:
            <int:longest_size>
    """
    return wlcsubsequence(A, B, lambda x: x, traceback)


def lcsubstring(A, B, traceback=True):
    """
    Computes the longest common, contiguous substring in both
    sequences `A`, `B`. The elements in `A` and `B` must be comparable
    with these comparison functions: 
    https://docs.python.org/3/reference/datamodel.html#object.__lt__

    :param A
        The first sequence of elements.
    
    :param B
        The second sequence of elements.

    :param traceback = True
        Whether to gather a list of all longest substrings in the
        resulting computation.
    
    :return
        if traceback:
            (<list:substrings>, <int:longest_size>)
        else:
            <int:longest_size>
    """
    # verification
    if not isinstance(A, Sequence) or not isinstance(B, Sequence):
        raise ValueError("Sequences A, B must be of Sequence type")

    La = len(A)
    Lb = len(B)

    # maxlen, table & stack
    maxlen = 0
    W = [ [ 0 for _ in range(Lb + 1) ] for _ in range(La + 1) ]
    S = []

    for i, a in enumerate(A):
        Ti = i + 1
        for j, b in enumerate(B):
            Tj = j + 1
            if a == b:
                k = W[Ti-1][Tj-1]
                W[Ti][Tj] = k + 1
                if k + 1 >= maxlen:
                    S.append((Ti, Tj))
                    maxlen = k + 1

    def _traceback(Ti, Tj):
        nonlocal W, S, A
        result = []
        while W[Ti][Tj] > 0:
            result.insert(0, A[Ti-1])
            Ti -= 1
            Tj -= 1
        return result
    
    if traceback:
        traceback = []
        for pos in reversed(S):
            Ti, Tj = pos
            if W[Ti][Tj] < maxlen:
                break
            traceback.append(_traceback(Ti, Tj))
        return traceback, maxlen
    return maxlen


def comb(A, n, i=0):
    """
    An iterating function that yields a tuple of size `n` for each
    combination of elements in sequence `A`. The combinations are 
    guaranteed to be stable, i.e., each element in every tuple will
    be arranged in their apparent order.

    :param A
        The sequence whose elements to traverse as combinations.
    
    :param n
        The size of each combination tuple.

    :param i
        The starting point in the sequence.
    
    :yield
        The next combination tuple.
    """
    # verification
    if not isinstance(A, Sequence):
        raise ValueError("Sequence A must be of Sequence type")

    if n < 1:
        return
    elif n == 1:
        while i < len(A):
            yield (A[i],)
            i += 1
    else:
        while i <= len(A) - n:
            for c in comb(A, n-1, i+1):
                yield (A[i], *c)
            i += 1