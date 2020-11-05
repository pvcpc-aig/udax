"""
An implementation of the Recall Oriented Understudy for Gisting
Evaluation. The original paper by Chin-Yew Lin is found at
https://www.aclweb.org/anthology/W04-1013.pdf
"""
import io
from math import factorial
from collections.abc import Iterable
from nltk.tokenize import word_tokenize


# --- Utilities --------------------------------------------------------
def _iterate_combinations(seq, start=0, size=1):
    i = start
    if size == 1:
        while i < len(seq):
            yield (seq[i],)
            i += 1
    else:
        while i <= len(seq) - size:
            item = seq[i]
            for c in _iterate_combinations(seq, start=i+1, size=size-1):
                yield (item, *c)
            i += 1


def _iterate_grams(seq, start=0, size=1, skip=1):
    i = start
    while i <= len(seq) - size:
        yield tuple(seq[i:i+size])
        i += skip


def _P(n, r):
    result = 1
    while n > r:
        result *= n
        n -= 1
    return result


def _C(n, r):
    d = n - r
    if r > d:
        return _P(n, r) // factorial(d)
    else:
        return _P(n, d) // factorial(r)


# --- Interface --------------------------------------------------------
def rouge_n(ref, candidate, tokenizer, n=1):
    if type(ref) is not str or type(candidate) is not str:
        raise ValueError("ref and candidate must be strings")

    if not callable(tokenizer):
        raise ValueError("tokenizer must be a callable object")

    if n < 1:
        raise ValueError("n must be >= 1")

    r_tokens = tokenizer(ref)
    c_tokens = tokenizer(candidate)

    r_size = len(r_tokens)
    c_size = len(c_tokens)
    if n > min(r_size, c_size):
        raise ValueError(f"n must be <= min(r_size = {r_size}, c_size = {c_size})")

    # build the collection of reference n-grams
    r_total = 1 + len(r_tokens) - n
    r_grams = {}
    for gram in _iterate_grams(r_tokens, size=n):
        if gram not in r_grams:
            r_grams[gram] = 1
        else:
            r_grams[gram] += 1
    
    # iterate and compare with the candidate summary
    c_total = 1 + len(c_tokens) - n
    c_matched = 0
    for gram in _iterate_grams(c_tokens, size=n):
        if gram in r_grams and r_grams[gram] > 0:
            c_matched += 1
            r_grams[gram] -= 1
    
    precision = c_matched / c_total
    recall = c_matched / r_total
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1



p, r, f = rouge_n(
    """
    The dog was hiding swiftly under the yellow umbreller.
    The yellow umbrellar was very large.
    """,
    """
    Dog hides under the yellow umbreller. Yellow umbreller
    was very large. The umbreller knew not.
    """,
    word_tokenize,
    n=4
)

print("precision = %.2f, recall = %.2f, f-measure = %.2f" % (p, r, f))