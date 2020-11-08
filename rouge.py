"""
An implementation of the Recall Oriented Understudy for Gisting
Evaluation. The original paper by Chin-Yew Lin is found at
https://www.aclweb.org/anthology/W04-1013.pdf
"""
from io import IOBase
from math import factorial
from pathlib import PurePath, Path
from collections.abc import Iterable, Sequence

from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# --- Utilities: General -----------------------------------------------
def _lcs(seq_a, seq_b):
    # TODO: implement O(n^2) with dynprog instead of the O(n^3) solution now
    # TODO: the paper uses LCS that allows sub-sequences to not be 
    # contiguous :mad_emoji:; fix that
    #   WAIT- the paper uses this algorithm (but with dynprog) for WLCS :thinking:
    """
    :return
        (<int:a-index>, <int:b-index>, <int:length>)
    """
    if not isinstance(seq_a, Sequence) or \
       not isinstance(seq_b, Sequence):
        return None

    sz_a = len(seq_a)
    sz_b = len(seq_b)

    a_index = 0
    b_index = 0
    length = 0

    i = 0
    while i < sz_a:
        j = 0
        while j < sz_b:
            k = 0
            m = min(sz_a - i, sz_b - j)
            while k < m and seq_a[i + k] == seq_b[j + k]:
                k += 1
            if k > length:
                a_index = i
                b_index = j
                length = k
            j += 1
        i += 1
    return a_index, b_index, length


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


# --- Utilities: Loading & Pre-processing ------------------------------
def _get_string_content(named_sources, include_name_list=True):
    """
    :return
        (<bool:success>, <str:report>, <list:result>)
    """
    names = []
    loaded = []
    for name, source in named_sources.items():
        try:
            if isinstance(source, str):
                loaded.append(source)
            elif isinstance(source, PurePath):
                loaded.append(source.open(mode="r").read())
            elif isinstance(source, IOBase):
                loaded.append(source.read())
            elif isinstance(source, Iterable):
                loaded.append(''.join(source))
            else:
                return False, f"unknown source '{name}' type {type(source)}", None
            names.append(str(name))
        except TypeError as e: # string join failed
            return False, f"iterable source '{name}' is not homogeneous", None
        except IOError as e: # reading a stream source
            return False, f"stream/file source '{name}' is not homogeneous", None
    if include_name_list:
        return True, None, names, loaded
    return True, None, loaded
        

def _rouge_prepare(named_documents, tokenizer=word_tokenize, preprocs=None, include_raw_content=False):
    # validate params
    if not callable(tokenizer):
        raise ValueError("a callable tokenizer object must be specified")

    # prepare the documents, i.e. tokenize and preprocess them
    success, report, loaded = _get_string_content(named_documents, include_name_list=False)
    if not success:
        raise ValueError(f"failed to load content: {report}")

    if preprocs is None:
        if include_raw_content:
            return [ (x, tokenizer(x)) for x in loaded ]
        else:
            return [ tokenizer(x) for x in loaded ]
    else:
        prepared = []
        for content in loaded:
            tokens = tokenizer(content)
            for preproc in preprocs:
                if not callable(preproc):
                    raise ValueError(f"pre-processor at index {i} is not a callable object")
                tokens = preproc(tokens)
            prepared.append(tokens)
        if include_raw_content:
            return loaded, prepared


# --- Interface --------------------------------------------------------
def rouge_n(ref, can, preprocs=None, tokenizer=word_tokenize, n=1):
    # TODO: implement Jackknifing option
    """
    Currently only single reference document is supported.

    Compute the precision, recall, and f-measure of the overlap
    of the candidate document with the reference documents.
    """
    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )
    
    r_size = len(r_tokens)
    c_size = len(c_tokens)
    if r_size == 0:
        raise ValueError("reference document does not have any content")
    if c_size == 0:
        raise ValueError("candidate document does not have any content")
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
    
    # compute statistics and finish
    R = c_matched / r_total
    P = c_matched / c_total
    F = 2 * (precision * recall) / (precision + recall)
    return R, P, F


def rouge_lcs_sentence(ref, can, tokenizer=word_tokenize, preprocs=None, beta=1):
    """
    Currently only single reference document is supported.

    Compute the harmonic weighted mean of the length of the 
    longest common sub-sequence of words of both documents.
    """
    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )

    lcs_result = _lcs(r_tokens, c_tokens)
    if lcs_result is None:
        raise ValueError("documents have not be tokenized into a sequence of tokens")

    _, _, length = lcs_result
    R = length / len(r_tokens)
    P = length / len(c_tokens)
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_lcs_summary(ref, can, sent_tokenizer=sent_tokenize, tokenizer=word_tokenize, preprocs=None, beta=1):
    if not callable(sent_tokenizer):
        raise ValueError("a callable sent_tokenizer object must be specified")

    r_data, c_data = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs,
        include_raw_content=True
    )
    r_content, r_tokens = r_data
    c_content, c_tokens = c_data

    # TODO: is this algorithm correct? LCS_u is a bit ambiguous in the paper
    common_tokens = set()
    common_tokens_total = 0
    for i, ref_sent in enumerate(sent_tokenizer(r_content)):
        r_sent_tokens, = _rouge_prepare(
            { f"reference_{i}": ref_sent },
            tokenizer=tokenizer,
            preprocs=preprocs
        )
        lcs_result = _lcs(r_sent_tokens, c_tokens)
        if lcs_result is None:
            raise ValueError("documents have not be tokenized into a sequence of tokens")

        r_index, c_index, length = lcs_result
        for common_token in r_sent_tokens[r_index : r_index + length]:
            common_tokens.add(common_token)
        common_tokens_total += length

    if common_tokens_total == 0:
        return None, None

    lcsu_score = len(common_tokens) / common_tokens_total
    R = lcsu_score / len(r_tokens)
    P = lcsu_score / len(c_tokens)
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_wlcs(ref, can, sent_tokenizer=sent_tokenize, tokenizer=word_tokenize, preprocs=None):
    # TODO: not implemented; review LCS/WLCS sections in the paper
    # again; apparently the current LCS implementations seem to be
    # the weighted LCS (WLCS) already, otherwise they're broken.
    pass


def rouge_s(ref, can, tokenizer=word_tokenize, preprocs=None, beta=1):
    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )

    r_size = len(r_tokens)
    c_size = len(c_tokens)
    r_skip_grams = set()
    for gram in _iterate_combinations(r_tokens):
        r_skip_grams.add(gram)
    
    matched_skip_grams = 0
    for gram in _iterate_combinations(c_tokens):
        if gram in r_skip_grams:
            matched_skip_grams += 1
    
    R = matched_skip_grams / _C(r_size, 2)
    P = matched_skip_grams / _C(c_size, 2)
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F
