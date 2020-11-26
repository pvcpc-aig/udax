"""
An implementation of the Recall Oriented Understudy for Gisting
Evaluation. The original paper by Chin-Yew Lin is found at
https://www.aclweb.org/anthology/W04-1013.pdf
"""
from io import IOBase
from math import factorial
from array import array
from pathlib import PurePath, Path
from collections.abc import Iterable, Sequence

from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# --- Utilities: General -----------------------------------------------
def _lcs_sequence_length(X, Y, traceback=False):
    if len(X) == 0 or len(Y) == 0:
        return 0

    # a len(Y) by len(X) table of lengths
    table = [ [ 0 for j, _ in enumerate(Y) ] for i, _ in enumerate(X) ]

    def _query(i, j):
        nonlocal table
        return (
            table[i][j],
            0 if i == 0 else table[i-1][j],
            0 if j == 0 else table[i][j-1],
            0 if i == 0 or j == 0 else table[i-1][j-1]
        )

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            _, top, left, top_left = _query(i, j)
            if x == y:
                table[i][j] = top_left + 1
            else:
                table[i][j] = max(top, left)
    
    def _trace_back(i, j):
        if i < 0 or j < 0:
            return []

        nonlocal X, Y, table, _query
        current, top, left, top_left = _query(i, j)
        if current == 0:
            return []

        table[i][j] = 0
        if X[i] == Y[j]:
            char = X[i]
            rest = _trace_back(i - 1, j - 1)
            if len(rest) == 0:
                return [ [ char ] ]
            else:
                return [ [ *x, char ] for x in rest ]
        elif top > left:
            return _trace_back(i - 1, j)
        elif left > top:
            return _trace_back(i, j - 1)
        else:
            rest_top = _trace_back(i - 1, j)
            rest_left = _trace_back(i, j - 1)
            return [ *rest_top, *rest_left ]
    
    if traceback:
        return _trace_back(len(X) - 1, len(Y) - 1), table[-1][-1]

    return table[-1][-1]


def _wlcs_sequence_length(X, Y, weight_f):
    if not callable(weight_f):
        raise ValueError("weight_f function must be specified")

    # the two tables used in the algorithm specified by the paper
    w_table = [ [ 0 for j, _ in enumerate(Y) ] for i, _ in enumerate(X) ]
    c_table = [ [ 0 for j, _ in enumerate(Y) ] for i, _ in enumerate(X) ]

    def _query(table, i, j):
        return (
            table[i][j],
            0 if i == 0 else table[i - 1][j],
            0 if j == 0 else table[i][j - 1],
            0 if i == 0 or j == 0 else table[i - 1][j - 1]
        )

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if x == y:
                _, _, _, k = _query(w_table, i, j)
                _, _, _, l = _query(c_table, i, j)
                c_table[i][j] = l + weight_f(k + 1) - weight_f(k)
                w_table[i][j] = k + 1
            else:
                _, top, left, _ = _query(c_table, i, j)
                if top > left:
                    c_table[i][j] = top
                else:
                    c_table[i][j] = left
    
    return c_table[-1][-1]


def _lcs_string_length(X, Y, traceback=False):
    if len(X) == 0 or len(Y) == 0:
        return 0

    # a len(Y) by len(X) table of lengths
    table = [ [ 0 for j, _ in enumerate(Y) ] for i, _ in enumerate(X) ]
    ref = []
    maxlen = 0

    def _query(i, j):
        nonlocal table
        return (
            table[i][j],
            0 if i == 0 else table[i-1][j],
            0 if j == 0 else table[i][j-1],
            0 if i == 0 or j == 0 else table[i-1][j-1]
        )
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            _, top, left, top_left = _query(i, j)
            top_left_p1 = top_left + 1
            if x == y:
                table[i][j] = top_left_p1
                if top_left_p1 >= maxlen:
                    ref.append((top_left_p1, i, j))
                    maxlen = top_left_p1
    
    def _trace_back(i, j):
        nonlocal X, _query
        result = []

        k = 0
        max_k = min(i, j)
        while k <= max_k:
            current, _, _, _ = _query(i - k, j - k)
            if current <= 0:
                break
            result.insert(0, X[i - k])
            k += 1
        
        return result

    if traceback:
        results = []
        for x in reversed(ref):
            if x[0] < maxlen:
                break
            results.append(_trace_back(x[1], x[2]))
        return results, maxlen

    return maxlen


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
    Iterates a dictionary of named document sources and loads in their
    content. The document values may be any of the following:

        - String:    the string itself,
        - Path-like: the content of the file it points to,
        - IO-like:   the content of the IO stream,
        - Iterable:  the concatenated __str__ of the objects,

    If the documents are not one of those types, a ValueError is raised.

    :param named_sources
        A dictionary of documents to load whose names are given by
        their key in the dictionary.
    
    :param include_name_list
        Option dictating whether to include the list of names for each
        document received in the dictionary.

    :return
        (<bool:success>, <str:report>, [<list:names>,] <list:result>)
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
                loaded.append(''.join([ str(x) for x in source ]))
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
    """
    A function wrapping common routines to extract content from documents
    to be evaluated by any of the ROUGE functions.

    :param named_documents
        The dictionary of named documents; see `_get_string_content(...)`
    
    :param tokenizer
        The word tokenizer to use when splitting the document sources
        into individual tokens. 

        NOTE: the tokenizer *always* runs before the `preprocs`.
    
    :param preprocs
        An iterable of callable objects that perform additional preprocessing
        steps on the tokenized data. The preprocessors are executed in the
        order in which they are iterated.

    :return
        If include_raw_content is true, the original content string list
        and the processed tokens are returned in a tuple.
            ([r1, r2, r3, ...], [p1, p2, p3, ...])
        
        Otherwise, the list of processed tokens is returned.
            [p1, p2, p3, ...]
    """
    # validate params
    if not callable(tokenizer):
        raise ValueError("a callable tokenizer object must be specified")

    # prepare the documents, i.e. tokenize and preprocess them
    success, report, loaded = _get_string_content(named_documents, include_name_list=False)
    if not success:
        raise ValueError(f"failed to load content: {report}")

    if preprocs is None: # initialize empty list for convenience
        preprocs = []

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
    return prepared


# --- Interface --------------------------------------------------------
def rouge_n(ref, can, tokenizer=word_tokenize, preprocs=None, n=1, beta=1):
    # TODO: implement Jackknifing option
    """
    Evaluates the candidate document against the reference document
    using the n-gram ROUGE evaluation routine.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.
    
    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
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
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_lcs_sentence(ref, can, tokenizer=word_tokenize, preprocs=None, beta=1):
    """
    Evaluates the candidate sentence document against the reference
    sentence document using the longest common subsequence ROUGE
    evaluation routine.

    This differs from `rouge_lcs_summary` by using a slightly altered
    algorithm on sentences rather than on summaries.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.
    
    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
    """
    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )

    length = _lcs_sequence_length(r_tokens, c_tokens)
    R = length / len(r_tokens)
    P = length / len(c_tokens)
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_lcs_summary(ref, can, sent_tokenizer=sent_tokenize, tokenizer=word_tokenize, preprocs=None, beta=1):
    """
    Evaluates the candidate summary document against the reference
    summary document using the longest common subsequence ROUGE
    evaluation routine.

    This differs from `rouge_lcs_sentence` by using a slightly altered
    algorithm on summaries rather than on sentences.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.
    
    :param sent_tokenizer
        A callable object used to transform strings into sentence-like
        tokens.
    
    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
    """
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
        traceback, length = _lcs_sequence_length(r_sent_tokens, c_tokens, traceback=True)
        if lcs_result is None:
            raise ValueError("documents have not be tokenized into a sequence of tokens")

        common_tokens.update(*traceback)
        common_tokens_total += length

    if common_tokens_total == 0:
        return None, None

    lcsu_score = len(common_tokens) / common_tokens_total
    R = lcsu_score / len(r_tokens)
    P = lcsu_score / len(c_tokens)
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_wlcs_sentence(ref, can, weight_f, inv_weight_f, tokenizer=word_tokenize, preprocs=None, beta=1):
    """
    Evaluates the candidate sentence document against the reference
    sentence document using the weighted longest common subseuquence
    ROUGE evaluation routine.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.

    :param weight_f
        The weight function used to attribute a score for longer
        contiguous sequences. This should have the property that
        for any, or most, integers x, y, f(x + y) > f(x) + f(y)
    
    :param inv_weight_f
        The inverse of `weight_f`.
    
    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :param beta
        The beta constant used to compute the F-measure.

    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
    """
    if not callable(weight_f) or not callable(inv_weight_f):
        raise ValueError("weight_f and inv_weight_f functions must be specified and callable objects.")

    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )

    wlcs_result = _wlcs_sequence_length(r_tokens, c_tokens, weight_f)
    R = inv_weight_f(wlcs_result / weight_f(len(ref)))
    P = inv_weight_f(wlcs_result / weight_f(len(can)))
    F = (1 + beta * beta) * R * P / (R + beta * beta * P)
    return R, P, F


def rouge_su(ref, can, tokenizer=word_tokenize, preprocs=None, sodm=None, beta=1):
    """
    Evaluates the candidate document against the reference document
    using the skip bigram ROUGE evaluation routine with the start of
    document mark extension.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.

    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :param beta
        The beta constant used to compute the F-measure.

    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
    """
    r_tokens, c_tokens = _rouge_prepare(
        { "reference": ref, "candidate": can },
        tokenizer=tokenizer,
        preprocs=preprocs
    )
    if sodm is not None:
        r_tokens.add(str(sodm))
        c_tokens.add(str(sodm))

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


def rouge_s(ref, can, tokenizer=word_tokenize, preprocs=None, beta=1):
    """
    Evaluates the candidate document against the reference document
    using the skip bigram ROUGE evaluation routine.

    :param ref
        The reference summary.
    
    :param can
        The candidate summary.

    :param tokenizer
        A callable object used to transform strings into word-like
        tokens.
    
    :param preprocs
        Preprocessor callable objects that perform operations on the
        tokens after the document has been tokenized.
    
    :param beta
        The beta constant used to compute the F-measure.

    :return
        (<float:recall>, <float:precision>, <float:f-measure>)
    """
    return rouge_su(ref, can, tokenizer, preprocs, "<s>", beta)