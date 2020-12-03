"""
Implementation of the ROUGE evaluation system as described by
https://www.aclweb.org/anthology/W04-1013.pdf
"""
import math
from enum import Enum
from pathlib import PurePath

import udax.algorithm as algo
import udax.statistics as stat


class Score:

    @staticmethod
    def average(*scores, scorelist=None):
        if scorelist is None:
            scorelist = []
        scorelist.extend(scores)
        avg_recall    = 0
        avg_precision = 0
        avg_f_score   = 0
        for x in scorelist:
            avg_recall    += x.recall
            avg_precision += x.precision
            avg_f_score   += x.f_score
        return Score(
            avg_recall    / len(scorelist), 
            avg_precision / len(scorelist), 
            avg_f_score   / len(scorelist)
        )

    def __init__(self, recall, precision, f_score):
        self.recall    = recall
        self.precision = precision
        self.f_score   = f_score
    
    def __repr__(self):
        return "R: %.3f, P: %.3f, F: %.3f" % (self.recall, self.precision, self.f_score)


class Report:

    def __init__(self, name, score, opaque=None):
        """
        :param name
            The name of the candidate document.
        
        :param score
            A `Score` object as described above.
        
        :param opaque
            Any object containing more specific data.
        """
        self.name = name
        self.score = score
        self.opaque = opaque

    def __repr__(self):
        return "Candidate: %s, Score(%s)" % (self.name, repr(self.score))


class Document:

    def __init__(self, name, content):
        self.name = name
        self.content = content


class Task:

    @staticmethod
    def autodocs(*sources, **named_sources):
        documents = []
        for name, source in [ *enumerate(sources), *named_sources.items() ]:
            try:
                content = None
                if isinstance(source, str):
                    if source.startswith("file:"):
                        content = open(source[5:], mode="r").read()
                    else:
                        content = source
                elif isinstance(source, PurePath):
                    content = source.open(mode="r").read()
                elif isinstance(source, IOBase):
                    content = source.read()
                elif isinstance(source, Iterable):
                    documents.extend(autodocs(*source))
                    continue # continue to avoid adding null content to list
                else:
                    raise ValueError(f"Unknown source type at index {i}")
                documents.append(Document(name, content))
            except IOError:
                raise IOError(f"processing source at index {i}")
        return documents

    def __init__(self, references: list, candidates: list):
        self.ref_documents = references
        self.ref_count     = len(self.ref_documents)
        self.can_documents = candidates
        self.can_count     = len(self.can_documents)

        if self.ref_count == 0:
            raise ValueError("At least one reference document is required")

        if self.can_count == 0:
            raise ValueError("At least one candidate document is required")
        

def n(task, word_tokenizer, N=1, jackknife=True, beta=1):
    # If multiple candidates are specified, multiple reports
    # are generated.
    if len(task.can_documents) > 1:
        reports = []
        for can in task.can_documents:
            ntask = Task(task.ref_documents, [ can ])
            reports.append(
                n(ntask, word_tokenizer, N, jackknife, beta))
        return reports
    
    # We are now guaranteed to have a single candidate document
    candidate = task.can_documents[0]

    if task.ref_count > 1 and jackknife:
        reports = []
        for refs in algo.comb(task.ref_documents, task.ref_count - 1):
            ntask = Task(list(refs), task.can_documents)
            reports.append(
                n(ntask, word_tokenizer, N, False, beta))
        return Report(candidate.name, Score.average(scorelist=[ x.score for x in reports ]))
    
    can_tokens = word_tokenizer(candidate.content)
    can_blocks = 1 + len(can_tokens) - N

    max_report = None
    for reference in task.ref_documents:
        ref_tokens = word_tokenizer(reference.content)
        ref_blocks = 1 + len(ref_tokens) - N
        ref_grams = dict()
        for gram in algo.blocks(ref_tokens, size=N):
            if gram not in ref_grams:
                ref_grams[gram] = 1
            else:
                ref_grams[gram] += 1
        
        ref_matches = 0
        can_matches = 0
        for gram in algo.blocks(can_tokens, size=N):
            if gram in ref_grams:
                can_matches += 1
                if ref_grams[gram] > 0:
                    ref_matches += 1
                    ref_grams[gram] -= 1
        
        R = ref_matches / ref_blocks
        P = can_matches / can_blocks
        F = stat.f_score(R, P, beta)
        score = Score(R, P, F)
        # and by the specification we take the "maximum" of the
        # reports. I will assume here that it means the highest
        # f-score.
        if max_report is None or max_report.score.f_score < score.f_score:
            max_report = Report(candidate.name, score)
    return max_report


class LCSMode(Enum):
    SENTENCE = 0
    SUMMARY  = 1


def wlcs(task, sent_tokenizer, word_tokenizer, weight_f=lambda x: x * x, inv_weight_f=lambda x: math.sqrt(x), lcsmode=LCSMode.SENTENCE, jackknife=True, beta=1):
    # If multiple candidates are specified, multiple reports
    # are generated.
    if len(task.can_documents) > 1:
        reports = []
        for can in task.can_documents:
            ntask = Task(task.ref_documents, [ can ])
            reports.append(
                wlcs(ntask, sent_tokenizer, word_tokenizer, weight_f, inv_weight_f, lcsmode, jackknife, beta))
        return reports

    # We are now guaranteed to have a single candidate document
    candidate = task.can_documents[0]

    if task.ref_count > 1 and jackknife:
        reports = []
        for refs in algo.comb(task.ref_documents, task.ref_count - 1):
            ntask = Task(list(refs), task.can_documents)
            reports.append(
                wlcs(ntask, sent_tokenizer, word_tokenizer, weight_f, inv_weight_f, lcsmode, False, beta))
        return Report(candidate.name, Score.average(scorelist=[ x.score for x in reports ]))
    
    if LCSMode.SENTENCE == lcsmode: # --- SENTENCE -----------------------------
        can_tokens = word_tokenizer(candidate.content)
        max_report = None
        for reference in task.ref_documents:
            ref_tokens = word_tokenizer(reference.content)

            wlcs_score = algo.wlcsubsequence(ref_tokens, can_tokens, weight_f, traceback=False)
            R = inv_weight_f(wlcs_score / weight_f(len(ref_tokens)))
            P = inv_weight_f(wlcs_score / weight_f(len(can_tokens)))
            F = stat.f_score(R, P, beta)
            score = Score(R, P, F)
            # and by the specification we take the "maximum" of the
            # reports. I will assume here that it means the highest
            # f-score.
            if max_report is None or max_report.score.f_score < score.f_score:
                max_report = Report(candidate.name, score)
        return max_report
    elif LCSMode.SUMMARY == lcsmode: # --- SUMMARY -----------------------------
        # can_tokens = word_tokenizer(candidate.content)
        # max_report = None
        # for reference in task.ref_documents:
        #     ref_tokens = word_tokenizer(reference.content)
        #     common_tokens = set()
        #     common_tokens_score = 0
        #     for ref_sentence in sent_tokenizer(reference.content):
        #         ref_sent_tokens = word_tokenizer(ref_sentence)
        #         traceback, wlcs_score = algo.wlcsubsequence(ref_sent_tokens, can_tokens, weight_f)
        #         for trace in traceback:
        #             common_tokens.update(*traceback)
        #             common_tokens_score += wlcs_score

        #     if common_tokens_score == 0:
        #         return Report(candidate.name, Score(0, 0, 0), False)
        #     
        #     lcsu_score = len(common_tokens) / common_tokens_score
        #     R = lcsu_score / len(ref_tokens)
        #     P = lcsu_score / len(can_tokens)
        #     F = stat.f_score(R, P, beta)
        #     score = Score(R, P, F)
        #     # and by the specification we take the "maximum" of the
        #     # reports. I will assume here that it means the highest
        #     # f-score.
        #     if max_report is None or max_report.score.f_score < score.f_score:
        #         max_report = Report(candidate.name, score, True)
        can_sentences = [ word_tokenizer(x) for x in sent_tokenizer(candidate.content) ]
        can_tokens = word_tokenizer(candidate.content)
        max_report = None
        for reference in task.ref_documents:
            ref_tokens = word_tokenizer(reference.content)
            total_score = 0
            for ref_sentence in sent_tokenizer(reference.content):
                ref_sent_tokens = word_tokenizer(ref_sentence)
                for can_sent_tokens in can_sentences:
                    # Using any other function but the identity function in this
                    # scenario will yield useless results.
                    total_score += algo.wlcsubsequence(ref_sent_tokens, can_sent_tokens, weight_f, traceback=False)
            R = total_score / len(ref_tokens)
            P = total_score / len(can_tokens)
            F = stat.f_score(R, P, beta)
            score = Score(R, P, F)
            # and by the specification we take the "maximum" of the
            # reports. I will assume here that it means the highest
            # f-score.
            if max_report is None or max_report.score.f_score < score.f_score:
                max_report = Report(candidate.name, score, True)
        return max_report
    else:
        raise ValueError(f"Unrecognized LCSMode {lcsmode}")


def lcs(task, sent_tokenizer, word_tokenizer, lcsmode=LCSMode.SENTENCE, jackknife=True, beta=1):
    weight_f = lambda x: x
    inv_weight_f = lambda y: y
    return wlcs(task, sent_tokenizer, word_tokenizer, weight_f, inv_weight_f, lcsmode, jackknife, beta)


def su(task, word_tokenizer, N=2, sodm="<s>", jackknife=True, beta=1):
    # If multiple candidates are specified, multiple reports
    # are generated.
    if len(task.can_documents) > 1:
        reports = []
        for can in task.can_documents:
            ntask = Task(task.ref_documents, [ can ])
            reports.append(
                su(ntask, word_tokenizer, N, sodm, jackknife, beta))
        return reports
    
    # We are now guaranteed to have a single candidate document
    candidate = task.can_documents[0]

    if task.ref_count > 1 and jackknife:
        reports = []
        for refs in algo.comb(task.ref_documents, task.ref_count - 1):
            ntask = Task(list(refs), task.can_documents)
            reports.append(
                su(ntask, word_tokenizer, N, sodm, False, beta))
        return Report(candidate.name, Score.average(scorelist=[ x.score for x in reports ]))
    
    sodm_str = str(sodm)
    can_tokens = word_tokenizer(candidate.content)
    if sodm is not None:
        can_tokens.insert(0, sodm_str)
    can_skips = stat.comb(len(can_tokens), N)

    max_report = None
    for reference in task.ref_documents:
        ref_tokens = word_tokenizer(reference.content)
        if sodm is not None:
            ref_tokens.insert(0, sodm_str)
        ref_skips = stat.comb(len(ref_tokens), N)

        ref_grams = dict()
        for gram in algo.comb(ref_tokens, N):
            if gram not in ref_grams:
                ref_grams[gram] = 1
            else:
                ref_grams[gram] += 1
        
        ref_matches = 0
        can_matches = 0
        for gram in algo.comb(can_tokens, N):
            if gram in ref_grams:
                can_matches += 1
                if ref_grams[gram] > 0:
                    ref_matches += 1
                    ref_grams[gram] -= 1

        R = ref_matches / ref_skips
        P = can_matches / can_skips
        F = stat.f_score(R, P, beta)
        score = Score(R, P, F)
        # and by the specification we take the "maximum" of the
        # reports. I will assume here that it means the highest
        # f-score.
        if max_report is None or max_report.score.f_score < score.f_score:
            max_report = Report(candidate.name, score)
    return max_report


def s(task, word_tokenizer, N=2, jackknife=True, beta=1):
    return su(task, word_tokenizer, N, None, jackknife, beta)


# --- Tests ------------------------------------------------------------
def TEST_n():
    from nltk.tokenize import word_tokenize

    golden = "The dog ran into the house."
    sys_perfect = "The dog ran into the house."
    sys_terrible = "Bonjour, je m'appelle Jean."

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_terrible)
    )

    # generate a report for grams of size 1-3
    print("Testing rouge.n()")
    for i in range(1, 4):
        reports = n(task, word_tokenize, N=i)
        print(f"Writing report for n={i}")
        for report in reports:
            print(report.name, report.score)


def TEST_wlcs():
    from nltk.tokenize import sent_tokenize, word_tokenize

    # sentence:
    golden           = "The dog ran into the house."
    sys_perfect      = "The dog ran into the house."
    sys_near_perfect = "The dog ran into the house for safety."
    sys_good         = "The dog had bravely ran into the house for safety."
    sys_terrible     = "Bonjour, je m'appelle Jean."

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_near_perfect, sys_good, sys_terrible)
    )

    print("Testing rouge.wlcs(SENTENCE)")
    reports = wlcs(task, sent_tokenize, word_tokenize)
    for report in reports:
        print(report.name, report.score)


def TEST_lcs():
    from nltk.tokenize import sent_tokenize, word_tokenize

    # sentence
    golden = "The dog was pondering about the newly installed wooden floor."
    sys_perfect = "The dog was pondering about the newly installed wooden floor."
    sys_good = "The dog was pondering about the wooden floor."
    sys_okay = "The dog, pondering about the wooden floor, was installed."
    sys_terrible = "Yes, tonight the dinner is hot."

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_good, sys_okay, sys_terrible)
    )

    print("Testing rouge.lcs(SENTENCE)")
    reports = lcs(task, sent_tokenize, word_tokenize)
    for report in reports:
        print(report.name, report.score)

    # summaries
    golden = "a b c d e f"
    sys_perfect = "a b c d e f"
    sys_good = "a b c x x x"
    sys_worse = "a x b x c x"

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_good, sys_worse)
    )
    
    print("Testing rouge.lcs(SUMMARY)")
    reports = lcs(task, sent_tokenize, word_tokenize, lcsmode=LCSMode.SUMMARY)
    for report in reports:
        print(report.name, report.score)


def TEST_su():
    from nltk.tokenize import word_tokenize

    golden = "Hi Carl, I'm John."
    sys_perfect = "Hi Carl, I'm John."
    sys_reverse = ".John I'm, Carl Hi"

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_reverse)
    )

    print("Testing rouge.su()")
    for i in range(1, 5):
        reports = su(task, word_tokenize, N=i)
        print(f"N={i}")
        for report in reports:
            print(report.name, report.score)


def TEST_s():
    from nltk.tokenize import word_tokenize

    golden = "Hi Carl, I'm John."
    sys_perfect = "Hi Carl, I'm John."
    sys_reverse = ".John I'm, Carl Hi"

    task = Task(
        Task.autodocs(golden),
        Task.autodocs(sys_perfect, sys_reverse)
    )

    print("Testing rouge.su()")
    for i in range(1, 5):
        reports = s(task, word_tokenize, N=i)
        print(f"N={i}")
        for report in reports:
            print(report.name, report.score)


if __name__ == "__main__":
    TEST_n()
    TEST_wlcs()
    TEST_lcs()
    TEST_su()
    TEST_s()