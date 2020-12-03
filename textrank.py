"""
TextRank summarization and keyword extraction implementations. These are
designed to be more streamlized than more general PageRank algorithms
by providing facilities specifically for keyword extraction and
summarization.
"""
from math import log10

import udax


# --- Utilities --------------------------------------------------------
def sequence_similarity(A, B):
    """
    Sequence similarity as described in the TextRank paper.
    """
    if len(A) == 0 or len(B) == 0:
        return 0

    if len(A) == 1 and len(B) == 1:
        if A[0] == B[0]:
            return 1
        return 0
    
    sim_count = 0
    B_data = set(B)
    for a in A:
        if a in B_data:
            sim_count += 1
    
    return sim_count / (log10(len(A)) + log10(len(B)))


# --- Ranking Algorithms -----------------------------------------------
def rank_pagerank(damp=0.85):
    def _rank_pagerank(gv, i):
        nonlocal damp
        inputs = gv.inputs_of(i)
        i_sum = 0
        for j, w_j in inputs:
            rank_j = gv.score_of(j)
            outputs = gv.outputs_of(j)
            j_sum = 0
            for _, w_k in outputs:
                j_sum += w_k
            if j_sum <= 0: # slight divergence from the paper, we ignore
                continue   # any input node that has a weight sum of zero
                           # because it ultimatately shouldn't contribute
                           # to the finals core.
            i_sum += w_j * rank_j / j_sum
        return (1 - damp) + damp * i_sum
    return _rank_pagerank


def rank_pagerank_unweighted(damp=0.85):
    def _rank_pagerank_unweighted(gv, i):
        nonlocal damp
        sum_ = 0
        for j, _ in gv.inputs_of(i):
            sum_ += gv.score_of(j) / len(list(gv.outputs_of(j)))
        return (1 - damp) + damp * sum_
    return _rank_pagerank_unweighted


# --- Linking Algorithms -----------------------------------------------
def link_proxy(radius=2, algorithm=lambda x, y: 1):
    def _link_proxy(tokens):
        nonlocal algorithm
        i = 0
        while i < len(tokens):
            j = max(i - radius, 0)
            while j < min(len(tokens), i + radius + 1):
                if i != j:
                    weight = algorithm(tokens[i], tokens[j])
                    if weight > 0:
                        yield (i, j, weight)
                        yield (j, i, weight)
                j += 1
            i += 1
    return _link_proxy


def link_total_proxy(algorithm=lambda x, y: 1):
    def _link_similarity(tokens):
        nonlocal algorithm
        i = 0
        while i < len(tokens):
            j = 0
            while j < len(tokens):
                if i != j:
                    weight = algorithm(tokens[i], tokens[j])
                    if weight > 0:
                        yield (i, j, weight)
                        yield (j, i, weight)
                j += 1
            i += 1
    return _link_similarity


# --- Interface --------------------------------------------------------
class GraphView:
    """
    The graph is a list of node-like list structures with each node
    list being of the form:
    
        N[0] = Index in a reference list.
        N[1] = The processed content of the node.
        N[2] = Score of the node.
        N[3] = Input links to the node (indices to other nodes in graph)
        N[4] = Input weights corresponding to their links.
        N[5] = Output links to the node (indices to other nodes in graph)
        N[6] = Output weights corresponding to their links.
    """

    def __init__(self, graph):
        self._graph = graph
    
    def reference_index_of(self, i):
        return self._graph[i][0]
    
    def content_of(self, i):
        return self._graph[i][1]
    
    def score_of(self, i):
        return self._graph[i][2]
    
    def inputs_of(self, i):
        """
        A list of tuples in the form (<link index>, <weight>)
        """
        links = self._graph[i][3]
        weights = self._graph[i][4]
        return zip(links, weights)
    
    def outputs_of(self, i):
        """
        A list of tuples in the form (<link index>, <weight>)
        """
        links = self._graph[i][5]
        weights = self._graph[i][6]
        return zip(links, weights)


def summarize(
    content, # a string of text is expected at the moment.
    sent_tokenizer,
    word_tokenizer,
    sequence_processors=None, # sequence meaning sequence of words
    link_func=link_total_proxy(algorithm=sequence_similarity),
    rank_func=rank_pagerank(),
    convthresh=1e-4):
    """
    :param link_func
        A generator taking in a list of processed tokens. The argument
        is guaranteed to be a list, but the contents are determined by
        the processors themselves. The function must be a generator
        returning a tuple (A, B, w) where
            A is the first element in the processed list,
            B is the second element in the processed list,
            w is the weight of the link.
        The link is strictly in the forward direction A -> B, if a link
        must be bidirectional, yield (A, B, w) and (B, A, w) in a single
        pass.
    
    :param rank_func
        A function taking in a GraphView and a node handle as parameters
        (in that order). The function must return a single floating
        point value for the rank of the node whose handle was passed
        as the parameter.
    """

    # First we tokenize the content into sentences which will be stored
    # in two different arrays, one for reference, and one for further
    # processing on the word/character level.
    reference = sent_tokenizer(content)

    # process the original sentence on a finer level than sentences if
    # need be.
    processed = [ word_tokenizer(x) for x in reference ]
    if sequence_processors is not None:
        for i in range(len(processed)):
            for processor in sequence_processors:
                processed[i] = processor(processed[i])

    index_map = {}
    graph = []
    for i, j, weight in link_func(processed):
        seqA = processed[i]
        seqB = processed[j]
        # we create "content" strings to be able to hash the sequence
        # so that duplicates are not added to the graph.
        contentA = ''.join([ str(x) for x in seqA ])
        contentB = ''.join([ str(x) for x in seqB ])
        I, J = None, None

        try:
            I = index_map[contentA]
        except:
            I = len(graph)
            index_map[contentA] = I
            graph.append([ i, seqA, 1, [], [], [], [] ])
        
        try:
            J = index_map[contentB]
        except:
            J = len(graph)
            index_map[contentB] = J
            graph.append([ j, seqB, 1, [], [], [], [] ])
        
        _, _, _, _, _, Iout, Iwout = graph[I]
        _, _, _, Jin, Jwin, _, _   = graph[J]

        # Link the nodes
        try:
            idx = Iout.index(J)
            Iwout[idx] = weight
        except:
            Iout.append(J)
            Iwout.append(weight)
        
        try:
            idx = Jin.index(I)
            Jwin[idx] = weight
        except:
            Jin.append(I)
            Jwin.append(weight)

    # Set the starting scores to the average of them all. 
    for node in graph:
        node[2] = node[2] / len(graph)

    # TODO: run heuristic to fix broken links 

    # Once all links are established, we can begin converging.
    graph_view = GraphView(graph)
    err = convthresh
    score_buffer = [ x[2] for x in graph ]
    while err >= convthresh:
        for i in range(len(graph)):
            n_score = rank_func(graph_view, i)
            err = n_score - score_buffer[i]
            score_buffer[i] = n_score
            if err < convthresh:  # the paper states to stop as soon as one
                break			  # evaluation goes below the threshold.
        for i, s in enumerate(score_buffer):
            graph[i][2] = s
    
    # Sort the table into the most important sentences and return the
    # sorted graph (for reuse), raw table (with parsed tokens), reference 
    # table (reference/unprocessed sentences).
    #
    # Each table is a list of tuples with the first element being the
    # rank index (0 being best, N being worst) and the second element
    # is the data itself.
    #
    # The sorted function sorts by the score of each node in reverse
    # order so that the top score is first.
    sorted_graph = list(sorted(graph, key=lambda x: x[2], reverse=True))
    raw_table = []
    ref_table = []
    for i, node in enumerate(sorted_graph):
        raw_table.append((i, node[1]))
        ref_table.append((i, reference[node[0]]))
    
    return sorted_graph, raw_table, ref_table


def keyphrases(
    content, # a string of text is expected at the moment.
    word_tokenizer,
    token_processors=None,
    link_func=link_proxy(),
    rank_func=rank_pagerank(),
    convthresh=1e-4):

    # The reference list will be used to collapse keywords into
    # keyphrases.
    reference = word_tokenizer(content)

    # Preprocessing, unlike summarization, preprocessors are fed the
    # entire token list.
    processed = [ x for x in reference ]
    if token_processors is not None:
        for processor in token_processors:
            processed = processor(processed)
    
    # index_map will maintain unique indices for each graph node. The graph
    # itself is the same array-based graph as described in summarize()
    index_map = {}
    graph = [] 
    for i, j, weight in link_func(processed):
        wordA = processed[i]
        wordB = processed[j]
        # use "content" strings to create hashes that will be used to ensure
        # uniqueness of elements in the graph.
        contentA = str(wordA)
        contentB = str(wordB)
        I, J = None, None

        try:
            I = index_map[contentA]
        except:
            I = len(graph)
            index_map[contentA] = I
            graph.append([ i, wordA, 1, [], [], [], [] ])
        
        try:
            J = index_map[contentB]
        except:
            J = len(graph)
            index_map[contentB] = J
            graph.append([ j, wordB, 1, [], [], [], [] ])

        _, _, _, _, _, Iout, Iwout = graph[I]
        _, _, _, Jin, Jwin, _, _   = graph[J]

        # Link the nodes
        try:
            idx = Iout.index(J)
            Iwout[idx] = weight
        except:
            Iout.append(J)
            Iwout.append(weight)
        
        try:
            idx = Jin.index(I)
            Jwin[idx] = weight
        except:
            Jin.append(I)
            Jwin.append(weight)

    # Set the starting scores to the average of them all.
    for node in graph:
        node[2] = node[2] / len(graph)
        
    # TODO: add heuristic to fix broken links
    
    # Once all links are established, we can begin converging.
    graph_view = GraphView(graph)
    err = convthresh
    score_buffer = [ x[2] for x in graph ]
    while err >= convthresh:
        for i in range(len(graph)):
            n_score = rank_func(graph_view, i)
            err = n_score - score_buffer[i]
            score_buffer[i] = n_score
            if err < convthresh:  # the paper states to stop as soon as one
                break			  # evaluation goes below the threshold.
        for i, s in enumerate(score_buffer):
            graph[i][2] = s
    
    # Sort the graph with the higest rated node at the front.
    sorted_graph = list(sorted(graph, key=lambda x: x[2], reverse=True))
    word_map = {}
    for node in sorted_graph: # word table maps unique words to their scores
        word_map[node[1]] = node[2]

    word_table = [ (i, x[1]) for i, x in enumerate(sorted_graph) ]

    # Take the keywords and make keyphrases out of them by collapsing with
    # respect to the reference table above.
    phrase_table = []
    phrase_set = set()
    i = 0
    while i < len(reference):
        score = 0
        phrase = []
        while i < len(reference):
            word = reference[i]
            i += 1
            if word in word_map:
                phrase.append(word)
                score += word_map[word]
            else:
                break
        if len(phrase) == 0:
            continue
        score /= len(phrase) # get the average score
        phrase_str = ' '.join(phrase)
        if phrase_str not in phrase_set:
            phrase_table.append((score, phrase_str))
            phrase_set.add(phrase_str)
    
    # Sort the phrase table by the score of the phrase and have the highest
    # score be first.
    phrase_table = list(sorted(phrase_table, key=lambda x: x[0], reverse=True))

    return sorted_graph, word_table, phrase_table


# --- Tests ------------------------------------------------------------
def TEST_paper_summarization_comparison():
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk import pos_tag

    def nouns_and_adj_only(processed):
        result = []
        for word, pos in processed:
            if pos.startswith("NN") or pos.startswith("JJ"):
                result.append(word)
        return result

    sample = \
    ' '.join([
        "BC-HurricaineGilbert, 09-11 339.",
        "BC-Hurricaine Gilbert, 0348.",
        "Hurricaine Gilbert heads toward Dominican Coast.",
        "By Ruddy Gonzalez.",
        "Associated Press Writer.",
        "Santo Domingo, Dominican Republic (AP).",
        "Hurricaine Gilbert Swept towrd the Dominican Republic Sunday, and the Civil Defense alerted its heavily populated south coast to prepare for high winds, heavy rains, and high seas.",
        "The storm was approaching from the southeast with sustained winds of 75 mph gusting to 92 mph.",
        "\"There is no need for alarm,\" Civil Defense Director Eugenio Cabral said in a television alert shortly after midnight Saturday."
        "Cabral said residents of the province of Barahona should closely follow Gilbert’s movement.",
        "An estimated 100,000 people live in the province, including 70,000 in the city of Barahona, about 125 miles west of Santo Domingo.",
        "Tropical storm Gilbert formed in the eastern Carribean and strenghtened into a hurricaine Saturday night.",
        "The National Hurricaine Center in Miami reported its position at 2 a.m. Sunday at latitude 16.1 north, longitude 67.5 west, about 140 miles south of Ponce, Puerto Rico, and 200 miles southeast of Santo Domingo.",
        "The National Weather Service in San Juan, Puerto Rico, said Gilbert was moving westard at 15 mph with a \"broad area of cloudiness and heavy weather\" rotating around the center of the storm.",
        "The weather service issued a flash flood watch for Puerto Rico and the Virgin Islands until at least 6 p.m. Sunday.",
        "Strong winds associated with the Gilbert brought coastal flooding, strong southeast winds, and up to 12 feet to Puerto Rico’s south coast.",
        "There were no reports on casualties.",
        "San Juan, on the north coast, had heavy rains and gusts Saturday, but they subsided during the night.",
        "On Saturday, Hurricane Florence was downgraded to a tropical storm, and its remnants pushed inland from the U.S. Gulf Coast.",
        "Residents returned home, happy to find little damage from 90 mph winds and sheets of rain.",
        "Florence, the sixth named storm of the 1988 Atlantic storm season, was the second hurricane.",
        "The first, Debby, reached minimal hurricane strength briefly before hitting the Mexican coast last month."
    ])

    graph, raw_table, ref_table = summarize(
        sample,
        sent_tokenize,
        word_tokenize,
        [ pos_tag, nouns_and_adj_only ]
    )
    for ref in ref_table:
        print(ref)


def TEST_paper_keyphrase_comparison():
    sample = \
    """
    Compatibility of systems of linear constraints over the set of natural numbers.
    Criteria of compatibility of a system of linear Diophantine equations, strict
    inequations, and nonstrict inequations are considered. Upper bounds for
    components of a minimal set of solutions and algorithms of construction of
    minimal generating sets of solutions for all types of systems are given.
    These criteria and the corresponding algorithms for constructing a minimal
    supporting set of solutions can be used in solving all considered types
    systems and systems of mixed types.
    """
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    useless_words = set(stopwords.words("english"))

    def rem_stopwords(processed):
        nonlocal useless_words
        result = []
        for word in processed:
            if word not in useless_words:
                result.append(word)
        return result
    
    def rem_punctuation(processed):
        result = []
        for word in processed:
            if len(udax.s_rempunct(word).strip()) != 0:
                result.append(word)
        return result

    graph, word_table, phrase_table = keyphrases(
        sample,
        word_tokenize,
        [ rem_stopwords, rem_punctuation ]
    )
    for phrase in phrase_table:
        print(phrase)


if __name__ == "__main__":
    TEST_paper_summarization_comparison()
    TEST_paper_keyphrase_comparison()