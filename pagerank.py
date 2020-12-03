"""
An implementation of the TextRank algorithm as specified in the paper
`TextRank: Bringing Order into Texts` by Rada Mihalcea and Paul Tarau:
https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

Given that the algorithm is inspired by PageRank, and this implementation
is general enough to rank anything, including text, the name 'PageRank'
will be used.
"""
import enum
from collections.abc import Iterable


# --- Classes ----------------------------------------------------------
class Direction(enum.IntEnum):
	FORWARDS  = 0x1
	BACKWARDS = 0x2
	BI        = 0x3


class GraphView:

	def __init__(self, graph):
		self._graph = graph

	def value_of(self, i):
		self._validate_node_reference(i)
		return self._graph[i][0]

	def score_of(self, i):
		self._validate_node_reference(i)
		return self._graph[i][1]

	def inputs_of(self, i, include_weights=True):
		self._validate_node_reference(i)
		if include_weights:
			return zip(self._graph[i][2], self._graph[i][3])
		else:
			return self._graph[i][2]

	def outputs_of(self, i, include_weights=True):
		self._validate_node_reference(i)
		if include_weights:
			return zip(self._graph[i][4], self._graph[i][5])
		else:
			return self._graph[i][4]

	def _validate_node_reference(self, i):
		if not isinstance(i, int) or i < 0 or len(self._graph) <= i:
			raise ValueError("invalid node reference")


class LinkError(RuntimeError):

	def __init__(self, message=None):
		super().__init__(message)


class LinkMaker:

	def __init__(self, i_tokens, j_tokens):
		self.i_tokens = i_tokens
		self.j_tokens = j_tokens

	@property
	def origin_tokens(self):
		return self.i_tokens

	@property
	def inbound_tokens(self):
		return self.j_tokens

	def origin(self, i1, i2, weight=1, direction=Direction.FORWARDS, objects=False):
		i1_pos, i2_pos = i1, i2
		if objects:
			i1_pos = self._index_no_except(self.i_tokens, i1)
			i2_pos = self._index_no_except(self.i_tokens, i2)
		return self.monolithic(i1_pos, i2_pos, weight, direction)

	def cross(self, i, j, weight=1, direction=Direction.FORWARDS, objects=False):
		i_pos, j_pos = i, j
		if objects:
			i_pos = self._index_no_except(self.i_tokens, i)
			j_pos = self._index_no_except(self.j_tokens, j)
		return self.monolithic(i_pos, len(self.i_tokens) + j_pos, weight, direction)

	def inbound(self, j1, j2, weight=1, direction=Direction.FORWARDS, objects=False):
		j1_pos, j2_pos = j1, j2
		if objects:
			j1_pos = self._index_no_except(self.j_tokens, j1)
			j2_pos = self._index_no_except(self.j_tokens, j2)
		return self.monolithic(
			len(self.i_tokens) + j1_pos,
			len(self.i_tokens) + j2_pos,
			weight, direction)

	def monolithic(self, i, j, weight=1, direction=Direction.FORWARDS):
		i_size = len(self.i_tokens)
		j_size = len(self.j_tokens)
		size = i_size + j_size
		self._validate_indices(i, j, size, size, seq_sizes=True)
		self._validate_direction(direction)
		return weight, direction, i, j

	def query_token(self, idx):
		if idx >= len(self.i_tokens):
			return self.j_tokens[idx - len(self.i_tokens)]
		else:
			return self.i_tokens[idx]

	@staticmethod
	def _index_no_except(seq, obj):
		try:
			return seq.index(obj)
		except ValueError:
			return -1

	@staticmethod
	def _validate_indices(i, j, i_seq, j_seq, seq_sizes=False):
		if not isinstance(i, int) or not isinstance(j, int):
			raise ValueError("indices must be integer types")
		if seq_sizes:
			if i < 0 or i_seq <= i or \
			   j < 0 or j_seq <= j:
				raise ValueError("link targets out of bounds or don't exist")
		else:
			if i < 0 or len(i_seq) <= i or \
			   j < 0 or len(j_seq) <= j:
				raise ValueError("link targets out of bounds or don't exist")

	@staticmethod
	def _validate_direction(direction):
		if direction != Direction.BI and \
			direction != Direction.FORWARDS and \
			direction != Direction.BACKWARDS:
			raise ValueError(
				"direction must be either PageRank.BI, PageRank.FORWARDS, or PageRank.BACKWARDS")


class PageRank:
	"""
	Node, T, structure:

		T[0] - sentence index in the sample
		T[1] - the score of the sentence
		T[2] - list of inward connections represented by sentence indices.
		T[3] - list of inward weights corresponding to some connections.
		T[4] - list of outward connections represented by sentence indices.
		T[5] - list of outward weights corresponding to some connections.
	"""

	def __init__(self, preprocs=None, link_func=None, rank_func=None):
		self._preprocs = preprocs
		self._link_func = link_func
		self._rank_func = rank_func
		self._graph = []
		self._graph_view = GraphView(self._graph)
		self._tokens = []
		self._validate_init()

	def feed(self, raw_data, defscore=1):
		# run the data transforms
		tokens = raw_data
		for p in self._preprocs:
			tokens = p(tokens)

		# validate that the final staged data & expand the graph
		self._validate_feed_tokens(tokens)
		self._expand_graph(tokens, defscore)

		# link the tokens
		for weight, direction, i, j in self._link_func(LinkMaker(self._tokens, tokens)):
			Icin, Iwin, Icout, Iwout = (None,) * 4
			Jcin, Jwin, Jcout, Jwout = (None,) * 4
			try:
				_, _, Icin, Iwin, Icout, Iwout = self._graph[i]
				_, _, Jcin, Jwin, Jcout, Jwout = self._graph[j]
			except ValueError:
				raise LinkError(
					"corrupted link table, do not modify the graph outside of the PageRank interface")

			if (direction & Direction.FORWARDS):
				i_has_ref = j in Icout
				j_has_ref = i in Jcin
				if i_has_ref and j_has_ref:
					raise LinkError("nodes are already linked in the forward direction")
				if i_has_ref != j_has_ref:
					raise LinkError(
						"corrupted link table, do not modify graph outside of the PageRank interface")
				Icout.append(j)
				Iwout.append(weight)
				Jcin.append(i)
				Jwin.append(weight)

			if (direction & Direction.BACKWARDS):
				i_has_ref = j in Icin
				j_has_ref = i in Jcout
				if i_has_ref and j_has_ref:
					raise LinkError("nodes are already linked in the backward direction")
				if i_has_ref != j_has_ref:
					raise LinkError(
						"corrupted link table, do not modify graph outside of the PageRank interface")
				Icin.append(j)
				Iwin.append(weight)
				Jcout.append(i)
				Jwout.append(weight)

		self._tokens.extend(tokens)

	def get_node_count(self):
		return len(self._graph)

	def set_scores(self, score_func=None):
		if score_func is None or not callable(score_func):
			raise ValueError("the score function must be a callable object")
		for i, g in enumerate(self._graph):
			g[1] = score_func(self._graph_view, i)

	def set_inverse_uniform_scores(self):
		u_score = 1 / len(self._graph)
		for g in self._graph:
			g[1] = u_score

	def set_constant_uniform_scores(self, const=1):
		for g in self._graph:
			g[1] = const

	def execute(self, convthresh=1e-4):
		err = convthresh
		rescore_buffer = [x[1] for x in self._graph]
		while err >= convthresh:
			for i, node in enumerate(self._graph):
				n_score = self._rank_func(self._graph_view, i)
				err = abs(n_score - rescore_buffer[i])
				rescore_buffer[i] = n_score
				if err < convthresh:  # the paper states to stop as soon as one
					break			# evaluation goes below the threshold.
			for i, s in enumerate(rescore_buffer):
				self._graph[i][1] = s

		ranked_graph = sorted(self._graph, key=lambda x: x[1], reverse=True)
		ranked_table = []
		for g in ranked_graph:  # manual zip to avoid zipping the 4 other lists
			ranked_table.append((g[0], g[1]))
		return ranked_table

	def _expand_graph(self, tokens, defscore):
		for token in tokens:
			self._graph.append([token, defscore, [], [], [], []])

	def _validate_init(self):
		# validate preprocessors
		if self._preprocs is None:
			raise ValueError("at least one preprocessor function must be specified")

		if not isinstance(self._preprocs, Iterable):
			raise ValueError("preprocessors must be in an iterable container")

		for p in self._preprocs:
			if not callable(p):
				raise ValueError(f"object {p} is not a valid preprocessor function")

		# validate the link function
		if self._link_func is None:
			raise ValueError("a link function must be specified")

		if not callable(self._link_func):
			raise ValueError("the link function must be a callable object")

		# validate the rank function
		if self._rank_func is None:
			raise ValueError("a rank function must be specified")

		if not callable(self._rank_func):
			raise ValueError("the rank function must be a callable object")

	def _validate_feed_tokens(self, tokens):
		# ensure that it is an iterable
		if not isinstance(tokens, Iterable):
			raise ValueError(
				"the final preprocessor stage must yield an iterable of tokens")


# --- Linking Algorithms -----------------------------------------------
def proxy(rad=2, weight_f=lambda x, y: 1, direction=Direction.FORWARDS):
	if direction == Direction.BI:
		raise ValueError(
			"the proxy linker already establish bidrectional links, use Direction.FORWARDS or Direction.BACKWARDS")

	rad = max(0, rad)

	def _proxy(linker: LinkMaker):
		nonlocal rad
		i_tokens = linker.i_tokens
		j_tokens = linker.j_tokens
		i_size = len(linker.i_tokens)
		j_size = len(linker.j_tokens)
		size = i_size + j_size
		k = i_size
		while k < size:
			low = max(0, k - rad)
			high = min(size - 1, k + rad)
			source_tkn = linker.query_token(k)
			while low <= high:
				if low != k:
					target_tkn = linker.query_token(low)
					yield linker.monolithic(k, low, weight_f(source_tkn, target_tkn), direction)
				low += 1
			k += 1

	return _proxy


def total_proxy(weight_f=lambda x, y: 1, direction=Direction.FORWARDS):
	if direction == Direction.BI:
		raise ValueError("the proxy linker already establish bidrectional links, use Direction.FORWARDS or Direction.BACKWARDS")

	def _total_proxy(linker: LinkMaker):
		i_tokens = linker.i_tokens
		j_tokens = linker.j_tokens
		i_size = len(linker.i_tokens)
		j_size = len(linker.j_tokens)
		size = i_size + j_size
		k = i_size
		while k < size:
			w = 0
			source_tkn = linker.query_token(k)
			while w < size:
				if w != k:
					target_tkn = linker.query_token(w)
					yield linker.monolithic(k, w, weight_f(source_tkn, target_tkn), direction)
				w += 1
			k += 1

	return _total_proxy


# --- Ranking Algorithms -----------------------------------------------
def pagerank(damp=0.85):
	if damp < 0 or 1 < damp:
		raise ValueError("damping factor must be in the range [0, 1]")
	def _pagerank(G, i):
		nonlocal damp

		sum = 0
		for j, w_ji in G.inputs_of(i):
			Jscore = G.score_of(j)
			denom = 0
			for k, w_jk in G.outputs_of(j):
				denom += w_jk
			if denom == 0: # hack :sad: TODO: remove broken weight links and stuff
				denom = 1e-4
			sum += w_ji * Jscore / denom
		return (1 - damp) + damp * sum

	return _pagerank


# --- Algorithms -------------------------------------------------------
def similarity():
	def _similarity(i, j):
		"""
		Computes the similarity between sequences `i` and `j` using the
		formula specified in the TextRank paper.
		"""
		from math import log10
		if len(i) == 0 or len(j) == 0:
			return 0

		if len(i) == 1 and len(j) == 1:
			if i[0] == j[0]:
				return 1
			return 0
		
		j_data = set(j)
		sim_count = 0
		for x in i:
			if x in j_data:
				sim_count += 1

		return sim_count / (log10(len(i)) + log10(len(j)))
	return _similarity


def text_similarity(tokenizer):
	def _text_similarity(i, j):
		"""
		Computes the similarity between `i` and `j` strings by word
		tokenizing and comparing overlap based on the algorithm specified
		in the TextRank paper.
		"""
		nonlocal tokenizer
		from math import log10

		i_words = tokenizer(i)
		j_words = tokenizer(j)
		if len(i_words) == 0 or len(j_words) == 0:
			return 0

		if len(i_words) == 1 and len(j_words) == 1:
			if i_words[0] == j_words[0]:
				return 1
			return 0

		j_data = set(j_words)

		sim_count = 0
		for x in i_words:
			if x in j_words:
				sim_count += 1

		return sim_count / (log10(len(i_words)) + log10(len(j_words)))
	return _text_similarity