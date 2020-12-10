import io
import os
import sys
import math
import time
import string
import enum
from collections.abc import Iterable, Sequence, Mapping
from pathlib import PurePath


def deprecated(since=None, reason=None, alternative=None):
	def _deprecated(func):
		import io
		import sys

		msg = io.StringIO()
		msg.write(f"[WARN]: {func.__name__} is deprecated")
		if since is not None:
			msg.write(f" since {since}")
		if reason is not None:
			msg.write(f": {reason}")
		if alternative is not None:
			msg.write(f", use {alternative} instead")
		msg.write('.')
		msgstr = msg.getvalue()

		def _run_deprecated(*args, **kwargs):
			nonlocal msgstr, func
			print(msgstr, file=sys.stderr)
			return func(*args, **kwargs)

		return _run_deprecated

	return _deprecated


def time_it(func):
	def _func(*args, **kwargs):
		nonlocal func

		begin = time.monotonic_ns()
		result = func(*args, **kwargs)
		end = time.monotonic_ns()
		delta = end - begin

		print("'%s' execution time: %.2fs | %.2fms | %dμs | %dns" % \
			(func.__name__, delta * 1e-9, delta * 1e-6, delta * 1e-3, delta))

		return result
	return _func


# +-- Strings --------------------------------------+
@deprecated(reason="moved to separate module", alternative="udax.strutil.map")
def s_map(p_string, domain, mapping):
	"""
	Replaces all characters of the `domain` in `p_string` with their
	respective mapping in `mapping`. The length of the domain string
	must be the same as the mapping string unless the mapping string
	is empty or a single character, in which case all domain 
	characters will	be either replaced or substituted with that mapping 
	character in the `p_string`.

	:param p_string
		The string whose characters in domain to substitute with their
		respective mappings in `mapping`.
	
	:param domain
		The string of characters to replace in the `p_string`. If some
		characters reappear, this will not affect the substitution process,
		the extra characters are simply ignored.

	:param mapping
		The corresponding mapping of the `domain`. This must match the
		length of the `domain` string, empty, or be a single character 
		to which all domain characters will be mapped to.
	
	If any of `p_string`, `domain`, or `mapping` are None, this function
	does nothing and simply returns `p_string`.

	If len(mapping) != len(domain) and len(mapping) > 1, this function
	raises a ValueError.
	"""
	if p_string is None or domain is None or mapping is None:
		return p_string

	res = io.StringIO()

	# void mapping
	if len(mapping) == 0:
		for c in p_string:
			if c not in domain:
				res.write(c)
	# surjective mapping
	elif len(mapping) == 1:
		for c in p_string:
			if c in domain:
				res.write(mapping)
			else:
				res.write(c)
	# injective mapping
	elif len(mapping) == len(domain):
		for c in p_string:
			pos = domain.find(c)
			if pos != -1:
				res.write(mapping[pos])
			else:
				res.write(c)
	else:
		raise ValueError("len(mapping) > 1 and len(mapping) != len(domain)")

	return res.getvalue()


@deprecated(reason="moved to separate module", alternative="udax.strutil.reppunct")
def s_reppunct(p_string, mapping):
	"""
	Replaces puncutation characters in the given string with a custom
	mapping. Punctuation characters are defined by `string.punctuation`
	from the standard library.

	See `s_map` for more information about the mapping parameter. This
	is directly passed into `s_map`. Usually, mapping needs to be a single
	character unless you know exaclty how long the `string.punctuation`
	string is.

	:param p_string
		The string whose punctuation to replace with the mapping.
	
	:param mapping
		Usually a single character string, passed into `s_map` to replace
		the punctuation. If you know the length of `string.punctuation` and
		which characters you would like to remap in it, you may pass in
		the appropriate mapping string.
	"""
	return s_map(p_string, string.punctuation, mapping)


@deprecated(reason="moved to separate module", alternative="udax.strutil.rempunct")
def s_rempunct(p_string):
	"""
	Removes punctuation characters from the given string. Punctuation
	characters are defined by `string.punctuation` from the standard
	library.

	:param p_string
		The string whose punctuation to remove.
	"""
	return s_reppunct(p_string, '')


@deprecated(reason="moved to separate module", alternative="udax.strutil.liftpunct")
def s_liftpunct(p_string):
	"""
	Similar to `s_rempunct`, but this will replace all puncutation with
	a single whitespace character rather than removing the punctuation
	entirely.

	:param p_string
		The string whose puncutation to replace with a whitespace, i.e.
		'lift' the punctuation.
	"""
	return s_reppunct(p_string, ' ')


@deprecated(reason="moved to separate module", alternative="udax.strutil.norm")
def s_norm(p_string, uppercase=False):
	"""
	Filters out all punctuation, normalizes the casing to either
	lowercase or uppercase of all letters, and removes extraneous
	whitespace between characters. That is, all whitespace will
	be replaced by a single space character separating the words.

	:param p_string
		The string to normalize.
	
	:param uppercase
		Whether to make the resulting string uppercase. By default,
		the resulting string will be all lowercase.
	"""
	nopunct = s_liftpunct(p_string)
	if uppercase:
		nopunct = nopunct.upper()
	else:
		nopunct = nopunct.lower()
	return ' '.join(nopunct.split())


# +-- Files ----------------------------------------+
@deprecated()
def f_open_large_read(path, *args, **kwargs):
	"""
	A utility function to open a file handle for reading 
	with a default 64MB buffer size. The buffer size may 
	be overriden in the `kwargs`, but there is no point 
	in doing so as this is meant to be a quick utility.

	:param path
		A string or pathlike object pointing to a desired
	"""
	if "mode" not in kwargs:
		kwargs["mode"] = "r"

	if "buffering" not in kwargs:
		kwargs["buffering"] = 2 ** 26

	return open(path, **kwargs)


@deprecated()
def f_open_large_write(path, *args, **kwargs):
	"""
	A utility function to open a file handle for reading 
	with a default 16MB buffer size. The buffer size may 
	be overriden in the `kwargs`, but there is no point 
	in doing so as this is meant to be a quick utility.

	:param path
		A string or pathlike object pointing to a desired
	"""
	if "mode" not in kwargs:
		kwargs["mode"] = "w"

	if "buffering" not in kwargs:
		kwargs["buffering"] = 2 ** 24

	return open(path, **kwargs)


@deprecated(reason="moved to separate module", alternative="udax.files.line_count")
def f_line_count(path, *args, **kwargs):
	"""
	A quick utility function to count the number of lines in
	the given file.

	:param path
		A string or pathlike object pointing to a file whose
		lines to count.
	
	:return int
		The number of lines in the given file.
	"""
	lines = 0
	with open(path, **kwargs) as handle:
		for _ in handle:
			lines += 1
	return lines


@deprecated(reason="moved to separate module", alternative="udax.files.line_count")
def f_line_count_fd(stream, *args, **kwargs):
	"""
	A quick utility function to count the remaining number of
	lines in an already opened stream. If the stream is seekable,
	this function will save and revert to the original position
	after counting

	:param stream
		A stream object which may be iterated line by line.

	:param offset
		A numeric offset to apply to the resulting line count.
	
	:return int
		The number of lines remaining in the given stream.
	"""
	prev = None
	if stream.seekable():
		prev = stream.tell()
	
	lines = 0
	if "offset" in kwargs:
		lines += kwargs["offset"]
	for _ in stream:
		lines += 1
	
	if prev is not None:
		stream.seek(prev)
	
	return lines


# +-- Data Structure -------------------------------+
@deprecated(reason="moved to separate module", alternative="udax.ds.index")
def d_index_no_except(seq, obj, i=None, j=None):
	"""
	Provides a no-exception wrapper for indexing sequence types. With
	the exception of the first parameter, the interface for the sequence
	`index` function is equivalent.

	:param seq
		The sequence object to index.

	:param obj
		The object whose index to retrieve.
	
	:param i
		The inclusive starting index of the search.
	
	:param j
		The exclusive ending index of the search.
	"""
	try:
		if i is not None:
			if j is not None:
				return seq.index(obj, i=i, j=j)
			else:
				return seq.index(obj, i=i)
		else:
			return seq.index(obj)
	except ValueError:
		return -1


# +-- CSV ------------------------------------------+
@deprecated(reason="moved to separate module", alternative="udax.csv.parse")
def csv_parseln(
		p_line, 
		delim=',', 
		quote='\"',
		esc='\\'):
	"""
	Given a sample CSV line, this function will parse the line into
	a list of cells representing that CSV row. If the given `p_line`
	contains newline characters, only the content present before
	the first newline character is parsed.

	:param p_line
		The string representing the CSV line to parse. This is usually
		a line in a CSV file obtained via `f.readline()` or of the likes.
	
	:param delim
		The cell delimiter. By default this is the standard comma.
	
	:param quote
		The quote character used to encase complex cell information that
		may otherwise break the entire CSV structure, for example, by
		containing an illegal delimiter character.
	
	:param esc
		The escape character used to escape sensitive characters.
	
	:return list
		The list of cells in the given row line.

	If `p_line` is None, this function does nothing and returns None.
	
	If `delim` is None or `quote` is None or `esc` is None, this function throws
	a ValueError.

	If len(`delim`) != 1 or len(`quote`) != 1 or len(`esc`) != 1, this function
	also throws a ValueError.
	"""
	if p_line is None:
		return None

	if delim is None or quote is None or esc is None:
		raise ValueError("delim, quote, and/or esc cannot be None")
	
	if len(delim) != 1 and len(quote) != 1 and len(esc) != 1:
		raise ValueError("len of delim, quote, and esc must be 1")

	cells = []
	buf = io.StringIO()

	in_quote = False
	esc_next = False

	for c in p_line:
		if c == '\n':
			break

		if esc_next:
			buf.write(c)
			esc_next = False
			continue

		if c == esc:
			esc_next = True
			continue

		if c == quote:
			in_quote = not in_quote
			continue

		if c == delim and not in_quote:
			cells.append(buf.getvalue())
			buf = io.StringIO()
			continue

		buf.write(c)
	
	leftover = buf.getvalue()
	if len(leftover) > 0:
		cells.append(leftover)
	
	return cells


@deprecated(reason="moved to separate module", alternative="udax.csv.render")
def csv_mkln(
		*args,
		delim=',',
		quote='\"',
		esc='\\'):
	"""
	Formats a CSV row that can be written to a CSV file to be
	reloaded later. The result of this function can be passed
	to `csv_parseln` to parse it back into a list of strings.

	:param args
		The list of cells to format into a CSV row ready for
		output to an external medium. If args is a list of
		a single value, being a list, then that list will
		be used as the cells for the row.

	:return str
		The string representing the formatted CSV row.

	If `delim` is None or `quote` is None or `esc` is None, this function throws
	a ValueError.

	If len(`delim`) != 1 or len(`quote`) != 1 or len(`esc`) != 1, this function
	"""
	if delim is None or quote is None or esc is None:
		raise ValueError("delim, quote, and/or esc cannot be None")
	
	if len(delim) != 1 and len(quote) != 1 and len(esc) != 1:
		raise ValueError("len of delim, quote, and esc must be 1")

	if len(args) == 1 and isinstance(args[0], Iterable):
		return csv_mkln(*args[0])

	def _format_cell(raw):
		return quote + str(raw).replace(quote, esc + quote) + quote

	return delim.join([_format_cell(x) for x in args])


@deprecated(reason="moved to separate module", alternative="udax.csv.write")
def csv_writeln(
		*args,
		stream=None,
		delim=',',
		quote='\"',
		esc='\\'):
	"""
	A utility wrapper around `csv_mkln` that writes out the generated CSV row
	string to the specified stream.

	:param stream
		The stream object to which to write the formatted CSV row to.
	
	:return void

	If `stream` is None, a ValueError is raised.
	"""
	if stream is None:
		raise ValueError("stream must not be None")

	stream.write(f"{csv_mkln(*args, delim=delim, quote=quote, esc=esc)}\n")


# +-------------------------------------------------+
# | Utility Classes                                 |
# +-------------------------------------------------+

SW_NANO  = 1e0
SW_MICRO = 1e-3
SW_MILLI = 1e-6
SW_SEC   = 1e-9
SW_MIN   = 6e-10

class Stopwatch:
	"""
	A simple stopwatch implementation to provide the user
	feedback on how fast a particular piece of code is running.
	"""
	
	def __init__(self):
		self._start = time.monotonic_ns()
		self._stop = time.monotonic_ns()
	
	def start(self):
		self._start = time.monotonic_ns()
	
	def stop(self):
		self._stop = time.monotonic_ns()
	
	def elapsed_ns(self):
		return self._stop - self._start

	def elapsed(self, unit=SW_SEC):
		return unit * self.elapsed_ns()

	def __repr__(self):
		return "%.2fs" % self.elapsed()
	
	def __str__(self):
		return "Stopwatch(%s)" % self.__repr__()


BP_16K = 2 ** 14
BP_32K = 2 ** 15
BP_64K = 2 ** 16

class BlockProcessReporter:
	"""
	A utility to report block processing operations to the user
	while limiting standard output for increased performance.
	"""
	@classmethod
	def file_lines(cls, path, block_size=BP_64K, large_bfsz=True, **kwargs):
		if block_size < 1:
			raise ValueError("block_size must be >= 1")
		
		handle = None
		if large_bfsz:
			handle = f_open_large_read(path, **kwargs)
		else:
			handle = open(path, **kwargs)

		lines = f_line_count_fd(handle)
		handle.close()

		return cls(block_size, lines)

	def __init__(self, block_size, tot_segments, fout=sys.stdout):
		self._fout = fout

		# control data
		self._segment = 0
		self._segments = tot_segments
		self._block = 0
		self._block_size = block_size
		self._blocks = int(math.ceil(tot_segments / block_size))

		# report data
		self.stopwatch = Stopwatch()
		self.message = "Processed Block"
		self.append_percentage = True
		self.append_message = True
		self.append_block_ratio = True
		self.append_stopwatch_time = True
	
	def start(self):
		self.stopwatch.start()

	def ping(self):
		self.stopwatch.stop()
		self._segment += 1
		if self._segment % self._block_size == 0:
			self._block += 1
			self._print()
			self.stopwatch.start()

	def finish(self):
		self.stopwatch.stop()
		if self._segment % self._block_size > 0:
			self._block += 1
			self._print()
	
	def _print(self):
		line = io.StringIO()

		if self.append_percentage:
			line.write("[%5.1f%%] " % (100 * self._block / self._blocks))

		if self.append_message:
			line.write(self.message)

		if self.append_block_ratio:
			line.write(" %d/%d" % (self._block, self._blocks))
		
		if self.append_stopwatch_time:
			line.write(" (%s)" % (repr(self.stopwatch)))

		self._fout.write(line.getvalue())
		self._fout.write('\n')


PR_FORWARDS  = 0x1
PR_BACKWARDS = 0x2
PR_BI        = 0x3

class PrGraphView:

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

class PrLinkError(RuntimeError):

	def __init__(self, message=None):
		super().__init__(message)

class PrLinkMaker:


	def __init__(self, i_tokens, j_tokens):
		self.i_tokens = i_tokens
		self.j_tokens = j_tokens

	@property
	def origin_tokens(self):
		return self.i_tokens
	
	@property
	def inbound_tokens(self):
		return self.j_tokens

	def origin(self, i1, i2, weight=1, direction=PR_FORWARDS, objects=False):
		i1_pos, i2_pos = i1, i2
		if objects:
			i1_pos = d_index_no_except(self.i_tokens, i1)
			i2_pos = d_index_no_except(self.i_tokens, i2)
		return self.monolithic(i1_pos, i2_pos, weight, direction)

	def cross(self, i, j, weight=1, direction=PR_FORWARDS, objects=False):
		i_pos, j_pos = i, j
		if objects:
			i_pos = d_index_no_except(self.i_tokens, i)
			j_pos = d_index_no_except(self.j_tokens, j)
		return self.monolithic(i_pos, len(self.i_tokens) + j_pos, weight, direction)
	
	def inbound(self, j1, j2, weight=1, direction=PR_FORWARDS, objects=False):
		j1_pos, j2_pos = j1, j2
		if objects:
			j1_pos = d_index_no_except(self.j_tokens, j1)
			j2_pos = d_index_no_except(self.j_tokens, j2)
		return self.monolithic(
			len(self.i_tokens) + j1_pos, 
			len(self.i_tokens) + j2_pos, 
			weight, direction)
	
	def monolithic(self, i, j, weight=1, direction=PR_FORWARDS):
		i_size = len(self.i_tokens)
		j_size = len(self.j_tokens)
		size = i_size + j_size
		self._validate_indices(i, j, size, size, seq_sizes=True)
		self._validate_direction(direction)
		return weight, direction, i, j

	@staticmethod
	def _validate_indices(i, j, i_seq, j_seq, seq_sizes=False):
		if not isinstance(i, int) or not isinstance(j, int):
			raise ValueError("indices must be integer types")
		if seq_sizes:
			if i < 0 or i_seq <= i or \
			   j < 0 or j_seq <= j:
				raise ValueError("link targets out of bounds")
		else:
			if i < 0 or len(i_seq) <= i or \
			   j < 0 or len(j_seq) <= j:
				raise ValueError("link targets out of bounds or don't exist")
	
	@staticmethod
	def _validate_direction(direction):
		if direction != PR_BI and \
			direction != PR_FORWARDS and \
			direction != PR_BACKWARDS:
			raise ValueError("direction must be either PageRank.BI, PageRank.FORWARDS, or PageRank.BACKWARDS")

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
		self._graph_view = PrGraphView(self._graph)
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
		for weight, direction, i, j in self._link_func(PrLinkMaker(self._tokens, tokens)):
			Icin, Iwin, Icout, Iwout = (None,) * 4
			Jcin, Jwin, Jcout, Jwout = (None,) * 4
			try:
				_, _, Icin, Iwin, Icout, Iwout = self._graph[i]
				_, _, Jcin, Jwin, Jcout, Jwout = self._graph[j]
			except ValueError:
				raise PrLinkError("corrupted link table, do not modify the graph outside of the PageRank interface")

			if (direction & PR_FORWARDS):
				i_has_ref = j in Icout
				j_has_ref = i in Jcin
				if i_has_ref and j_has_ref:
					raise PrLinkError("nodes are already linked in the forward direction")
				if i_has_ref != j_has_ref:
					raise PrLinkError("corrupted link table, do not modify graph outside of the PageRank interface")
				Icout.append(j)
				Iwout.append(weight)
				Jcin.append(i)
				Jwin.append(weight)
			
			if (direction & PR_BACKWARDS):
				i_has_ref = j in Icin
				j_has_ref = i in Jcout
				if i_has_ref and j_has_ref:
					raise PrLinkError("nodes are already linked in the backward direction")
				if i_has_ref != j_has_ref:
					raise PrLinkError("corrupted link table, do not modify graph outside of the PageRank interface")
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
		rescore_buffer = [ x[1] for x in self._graph ]
		while err >= convthresh:
			for i, node in enumerate(self._graph):
				n_score = self._rank_func(self._graph_view, i)
				err = abs(n_score - rescore_buffer[i])
				rescore_buffer[i] = n_score
				if err < convthresh: # the paper states to stop as soon as one
					break            # evaluation goes below the threshold.
			for i, s in enumerate(rescore_buffer):
				self._graph[i][1] = s
		
		ranked_graph = sorted(self._graph, key=lambda x: x[1], reverse=True)
		ranked_table = []
		for g in ranked_graph: # manual zip to avoid zipping the 4 other lists
			ranked_table.append((g[0], g[1]))
		return ranked_table

	def _expand_graph(self, tokens, defscore):
		for token in tokens:
			self._graph.append([ token, defscore, [], [], [], []])

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
			raise ValueError("the final preprocessor stage must yield an iterable of tokens")