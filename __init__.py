import io
import os
import sys
import math
import time
import string
from collections.abc import Iterable
from pathlib import PurePath
	
# +-------------------------------------------------+
# | Utility Functions:                              |
# |                                                 |
# | - Prefix meanings:                              |
# |   - s_*   : string utilities                    |
# |   - f_*   : file/stream utilities               |
# |   - csv_* : CSV format utilities                |
# |                                                 |
# +-------------------------------------------------+

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


def s_rempunct(p_string):
	"""
	Removes punctuation characters from the given string. Punctuation
	characters are defined by `string.punctuation` from the standard
	library.

	:param p_string
		The string whose punctuation to remove.
	"""
	return s_reppunct(p_string, '')


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


