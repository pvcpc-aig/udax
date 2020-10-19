import os
import sys
import math
import time
from pathlib import PurePath


def f_line_count(path, p_buffering=-1, p_encoding=None, p_errors=None, p_newline=None):
	lines = 0
	with os.open(path, buffering=p_buffering, encoding=p_encoding, errors=p_errors, newline=p_newline) as handle:
		for _ in handle:
			lines += 1
	return lines


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


class BlockProcessReporter:
	"""
	A utility to report block processing operations to the user
	while limiting standard output for increased performance.
	"""
	def __init__(self, block_size, tot_segments, fout=sys.stdout):
		self._fout = fout

		# control data
		self._segment = 0
		self._segments = tot_segments
		self._block = 0
		self._block_size = block_size
		self._blocks = int(math.ceil(tot_segments / block_size))

		# report data
		self.message = "Processed Block"
		self.append_block_ratio = True
		self.prepend_percentage = True
	
	def ping(self):
		self._segment += 1
		if self._segment % self._block_size == 0:
			self._block += 1
			self._print()

	def finish(self):
		if self._segment % self._block_size > 0:
			self._block += 1
			self._print()
	
	def _print(self):
		parts = (
			"[%5.1f%%]" % (100 * self._block / self._blocks) if self.prepend_percentage else "",
			self.message,
			f"{self._block}/{self._blocks}" if self.append_block_ratio else ""
		)
		msgstr = ' '.join(parts)
		self._fout.write(f"{msgstr}\n")
