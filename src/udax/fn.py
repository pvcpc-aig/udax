"""
Functional programming capabilities.
"""
from collections.abc import Iterable, Mapping, Sequence


class Operator:

    def __init__(self):
        self.name = self.__class__.__name__
        self._source = iter(())
        if '__iter__' not in dir(self.__class__):
            raise TypeError(f'{self.name} must implement __next__')
    
    @property
    def source(self):
        return self._source
    
    @source.setter
    def source(self, new):
        if not isinstance(new, Iterable):
            raise ValueError('Operator source must be an iterable type')
        self._source = iter(new)

    def __iter__(self):
        return self
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name


class Collect(Operator):

    def __init__(self):
        super(Collect, self).__init__()
        self._retainer = []

    def __iter__(self):
        try:
            while True:
                self._retainer.append(next(self.source))
        except StopIteration:
            self._retainer = iter(self._retainer)
            return self

    def __next__(self):
        return next(self._retainer)


class Print(Operator):

    def __init__(self, prefix=None, suffix=None):
        super(Print, self).__init__()
        self.prefix = prefix
        self.suffix = suffix
    
    def __next__(self):
        item = next(self.source)
        if self.prefix is not None:
            print(self.prefix, end='')
        print(item, end='')
        if self.suffix is not None:
            print(self.suffix, end='')
        print()
        return item


class Head(Operator):

    def __init__(self, count=10):
        super(Head, self).__init__()
        if not isinstance(count, int):
            raise ValueError('count must be an integer type')
        self.count = max(0, count)

    def __next__(self):
        if self.count <= 0:
            raise StopIteration()
        self.count -= 1
        return next(self.source)


class Tail(Operator):

    def __init__(self, count=10):
        super(Tail, self).__init__()
        if not isinstance(count, int):
            raise ValueError('count must be an integer type')
        self.count = max(0, count)
        self._buffer = []

    def __iter__(self):
        try:
            while True:
                self._buffer.append(next(self.source))
                if len(self._buffer) >= (2 * self.count):
                    self._buffer = self._buffer[-self.count:]
        except StopIteration:
            size = min(self.count, len(self._buffer))
            self._buffer = iter(self._buffer[-size:])
            return self

    def __next__(self):
        return next(self._buffer)


class Filter(Operator):

    def __init__(self, predicate):
        super(Filter, self).__init__()
        if not callable(predicate):
            raise ValueError('predicate function must be callable returning bool')
        self.predicate = predicate

    def __next__(self):
        item = next(self.source)
        while not self.predicate(item):
            item = next(self.source)
        return item


class Sorted(Operator):

    def __init__(self, key=None, reverse=False):
        super(Sorted, self).__init__()
        self.key = key
        self.reverse = reverse
        self._sorted = self.source

    def __iter__(self):
        self._sorted = iter(sorted(
            self.source, 
            key=self.key, 
            reverse=self.reverse
        ))
        return self

    def __next__(self):
        return next(self._sorted)


class Map(Operator):

    def __init__(self, func):
        super(Map, self).__init__()
        if not callable(func):
            raise ValueError('mapping function must be callable')
        self.func = func

    def __next__(self):
        return self.func(next(self.source))


class FlatMap(Operator):

    def __init__(self, func):
        super(FlatMap, self).__init__()
        if not callable(func):
            raise ValueError('mapping function must be callable returning iterable')
        self.func = func
        self._flat = iter(())

    def __next__(self):
        while True:
            try:
                return next(self._flat)
            except StopIteration:
                self._flat = iter(self.func(next(self.source)))


class KeyMap(Operator):

    def __init__(self, mapping, graceful=True):
        super(KeyMap, self).__init__()
        if not isinstance(mapping, Mapping):
            raise ValueError('mapping object must be of Mapping type')
        self.graceful = graceful
        self.mapping = mapping
    
    def __next__(self):
        item = next(self.source)
        if self.graceful:
            while item not in self.mapping:
                item = next(self.source)
        return self.mapping[item]


def stream(iterable, operators):
    # defaults
    if not isinstance(iterable, Iterable):
        raise ValueError('iterable must be iterable')
    if not isinstance(operators, Sequence):
        raise ValueError('operators must be a sequence of Operator types')

    # validate operators
    invalid = list(filter(lambda x: not isinstance(x, Operator), operators))
    if len(invalid) > 0:
        raise ValueError('invalid operator types:', ', '.join(invalid))
    
    # push iterable into operator sequence as source
    operators.insert(0, iterable)
    
    # link operators
    for i in range(len(operators)-1):
        source = operators[i]
        dest = operators[i+1]
        dest.source = source

    return operators[-1]


def sideeffect(iterable, operators):
    """
    See `udax.fn.stream()`
    """
    for _ in stream(iterable, operators):
        pass