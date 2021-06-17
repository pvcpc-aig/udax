"""
Functional programming capabilities.
"""
from collections.abc import Iterable, Sequence, Mapping


# administrative
class StreamConstants:

    # special return item from any Operator to signal item removal
    REMOVE = object()

    # special return item from terminal Operators to signal end of stream.
    EOS = object()


class StreamError(RuntimeError):

    def __init__(self, *args, **kwargs):
        super(StreamError, self).__init__(*args, **kwargs)


# operator definition
class Operator:

    def __init__(self, terminal=False):
        self.name = self.__class__.__name__
        self.terminal = terminal
        if not callable(self.__class__):
            raise StreamError(f'{self.name} operator must implement callable')


class OperatorResult:

    # special result used by terminal operators to indicate end of stream.
    EOS = object()

    # special result used by any operator to ignore the result.
    IGNORE = object()

    def __init__(self, single=None, multiple=None):
        self.single = single
        self.multiple = multiple


class OperatorChain:

    @staticmethod
    def verify_oplist(oplist):
        if not isinstance(oplist, Sequence):
            raise StreamError('oplist must be a sequence of fn.Operator objects')

        invalid = list(filter(lambda x: not isinstance(x, Operator), oplist))

        if len(invalid) > 0:
            raise StreamError(
                'following do not adhere to fn.Operator spec: ' + ', '.join(invalid))

    @staticmethod
    def parse_oplist(oplist):
        # verify the list only contains valid operators
        OperatorChain.verify_oplist(oplist)

        # continue to find all operator chains
        i = 0
        limit = len(oplist)
        chain = []
        while i < limit:
            # parse single operator chain
            functions = []
            terminal = None
            while i < limit:
                op = oplist[i]
                i += 1
                if op.terminal:
                    terminal = op
                    break
                functions.append(op)
            chain.append(OperatorChain(functions, terminal))
        
        return chain

    def __init__(self, functions=None, terminal=None):
        self.functions = functions if isinstance(functions, Iterable) else []
        self.terminal = terminal if terminal is not None else Collect()


# terminal operators
class Collect(Operator):

    def __init__(self):
        super(Collect, self).__init__(terminal=True)
        self.items = []
    
    def __call__(self, item):
        self.items.append(item)
    
    def __iter__(self):
        return iter(self.items)


class Head(Operator):

    def __init__(self, count=10):
        super(Head, self).__init__(terminal=True)
        if not isinstance(count, int):
            raise StreamError('Head operator requires an integer count')
        self.count = count
        self.items = []

    def __call__(self, item):
        if len(self.items) >= self.count:
            return StreamConstants.EOS
        self.items.append(item)
    
    def __iter__(self):
        return iter(self.items)


class Tail(Operator):

    def __init__(self, count=10):
        super(Tail, self).__init__(terminal=True)
        if not isinstance(count, int):
            raise StreamError('Tail operator requires an integer count')
        self.count = count
        self.block_old = []
        self.block_new = []

    def __call__(self, item):
        if len(self.block_new) >= self.count:
            self.block_old = self.block_new
            self.block_new = []
        self.block_new.append(item)
    
    def __iter__(self):
        fused = self.block_old + self.block_new
        limit = min(len(fused), self.count)
        return iter(fused[-limit:])


class FlatMap(Operator):

    def __init__(self, mapping):
        super(FlatMap, self).__init__(terminal=True)
        if not callable(mapping):
            raise StreamError('FlatMap operator requires a callable function returning Iterable')
        self.mapping = mapping
        self.items = []
    
    def __call__(self, item):
        self.items.extend(self.mapping(item))
    
    def __iter__(self):
        return iter(self.items)


class Sorted(Operator):

    def __init__(self, key=None, reverse=False):
        super(Sorted, self).__init__(terminal=True)
        self.key = key
        self.reverse = reverse
        self.items = []

    def __call__(self, item):
        self.items.append(item)
    
    def __iter__(self):
        return iter(sorted(self.items, key=self.key, reverse=self.reverse))


# functional operators
class SideEffect(Operator):

    def __init__(self, function):
        super(SideEffect, self).__init__()
        if not callable(function):
            raise StreamError('SideEffect operator requires function')
        self.function = function
    
    def __call__(self, item):
        self.function(item)
        return item


class Print(SideEffect):

    @staticmethod
    def create_function(prefix, suffix):
        def function(item):
            if prefix:
                print(prefix, end='')
            print(item, end='')
            if suffix:
                print(suffix, end='')
            print()
        return function

    def __init__(self, prefix=None, suffix=None):
        super(Print, self).__init__(Print.create_function(prefix, suffix))
    

class Filter(Operator):

    def __init__(self, predicate):
        super(Filter, self).__init__()
        if not callable(predicate):
            raise StreamError('Filter operator requires a callable predicate')
        self.predicate = predicate
    
    def __call__(self, item):
        if not self.predicate(item):
            return StreamConstants.REMOVE
        return item


class Map(Operator):

    def __init__(self, mapping):
        super(Map, self).__init__()
        if not callable(mapping):
            raise StreamError('Map operator requires a callable function')
        self.mapping = mapping

    def __call__(self, item):
        return self.mapping(item)


class KeyMap(Operator):

    def __init__(self, dictionary, graceful=True):
        super(KeyMap, self).__init__()
        if not isinstance(dictionary, Mapping):
            raise StreamError('KeyMap operator requires a Mapping object')
        self.dictionary = dictionary
        self.graceful = graceful
    
    def __call__(self, item):
        if self.graceful and item not in self.dictionary:
            return StreamConstants.REMOVE
        return self.dictionary[item]


# stream implementations
class Stream:

    def __init__(self, 
        target=None, 
        autohint=True,
        objectify=False, 
    ):
        if autohint and (
            isinstance(target, str)
        ):
            objectify = True
        
        if objectify or not isinstance(target, Iterable):
            self.target = iter([ target ])
        else:
            self.target = iter(target)

    def apply(self, *ops):
        chains = OperatorChain.parse_oplist(ops)
        for chain in chains:
            for item in self.target:
                for func in chain.functions:
                    item = func(item)
                    if item is StreamConstants.REMOVE:
                        break
                if item is StreamConstants.REMOVE:
                    continue
                if chain.terminal(item) is StreamConstants.EOS:
                    break
            self.target = iter(chain.terminal)
        return self.target