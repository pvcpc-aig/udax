import os
import tempfile
import numpy as np
from urllib.request import urlretrieve
from urllib.parse import urlparse
from zipfile import ZipFile
from pathlib import Path, PurePath


sets = {
    'wikipedia-gigaword': {
        'url': 'https://nlp.stanford.edu/data/glove.6B.zip',
        'pattern': 'glove.6B.$dim$d.$ext$'
    },
    'common-small': {
        'url': 'https://nlp.stanford.edu/data/glove.42B.300d.zip',
        'pattern': 'glove.42B.300d.$ext$'
    },
    'common-large': {
        'url': 'https://nlp.stanford.edu/data/glove.840B.300d.zip',
        'pattern': 'glove.840B.300d.$ext$'
    },
    'twitter': {
        'url': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'pattern': 'glove.twitter.27B.$ext$'
    }
}


def patternexpand(pattern, **vars):
    if not isinstance(pattern, str):
        raise ValueError('pattern must be of a string type')

    # check all vars first
    for i, (var, val) in enumerate(vars.items()):
        if not isinstance(var, str):
            raise ValueError(f'at index {i}, not a string key')
    
    # replace all vars
    for var, val in vars.items():
        pattern = pattern.replace(f'${var}$', str(val))
    
    return pattern


def fetch(name=None, workspace=None, dim=None):
    # setup defaults
    if not isinstance(name, str):
        name = 'wikipedia-gigaword'
    if name not in sets:
        raise ValueError('invalid GloVe pre-trained set, see udax.dataset.glove.sets')

    if isinstance(workspace, str):
        workspace = Path(workspace)
    if not isinstance(workspace, PurePath):
        workspace = Path(tempfile.mkdtemp())
    if not workspace.exists():
        workspace.mkdir(parents=True)
    
    if not isinstance(dim, int):
        dim = 100

    # prepare for load
    info = sets[name]
    pattern = info['pattern']
    url_raw = info['url']
    url_parsed = urlparse(url_raw)
    target_name = patternexpand(pattern, dim=dim, ext='txt')

    zip_name = os.path.basename(url_parsed.path)
    zip_file = workspace.joinpath(zip_name)
    zip_dir = workspace.joinpath(zip_name + '.d')
    target_file = zip_dir.joinpath(target_name)

    def _download():
        nonlocal name, url_raw, zip_file

        def _report(count, block_size, file_size):
            nonlocal name
            percent = 100 * count * block_size / file_size
            if percent >= 100:
                print('\r%s   Done' % (name))
            else:
                print('\r%s %4.1f%%' % (name, percent), end='')
        
        urlretrieve(url_raw, zip_file, _report)

    def _extract():
        nonlocal _download, zip_file, zip_dir

        # prereq
        if not zip_file.exists():
            _download()
        if not zip_dir.exists():
            zip_dir.mkdir(parents=True)
        
        ZipFile(zip_file).extractall(zip_dir)

    def _load():
        nonlocal _extract, target_file
        
        # prereq
        if not target_file.exists():
            _extract()
        
        # check again for match
        if not target_file.exists():
            raise ValueError(f'datafile {target_name} not found')
        
        mapping = {}
        with open(target_file, mode='r', buffering=65535) as fin:
            for line in fin:
                split = line.split()
                word = split[0]
                vector = np.array([ float(x) for x in split[1:] ], dtype=np.float32)
                mapping[word] = vector
        
        return mapping
    
    return _load()