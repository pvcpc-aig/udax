"""
A specialized module to handle reading emails
in the form of HTTP responses.
"""
import sys
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser

from .strutil import STR_EXTRANEOUS
from .strutil import surjective_map


class ConcatParser(HTMLParser):
    
    def __init__ (self, stopwords=[], errcb=None):
        super(ConcatParser, self).__init__()
        self.errcb = errcb
        self.words = []
        self.parts = []
        
    def handle_starttag(self, tag, attrs):
        pass
    
    def handle_endtag(self, tag):
        pass
    
    def handle_data(self, data):
        words = surjective_map(data.lower(), STR_EXTRANEOUS, ' ').split()
        self.words.extend(words)
        self.parts.append(data)
        
    def error(self, message):
        if self.errcb is not None:
            self.errcb(message)

    @property
    def current_email(self):
        return ' '.join(self.parts)


class HttpEmail:

    def __init__(self, path, stopwords=[], errcb=None):
        self.path = Path(path)
        self.parser = ConcatParser(errcb)
        self.body = None
        self.words = None           # standalone list of words
        self.stopwords = stopwords
        self.word_table = {}        # map <word> -> (<count>, <relative-freq>) 

        self._load_email()
        self._gen_word_frequencies()

    def print_word_table(self, fd=sys.stdout): 
        for word, statistic in self.word_table.items():
            fd.write(f"{word} {statistic[0]} {statistic[1]}\n")

    def _gen_word_frequencies(self):
        total_words = len(self.words)
        if total_words == 0:
            return

        for word, count in Counter(self.words).items():
            relative_freq = count / total_words
            self.word_table[word] = (count, relative_freq)

    def _load_email(self):
        encoded_message = []
        with self.path.open(mode="rb") as handle:
            encoded_message = self._filter_message_headers(handle.readlines())

        sender,       \
        content_type, \
        boundary,     \
        split_indices = self._extract_metadata(encoded_message)
        charset = "latin-1"

        if len(split_indices) > 2:
            # If there is more than one email in the single
            # file, we must get rid of the top HTTP headers
            # once more.
            for i in range(len(split_indices) - 1):
                x = split_indices[i]
                y = split_indices[i + 1]
                j = x + 1
                while j < y:
                    line = encoded_message[j].decode(charset).strip()
                    j += 1
                    if len(line) == 0:
                        break
                lineset = encoded_message[j:y]
                self.parser.feed(b"".join(lineset).decode(charset))
            self.parser.close()
        else:
            try:
                begin = split_indices[0]
                end = split_indices[1]
                lineset = encoded_message[begin:end]
                self.parser.feed(b"".join(lineset).decode(charset))
                self.parser.close()
            except:
                # NOTE(max): do we even need this anymore?
                pass
        self.words = self.parser.words
        self.body = self.parser.current_email

    def _filter_message_headers(self, encoded_message):
        new_encoded_message = []
        i = 0
        while i < len(encoded_message):
            sline = encoded_message[i].decode("latin-1")
            if "-----Original Message-----" in sline:
                while i < len(encoded_message) and len(encoded_message[i].decode("latin-1").strip()) != 0:
                    i += 1
            if i < len(encoded_message):
                new_encoded_message.append(encoded_message[i])
            i += 1
        return new_encoded_message
            
    def _extract_metadata(self, encoded_message):
        sender = ""
        content_type = ""
        boundary = None
        body_index = None
        
        # [0] = body start
        # [-1] = file end
        split_indices = []
        
        for i, line in enumerate(encoded_message):
            sline = line.decode('latin-1')
            if str.encode("From: ") in line:
                sender = sline[6:].strip()
            if content_type == '' and str.encode("Content-Type: ") in line:
                content_type = sline[14:sline.find(';')]
            if 'boundary' in sline and sline.find('=') != -1:
                boundary = sline[sline.find('=')+1:][1:-2]
                continue
            if boundary and boundary in sline:
                split_indices.append(i)
            if not body_index and sline.strip() == "":
                body_index = i
        if len(split_indices) == 0:
            split_indices.append(body_index)
        split_indices.append(len(encoded_message)-1)
        return (sender, content_type, boundary, split_indices)

