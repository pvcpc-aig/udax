import string


# redefine string constants as sets for performance
ascii_letters = set(string.ascii_letters)
ascii_lowercase = set(string.ascii_lowercase)
ascii_uppercase = set(string.ascii_uppercase)
digits = set(string.digits)
hexdigits = set(string.hexdigits)
octdigits = set(string.octdigits)
punctuation = set(string.punctuation)
printable = set(string.printable)
whitespace = set(string.whitespace)

# predefined translation tables
tab_punct = str.maketrans('', '', string.punctuation)
tab_space = str.maketrans('', '', string.whitespace)


def rempunct(target):
    return str.translate(target, tab_punct)


def remspace(target):
    return str.translate(target, tab_space)