import string


STR_EXTRANEOUS = string.punctuation + string.digits + "\t\r\n"


def surjective_map(subject, domain, target):
    """
    Maps all characters in the |domain| string to
    a single character |target|, hence a surjective
    mapping, of the |subject| string.

    ex. subject = '{a,b,c}!'
        domain = string.punctuation
        target = '.'

        returns '.a.b.c..'

    I couldn't find anything in the python standard
    library that does exactly this, but I may be missing
    something ¯\_(ツ)_/¯ (btw regex is about 4x slower)
    """
    buf = list(subject)
    for i, c in enumerate(buf):
        if c in domain:
            buf[i] = target
    return ''.join(buf)


def wordify_wordlist(subject, uppercase=False):
	"""
	Takes the `subject` string, transforms all of the letters into
	lowercase (or uppercase), removes all punctuation as defined by 
	`string.punctuation` in the python standard library, and splits 
	the string into words.

	ex. subject = "Please, for the greater good--if you will--wordify this!"
		uppercase = False

		returns [ "please", "for", "the", "greater", "good", "if", "you", "will", "wordify", "this" ]
	
	If `uppercase` is specified, of course, each word returned in the list
	will be uppercase.
	"""
	return surjective_map(subject.upper() if uppercase else subject.lower(), STR_EXTRANEOUS, ' ').split()


def wordify(subject, uppercase=False):
	"""
	Takes the `subject` string, transforms all of the letters into
	lowercase (or uppercase), removes all punctuation as defined by 
	`string.punctuation` in the python standard library, and joins
	the individual words by a single space into a single string. 

	ex. subject = "Please, for the greater good--if you will--wordify this!"
		uppercase = True

		return "PLEASE FOR THE GREATER GOOD IF YOU WILL WORDIFY THIS"
	
	If `uppercase` is specified, of course, the returned string will be
	in uppercase form.
	"""
	return ' '.join(wordify_wordlist(subject, uppercase))