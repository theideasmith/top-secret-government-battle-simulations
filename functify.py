import re
import sys

#Matches dictionary definitions of integer or string values
#                Match key            String Values         Numerical Values   Exponents
# kwargs_words = "([A-z]+[A-z0-9_]*)\ *\=(?:\"|\')(.+)(?:\"|\')"
kwargs_words = "([A-z]+[A-z0-9_]*)\ *\=(.+)"
kwargs_numbers = "([A-z]+[A-z0-9_]*)\ *\=(\-*[0-9]*\.*[0-9]+(?:e\-*\d+)*)"

re_kwargs_words = re.compile(kwargs_words)
re_kwargs_numbers = re.compile(kwargs_numbers)

args= "(\-*[0-9]*\.*[0-9]+)(e\-*\d+)|(.+)"

re_args = re.compile(args)


def numread(string):
  try:
    return int(string)
  except ValueError:
    try:
        return float(string)
    except ValueError:
        return string

def matchany_or_last(regex, string):
    match_first = regex +"\ *\,\ *"
    match_any = regex
    match_last =  "\ *\,_\ *" + regex

    re_match_first = re.compile(match_first)
    re_match_any = re.compile(match_any)
    re_match_last = re.compile(match_last)

    firsts = re_match_first.findall(string)
    string = re_match_first.sub('', string)

    anys = re_match_any.findall(string)
    string = re_match_any.sub('', string)

    lasts = re_match_last.findall(string)
    string = re_match_last.sub('', string)

    total = []
    total.extend(firsts)
    total.extend(anys)
    total.extend(lasts)
    return total, string

def _collectkwargs(argv):
  words, string = matchany_or_last(kwargs_words, argv)
  numbers, _ = matchany_or_last(kwargs_numbers, string)
  total = []
  total.extend(words)
  total.extend(numbers)
  ret = dict(total)
  for k in ret.keys():
      ret[k] = numread(ret[k])
  return ret, string

def collectkwargs(argv):
    m, _ = _collectkwargs(argv)
    return m

def collectargs(argv):
  total = re_args.findall(argv)
  total = map(lambda x: numread(''.join(list(x))), total)
  return total

def parseargs(argc):
  kwargs, string = _collectkwargs(argc)
  args = collectargs(string)
  return kwargs, args

def functify(f, string=''):
  if string=='' and len(sys.argv) >= 1:
      string = sys.argv[1:]
  else:
    return InputError("No string passed")
  """
  This is a micro argparse for turning any
  python function into a bash script.

  For scientists who are busy and need
  to throw something together quickly

  Handles strings and integers
  as kwargs and args
  """

  results=map(parseargs, string)
  kwargs = {}
  args = []
  for kwarg, arg in results:
    kwargs.update(kwarg)
    args.extend(arg)
  f(*args, **kwargs)
