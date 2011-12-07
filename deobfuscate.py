import hmm
from string import punctuation

# Global parameters
eta = None

# Default corpus to use for english language word frequencies
EL_corpus = '../data/words.txt'

def prefix_rel_freq(prefix1,prefix2,corpus=EL_corpus):
  """Computes ratio of words in the corpus beginning with prefix1
  compared to the number beginning with prefix2"""
  words_gen1 = (word.strip(punctuation).lower() 
      for line in open(corpus)
      for word in line.split())
  words_gen2 = (word.strip(punctuation).lower() 
      for line in open(corpus)
      for word in line.split())

  def F(prefix,gen):
    return sum(1.0 for w in gen if w.startswith(prefix))

  return F(prefix1,words_gen1) / F(prefix2,words_gen2)

def word_rel_freq(word,prefix,corpus=EL_corpus):
  """Computes ratio of frequency of word in the corpus with frequency
  of words beginning with prefix in the corpus"""
  words_gen1 = (word.strip(punctuation).lower() 
      for line in open(corpus)
      for word in line.split())
  words_gen2 = (word.strip(punctuation).lower() 
      for line in open(corpus)
      for word in line.split())

  return sum(1.0 for w in words_gen1 if w == word) / sum(1.0 for w in
      words_gen2 if
      w.startswith(prefix))

def init_hmm(dict_path):
  """Build hmm states from words in dict_path"""
  words = open ( dict_path, 'r' )
  states = ['word_start','word_end']
  trans = {'word_end' : {'word_start' : 1},
      'word_start' : {}}
  for w in words:
    w = w[:-1] # remove EOL char
    for i in range( len(w) ): 
      new = w[:i+1]
      if new not in states:
        states.append(new)
        trans[new] = {}
      if i == 0:
        trans['word_start'][new] = prefix_rel_freq(w[:i+1],'')
      else:
        prev = w[:i]
        trans[prev][new] = prefix_rel_freq(w[:i+1],w[:i])
      if i == len(w) - 1: # last character in a word
        trans[new]['word_end'] = word_rel_freq(w,w[:i])


  states.sort()
  return (tuple(states), trans)

wordlist = '../data/words.txt'
states, trans_probs = init_hmm(wordlist)

# Testing stuff
print states
print
sortedkeys = sorted( trans_probs.keys() )
print sortedkeys
print
for state in sortedkeys:
  print state, ' - ' ,trans_probs[state]

