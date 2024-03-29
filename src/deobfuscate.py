##################################################
# Title: deobfuscate.py
# Descr: Hidden markov model based approach to text deobfuscation
# Author: Justin Cullen
##################################################
import pickle
import pdb
#import hmm
from ghmm import *
import numpy
import itertools
from cvxopt import matrix,solvers
from string import punctuation,digits,ascii_lowercase

# Global parameters
eta = 0.7
tau = None

# Default corpus to use for english language word frequencies
EL_corpus = '../data/freq_corpus.txt'
wordlist = '../data/words.txt'
inputs_path = '../data/training_data.txt'
outputs_path = '../data/training_outputs.txt'
states_file = '../data/pickled_states'
observations_file = '../data/pickled_observations'
state_trans_file = '../data/pickled_state_trans'
emission_probs_file = '../data/pickled_emission_probs'

prefix_count = {}
word_count = {}

def init_counts(corpus=EL_corpus):
  def prefixes(w):
    return (w[:i] for i in range(len(w)+1))

  word_gen = (word.strip(punctuation).lower()
      for line in open(corpus)
      for word in line.split())

  for w in word_gen:
    for p in prefixes(w):
      prefix_count[p] = prefix_count.get(p,0) + 1.0
    word_count[w] = word_count.get(w,0) + 1.0

  return prefix_count,word_count

def prefix_rel_freq(prefix1,prefix2,corpus=EL_corpus):
  """Computes ratio of words in the corpus beginning with prefix1
  compared to the number beginning with prefix2"""
  try:
    return prefix_count.get(prefix1,0) / prefix_count.get(prefix2,0)
  except ZeroDivisionError, e:
    return 0

def word_rel_freq(word,prefix,corpus=EL_corpus):
  """Computes ratio of frequency of word in the corpus with frequency
  of words beginning with prefix in the corpus"""
  try:
    return word_count.get(word,0) / prefix_count.get(prefix,0)
  except ZeroDivisionError, e:
    return 0

def learn_hmm(dict_path = wordlist, training_inputs = inputs_path,
    training_outputs = outputs_path):
  """Build hmm states from words in dict_path"""
  init_counts()
  words = open ( dict_path, 'r' )
  states = set(['word_start'])
  trans = {'word_start' : {}}
  observations = tuple ( punctuation + ' ' + digits + ascii_lowercase)
  
  # Compute states and state transition probabilities
  for w in words:
    w = w.lower()
    w = w[:-1] # remove EOL char
    for i in range( len(w) ): 
      new = w[:i+1]
      if new not in states:
        states.add(new)
        trans[new] = {}
        if i == 0:
          trans['word_start'][new] = eta * prefix_rel_freq(w[:i+1],'')
        else:
          prev = w[:i]
          trans[prev][new] = eta * prefix_rel_freq(w[:i+1],w[:i])
        if i == len(w) - 1: # last character in a word
          trans[new]['word_start'] = word_rel_freq(w,w[:i])

  for state in trans:
    trans[state][state] = 1 - eta
  states = list(states)
  num_states = len(states)
  num_obs = len(observations)

  # Compute observation emission probabilities via MLE
  observed_chars = (char.lower()
      for line in open(training_inputs)
      for char in line[:-1])
  true_chars = (char.lower()
      for line in open(training_outputs)
      for char in line[:-1])
  paired = itertools.izip(observed_chars,true_chars)

  def c_to_i(s):
    if s == 'word_start':
      return len(ascii_lowercase)
    else:
      return ascii_lowercase.index(s)

  def c_from_i(i):
    if i == len(ascii_lowercase):
      return 'word_start'
    else:
      return ascii_lowercase[i]

  def to_index(letter,ob):
    return c_to_i(letter) * num_obs + observations.index(ob) 
  def from_index(i):
    char_index = i / num_obs
    ob_index = i % num_obs
    return (c_from_i(char_index),observations[ob_index])

  # Construct linear programming problem for cvxopt
  P = matrix(numpy.zeros( (27 * num_obs,27 * num_obs) ),tc='d')
  q = matrix(numpy.zeros(27 * num_obs),tc='d')
  G = matrix(numpy.diag([-1] * (27 * num_obs)),tc='d')
  h = matrix(numpy.zeros(27 * num_obs),tc='d')
  A = numpy.zeros( (27, 27*num_obs) )
  b = matrix(numpy.ones(27),tc='d')
  # construct q
  for o,a in paired:
    if o not in observations: continue
    if a == '-':
      q[to_index(last_a,o)] += 1
    elif a != ' ':
      if a not in ascii_lowercase: continue
      q[to_index(a,o)] += 1
      last_a = a
    else:
      q[to_index('word_start',o)] += 1
      last_a = 'word_start'
  q = -q # Invert since we want maximum not minimum

  # construct A
  for i in range(27):
    for k in range(num_obs):
      A[i][i * num_obs + k] = 1
  A = matrix(A,tc='d')

  # Solve linear program
  sol = list(solvers.qp(P,q,G,h,A,b)['x'])

  # Convert solution into dictionary of emission probabilities
  emission_probs = dict( [(s,{}) for s in states] )
  for s in emission_probs.keys():
    for o in observations:
      if s != 'word_start':
        emission_probs[s][o] = sol[to_index(s[-1],o)]
      else:
        emission_probs[s][o] = sol[to_index(s,o)]

  return (tuple(states), observations, trans, emission_probs)

def init_hmm(dict_path = wordlist, training_inputs = inputs_path,
    training_outputs = outputs_path):
  print 'Initializing states and state transitions'
  try:
    states = pickle.load(open(states_file))
    observations = pickle.load(open(observations_file))
    trans_probs = pickle.load(open(state_trans_file))
    emission_probs = pickle.load(open(emission_probs_file))
    print 'Cached verions exist - using those'
    return (states,observations,trans_probs,emission_probs)
  except IOError, e:
    states,observations,trans_probs,emission_probs = learn_hmm(dict_path,training_inputs,training_outputs)
    pickle.dump(states,open(states_file,'w'))
    pickle.dump(observations,open(observations_file,'w'))
    pickle.dump(trans_probs,open(state_trans_file,'w'))
    pickle.dump(emission_probs,open(emission_probs_file,'w'))
    return (states,observations,trans_probs,emission_probs)
# Initializei HMM
states,observations,trans_probs,emission_probs = init_hmm()
N = len(states)
M = len(observations)

# Convert to form that GHMM can use
state_indices = dict( [(s,i) for s,i in zip(states,range(N))] )
index_states = dict( [(i,s) for s,i in zip(states,range(N))] )
obs_indices = dict( [(o,i) for o,i in zip(observations,range(M))] )
index_obs = dict( [(i,o) for o,i in zip(observations,range(M))] )

print "Initializing HMM solver"
A = [[trans_probs[index_states[s1]].get(index_states[s2],0) for s2 in
  range(N)]
  for s1 in range(N)]

B = [[emission_probs[index_states[s]].get(index_obs[o],0) for o in
  range(M)]
  for s in range(N)]

pi = [0] * N
pi[state_indices['word_start']] = 1
sigma = IntegerRange(0,M)

m = HMMFromMatrices(sigma,DiscreteDistribution(sigma),A,B,pi)

def string_to_obs(s):
  return EmissionSequence(sigma, [obs_indices[o] for o in list(s)])

# Wrapped function to simply deobfuscate a given string using the
# parameters specified in the model.  This is the main function of the
# module
def deobfuscate(s):
  a,b = m.viterbi(string_to_obs(' '+s))
  def split_list(ls,split_on):
    result = []
    while split_on in ls:
      i = ls.index(split_on)
      temp = ls[:i]
      if temp != []:
        result.append(temp[-1])
      ls = ls[i+1:]
    if ls != []: result.append(ls[-1])
    return result
  return ' '.join(split_list([index_states[x] for x in
    a],'word_start'))


