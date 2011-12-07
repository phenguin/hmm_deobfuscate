import pickle
import pdb
#import hmm
from ghmm import *
import numpy
import itertools
from cvxopt import matrix,solvers
from string import punctuation,digits,ascii_lowercase

# Global parameters
eta = 0.9
epsilon = None

# Default corpus to use for english language word frequencies
EL_corpus = '../data/words.txt'
wordlist = '../data/words.txt'
inputs_path = '../data/training_data.txt'
outputs_path = '../data/training_outputs.txt'
states_file = '../data/pickled_states'
observations_file = '../data/pickled_observations'
state_trans_file = '../data/pickled_state_trans'
emission_probs_file = '../data/pickled_emission_probs'


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

def learn_hmm(dict_path = wordlist, training_inputs = inputs_path,
    training_outputs = outputs_path):
  """Build hmm states from words in dict_path"""
  words = open ( dict_path, 'r' )
  states = ['word_start','word_end']
  trans = {'word_end' : {'word_start' : 1},
      'word_start' : {}}
  observations = tuple ( punctuation + ' ' + digits + ascii_lowercase)
  
  # Compute states and state transition probabilities
  for w in words:
    w = w[:-1] # remove EOL char
    for i in range( len(w) ): 
      new = w[:i+1]
      if new not in states:
        states.append(new)
        trans[new] = {}
      if i == 0:
        trans['word_start'][new] = eta * prefix_rel_freq(w[:i+1],'')
      else:
        prev = w[:i]
        trans[prev][new] = eta * prefix_rel_freq(w[:i+1],w[:i])
      if i == len(w) - 1: # last character in a word
        trans[new]['word_end'] = word_rel_freq(w,w[:i])

  for state in trans:
    # self transition probability param'd by eta
    trans[state][state] = 1 - eta
    #for s in states:
      #if s not in trans[state].keys():
        #trans[state][s] = 0

  trans['word_end']['word_end'] = 0 # except for word_end

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

  def to_index(letter,ob):
    return ascii_lowercase.index(letter)* num_obs + observations.index(ob) 
  def from_index(i):
    char_index = i / num_obs
    ob_index = i % num_obs
    return (ascii_lowercase[char_index],observations[ob_index])

  # Construct linear programming problem for cvxopt
  P = matrix(numpy.zeros( (26 * num_obs,26 * num_obs) ),tc='d')
  q = matrix(numpy.zeros(26 * num_obs),tc='d')
  G = matrix(numpy.diag([-1] * (26 * num_obs)),tc='d')
  h = matrix(numpy.zeros(26 * num_obs),tc='d')
  A = numpy.zeros( (26, 26*num_obs) )
  b = matrix(numpy.ones(26),tc='d')
  # construct q
  for o,a in paired:
    if a == '-':
      print last_a,o
      q[to_index(last_a,o)] += 1
    elif a != ' ':
      print a,o
      q[to_index(a,o)] += 1
      last_a = a
  q = -q # Invert since we want maximum not minimum

  # construct A
  for i in range(26):
    for k in range(num_obs):
      A[i][i * num_obs + k] = 1
  A = matrix(A,tc='d')

  # Solve linear program
  sol = list(solvers.qp(P,q,G,h,A,b)['x'])

  # Convert solution into dictionary of emission probabilities
  emission_probs = dict( [(s,{}) for s in states] )
  for s in emission_probs.keys():
    for o in observations:
      emission_probs[s][o] = sol[to_index(s[-1],o)]

  return (tuple(states), observations, trans, emission_probs)

def init_hmm(dict_path = wordlist, training_inputs = inputs_path,
    training_outputs = outputs_path):
  try:
    states = pickle.load(open(states_file))
    observations = pickle.load(open(observations_file))
    trans_probs = pickle.load(open(state_trans_file))
    emission_probs = pickle.load(open(emission_probs_file))
    return (states,observations,trans_probs,emission_probs)
  except IOError, e:
    states,observations,trans_probs,emission_probs = learn_hmm(dict_path,training_inputs,training_outputs)
    pickle.dump(states,open(states_file,'w'))
    pickle.dump(observations,open(observations_file,'w'))
    pickle.dump(trans_probs,open(state_trans_file,'w'))
    pickle.dump(emission_probs,open(emission_probs_file,'w'))
    return (states,observations,trans_probs,emission_probs)

states,observations,trans_probs,emission_probs = init_hmm()
N = len(states)
M = len(observations)

state_indices = dict( [(s,states.index(s)) for s in states] )
index_states = dict( [(states.index(s),s) for s in states] )
obs_indices = dict( [(o,observations.index(o)) for o in observations] )
index_obs = dict( [(observations.index(o),o) for o in observations] )

print 'trying to deobfuscate'
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

def my_viterbi(s):
  a,b = m.viterbi(string_to_obs(s))
  return [index_states[x] for x in a]

#deobfuscate_hmm = hmm.HMM(states,observations,trans_probs,emission_probs)
#answer = deobfuscate_hmm.viterbi(list('beau *** tiful'))
#print answer
#answer = deobfuscate_hmm.viterbi(list('beau *** t iful cat haaas fu * r'))
