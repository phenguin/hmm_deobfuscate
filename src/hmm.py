##################################################
# Title: hmm.py
# Descr: Silly slow HMM viterbi algorithm implementation
#        not actually used in deobfuscation, but I tried so I'll leave
#        it here
# Author: Justin Cullen
##################################################
import pdb

class HMM(object):
# """Implmentation of a Hidden Markov Model"""
  def __init__(self, states, obs, trans_probs, output_probs):
    self.states = states # States
    self.num_states = len(self.states)
    self.obs = obs # Possible observations

    # Dictionary of nested dictionaries where
    # self.trans_probs[state1][state2] is the probability of
    # transitioning from state1 to state2 on a time step
    self.trans_probs = trans_probs 

    # Dictionary of nested dictionaries where
    # self.output_probs[state][observation] is the probability of
    # observing observation given that the HMM is in state 'state'
    self.output_probs = output_probs 

  def viterbi(self,observed):
    # initialize
    start_dist = dict([(s,1/float(self.num_states)) for s in self.states])
    T = len(observed)
    V = [{}]
    P = [{}]

    # Base case for DP algorithm
    for k in self.states:
      V[0][k] = self.output_probs[k][observed[0]] * start_dist[k]
      P[0][k] = k

    # Compute V[self.num_states] with bottom-up DP
    for t in range(1,T):
      V.append({})
      P.append({})
      for k in self.states:
        cur_max = 0
        for y in self.states:
          val = self.trans_probs[y].get(k,0) * V[t-1][y] 
          if val > cur_max:
            cur_max = val
            best_state = y
        V[t][k] = cur_max * self.output_probs[k][observed[t]]
        P[t][k] = best_state

    # Now compute the most likely path
    temp_max = 0
    for y in self.states:
      if V[T-1][y] > temp_max:
        best = y
        temp_max = V[T-1][y]

    state_seq = [best]
    for t in range(T-2,-1,-1):
      state_seq.insert(0,P[t+1][state_seq[0]])

    return state_seq
    
# Test data----------------------------------------
#states = ('rainy','sunny')
#observations = ('walk','shop','clean')

#transitions = {'rainy' : {'rainy' : 0.7,'sunny' : 0.3},
    #'sunny' : {'rainy' : 0.4,'sunny' : 0.6}}

#obs_probs = {'rainy' : {'walk' : 0.1,'shop' : 0.4,'clean' : 0.5},
    #'sunny' : {'walk' : 0.6,'shop' : 0.3,'clean' : 0.1}}

#weather_HMM = HMM(states,observations,transitions,obs_probs)

#observed = ['walk','shop','clean','clean','walk','walk','walk','clean']
#initial_dist = {'rainy' : 0.5,'sunny' : 0.5}

#weather_HMM.viterbi(observed,initial_dist)
