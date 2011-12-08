<TeXmacs|1.0.7.7>

<style|article>

<\body>
  \;

  <doc-data|<doc-title|De-Obfuscating English Text using Hidden Markov
  Models>|<doc-author-data|<author-name|Justin Cullen>|<\author-address>
    December 06, 2011
  </author-address>>>

  <section|Introduction>

  There are many applications that analyze English text by trying to infer
  something from the freqeuncy and type of words used in the document. \ In
  particular, the majority of modern spam filters use a approach based the
  Naive Bayes algorithm to classify an email as either spam or not spam. \ In
  this approach, a training set of emails known to be either spam or not spam
  is used. \ Words that are used frequently in spam emails such as 'viagra'
  or 'offer' will be identified as such. \ On an email that isn't known to be
  spam or not, the Naive Bayes classifier can infer that if the email
  contains one of these words commonly found in spam emails, it is more
  likely to be spam. \ Through examining all of the words in this message,
  the email can be classified one way or another. \ Many spam detection
  systems in use today are a variant of this type of algorithm.

  There is a major vulnerability in these types of spam detection systems, as
  well as in other systems which learn and classify text based on tokenizing
  it into words - text obfuscation. \ In the particular case of detecting
  spam, spammers often try to bypass spam filters by obfuscating words in
  ways that are still readable by humans, but not easily recognizable as
  characteristic 'spam' words by computers. \ Examples of obfuscated words
  are below:

  \;

  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ <block*|<tformat|<table|<row|<cell|<strong|Plain
  Text>>|<cell|<strong|Obfuscated Text>>>|<row|<cell|viagra>|<cell|v-i-a-g-r-a>>|<row|<cell|subscription>|<cell|s*ub
  scri_ption>>|<row|<cell|offer>|<cell|o ff fer>>>>>

  \;

  This poses serious problems for spam detection algorithms. \ With a large
  enough training dataset, such algorithms could indeed learn to recognize
  the obfuscated words as being indicative of spam as well. \ However, there
  are many, many ways spammers can obfuscate words while still maintaining
  human readability. \ Particularly with unicode characters, there are simply
  too many possible obfuscations of any given word to make this a viable
  option.

  One solution for this problem was proposed in a paper by Honglak Lee and
  Andrew Ng titled <em|``Spam Deobfuscation using a Hidden Markov''> model.
  \ In it, they used a hidden markov model with a particular structure to
  convert obfuscated text into its cleartext counterpart. \ In this paper,
  I'll construct a hidden markov model based on their design and test it's
  performance.

  <section|Basic Definitions>

  First I'll give the definition of a hidden Markov model that will be used
  throughout. \ A hidden markov model is defined by N states and M possible
  observations. \ We'll let <math|S> and <math|O> denote the set of states
  and observations respectively. \ The model also must define transition
  probabilities between states. \ <math|P<rsub|s,r>\<assign\>P(X<rsub|t+1>=r\|X<rsub|t>=s),
  \<forall\>s,r\<in\>S> is the probability of a transition from state s to
  state r at any two consecutive times. \ The sequence of states,
  <math|X<rsub|0>,X<rsub|1,\<hdots\>>> is a markov process parameterized by
  these transition probabilities. \ In general, the states of the model are
  hidden and instead we can only view emitted observations <math|o<rsub|t>>.
  \ The model must also specify emission probabilities
  <math|E<rsub|s,o>\<assign\>P(o<rsub|t>=o\|X<rsub|t>=s),
  \<forall\>s\<in\>S,o\<in\>O>. \ Finally, we must define a distribution
  <math|\<pi\>> over the states of the model such that
  <math|\<pi\>(s)\<assign\>P(X<rsub|0>=s)>.

  <section|Structure of the Hidden Markov Model>

  <subsection|States and Transition Probabilities>

  Next, we define the structure of the HMM to be used to deobfuscate text.
  \ We use a dictionary of English words to define the words that are 'valid'
  deobfuscated text. \ The states of the model correspond to characters of
  words in the dictionary. \ We use prefix tree to represent this to reduce
  the total number of states in the model. \ The states are strings <math|s>
  such that <math|s> a prefix of some word in the dictionary (and not the
  empty string). \ We also have one distinguished state to separate words
  which we'll denote <math|\<alpha\>>. \ In the model, <math|P<rsub|s,t>>
  will be nonzero only if <math|s> is a prefix of <math|t> (including if s =
  t). \ We define the transition probabilities with the help of a large
  corpus of text used to compute approximate frequencies of various words and
  prefixes in normal English language usage. \ We define <math|F<rsub|s>> to
  be the number of words in the corpus with <math|s> as a prefix, and
  <math|H<rsub|s>> to be the number of words in the corpus equal to <math|s>,
  and <math|L> to be the number of words in the corpus. \ We then define the
  transition probabilities as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|P<rsub|\<alpha\>,s>>|<cell|=>|<cell|\<eta\>F<rsub|s>/L>>|<row|<cell|P<rsub|e,\<alpha\>>>|<cell|=>|<cell|\<eta\>H<rsub|e>/F<rsub|e>>>|<row|<cell|P<rsub|s,t>>|<cell|=>|<cell|\<eta\>F<rsub|t>/F<rsub|s>>>|<row|<cell|P<rsub|s,s>>|<cell|=>|<cell|1-\<eta\>>>|<row|<cell|P<rsub|\<alpha\>,\<alpha\>>>|<cell|=>|<cell|1-\<eta\>>>>>
  </eqnarray*>

  For all <math|s,t\<in\>S/{\<alpha\>}> and <math|e\<in\>S> such that
  <math|e> is a complete word in the dictionary, and ``end node'' so to
  speak. \ We assume that the corpus for word frequency is large enough for
  all of these definitions to be well defined. \ In these definitions,
  <math|\<eta\>> is a parameter to be learned that controls how often we
  expect self transitions. \ The basic idea with this model of states is
  that, when <math|\<eta\>=1>, sequences of states correspond to sequences of
  words from our dictionary. \ Further, the likelihood of the sequence of
  states corresponding to a particular word is equal to the frequency of it's
  use in our corpus. \ When <math|\<eta\>\<neq\>1>, \ then we allow for the
  possibility that states will be repeated, and this allows us to account for
  character insertions designed to obfuscate words.

  In the original paper, these transition probabilities were modified further
  to allow ``epsilon transitions'' where no observation was emitted to
  account for character deletions. \ However, this changes the model to a
  non-standard HMM model which wasn't compabitible with the HMM library we
  used (Tried writing one in python, however it was too slow to be of any
  use). \ 

  Another difference in our implementation is that in the original paper,
  they used two distinguished states instead of one. \ They used one state to
  signify the start of a word, and another state signifying the end of the
  word, that always transitioned back to the 'start of word' state. \ We
  found this to be unneccessary to our simplified model so didn't include it.

  <subsection|Observations and Emission Probabilities>

  The set of observations in the model correspond to possible characters that
  an adversary could use to represent the characters in the cleartext word.
  \ Thus the observation set <math|O> contains alphanumeric characters
  together with punctuation characters. The approach used here to determine
  emission probabilities differs slightly from the one used in the original
  paper and is somewhat less robust, though easier to implement. \ The
  emission probabilities are parameterized by <math|{\<theta\><rsub|c,o>} for
  c\<in\>{a\<ldots\>z} \<cup\>{<rprime|'> space<rprime|'>}, o\<in\>O>.
  \ <math|\<theta\><rsub|c,o>> roughly corresponds to the probability that
  <math|o> will be used in place of the character <math|c> in a word, however
  it also allows for the possibility of distinct 'filler characters' to be
  inserted into the middle or at the end of a word due to the nonzero
  probability of state self transitions shown in the previous section.
  \ Given these parameters, the emission probabilities are defined as:

  <\eqnarray*>
    <tformat|<table|<row|<cell|E<rsub|w,o>>|<cell|=>|<cell|\<theta\><rsub|c,o>,
    where w\<in\>S, w\<neq\>\<alpha\>, and c is the last character of
    w>>|<row|<cell|E<rsub|\<alpha\>,o>>|<cell|=>|<cell|\<theta\><rsub|space,\<omicron\>>>>>>
  </eqnarray*>

  The parameters <math|\<theta\><rsub|c,o>> were estimated with a maximum
  likelihood estimated on a manually crafted training set. \ We took a
  selection of obfuscated text and manually converted it into its appropriate
  state sequence. \ For instance \|/i_aa_g_r_a would be converted into
  v,-,vi,-,via,-,viag,-,viagr,-,viagra. \ The dashes represent self
  transitions. \ A maximum likelihood estimation was then done to determine
  which parameters maximized the probability of the training data. \ The
  emission probabilities were then set accordingly. \ 

  This method was considered as a possibility for estimating these parameters
  in the original paper, but dismissed because they speculated it wouldn't
  generalize well. \ We decided to test the method anyways to see how well it
  worked in practice. \ In the original paper, they feared that relying on a
  particular training set for estimation of emission probabilities meant that
  only a particular set of substitutions would be recognized. \ Their
  estimation method instead used intrinsic properties of the characters
  involved which they thought would help the method be more robust. \ Its no
  doubt true that this estimation method is more robust, but we found that
  the MLE method works suprisingly effectively despite this.\ 

  A final additional parameter that must be estimate is the state self
  transition probability parameter which we called <math|\<eta\>>. \ This was
  also estimated using a MLE heuristic. \ We simply set it to the ratio of
  dashes in the manually corrected obfuscated text to the total number of
  characters. \ This setting seemed to perform well compared to others
  tested.

  <section|Decoding Method>

  Once the training of the model is complete, decoding an obfuscated text
  using the model is simple. \ We simply run the Viterbi algorithm (The GHMM
  C library with a python wrapper was used) on the obfuscated text to obtain
  the most likely state sequence given our parameters. \ The state sequence
  will consist of consecutive prefixes leading up to the spelling of words,
  separated by <math|\<alpha\>> states separating words. \ We simply take the
  words preceding each alpha, as well as the last state, to be the words of
  the deobfuscated text, separated by spaces where the <math|\<alpha\>>
  states occured. \ For instance, if in decoding the text 'v-i-a-g-r-a ca*t',
  the outputed state sequence was v,v,vi,vi,via,via,viag,viag,viagr,viagr,viagra,<math|\<alpha\>>,c,ca,ca,cat,
  the corresponding cleartext would be 'viagra cat'. \ 

  <section|Performance Evaluation>

  Overall, the performance of our simplified model was somewhat worse than
  the results described in the original paper, as expected. \ However, it
  still performed fairly well. \ The original paper does a preliminary test
  on 60 obfuscated variants of the word 'viagra'. \ We ran our model on the
  59 of these variants (one wasn't used due to our models incompatibility
  with unicode), and it correctly identified 55 out of 59 as being the word
  'viagra', compared to 59 out of 60 for their more robust model. \ 

  On the other more complicated examples, the model didn't perform quite as
  well. \ There are a number of reasons for this, most of which have little
  to do with the model and more to do with the data used to train the model.
  \ Due to the fact that much of this was implemented in python, it isn't as
  fast as it could be so we had to use a smaller dictionary file, frequency
  corpus, and training dataset than we would have otherwise. \ The size of
  the frequency corpus is particularly important. \ Even if a word is in the
  dictionary, if it appears very infrequently (or not at all) in our
  frequency corpus, then it will almost certainly not be produced in
  deobfuscated text since the state transitions will be so unlikely. \ Thus
  using a very large frequency is very important to the accuracy of the
  model.

  Another issue is that our model doesn't handle character deletions. \ This
  certainly detracts from the accuracy since it isn't able to handle
  obfuscations which involve deleting characters. \ For instance 'viaga' will
  never be corrected to 'viagra', almost irregardless of the size of the data
  sets. \ The table below summarizes the results of running our model on the
  other obfuscated text examples used in the original paper. \ The bolded
  phrases are those that were incorrectly decoded.

  \;

  \ <block*|<tformat|<table|<row|<cell|<strong|Obfuscated
  Text>>|<cell|<strong|Decoded Text>>>|<row|<cell|u-n-s-u-s-c-r-i-b-e
  link>|<cell|unsubscribe link>>|<row|<cell|re xe
  finance>|<cell|refinance>>|<row|<cell|<math|m/\<backslash\>
  cromei)d/\<backslash\>>>|<cell|macromedia>>|<row|<cell|gre4t
  pr1ces>|<cell|great <strong|proces>>>|<row|<cell|heyllooo its me
  chelsea>|<cell|hello its me chelsea>>|<row|<cell|veerrryyy
  cheeaapp>|<cell|very cheap>>|<row|<cell|u.n
  sabscjbe>|<cell|<strong|<em|<em|own says
  be<strong|>><strong|>>>>>|<row|<cell|con. tainsn forwa. rdlook. ing sta.
  tements>|<cell|contains forward looking statements>>|<row|<cell|pa, rty for
  sen, ding this re.port.>|<cell|party for sending this
  report>>|<row|<cell|get your u n iversi t y d i pl0 ma>|<cell|get your
  university <strong|i pla>>>|<row|<cell|ree movee below>|<cell|remove
  below>>>>>

  <section|Conclusion>

  In this paper, we presented a simplified model of a Hidden Markov Model
  based system for restoring obfuscated text. \ Despite the simplified model
  and the small training datasets used here compared to those used in the
  original paper, the model performed fairly well. \ Although there are
  somewhat more decoding errors than the more robust model in the original
  paper, I would guess that a simplified model of the type demonstrated here
  works well enough to be used effectively in a spam filtering program,
  assuming it were trained with more data and were written in a faster
  language.

  \;
</body>

<\initial>
  <\collection>
    <associate|font|helvetica>
    <associate|font-base-size|12>
    <associate|language|american>
    <associate|page-type|letter>
    <associate|par-sep|0.3fn>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|3|2>>
    <associate|auto-4|<tuple|3.1|2>>
    <associate|auto-5|<tuple|3.2|3>>
    <associate|auto-6|<tuple|4|4>>
    <associate|auto-7|<tuple|5|4>>
    <associate|auto-8|<tuple|6|5>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Introduction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Basic
      Definitions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Structure
      of the Hidden Markov Model> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1.5fn>|3.1<space|2spc>States and Transition
      Probabilities <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1.5fn>|3.2<space|2spc>Observations and Emission
      Probabilities <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Decoding
      Method> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Performance
      Evaluation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Conclusion>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>