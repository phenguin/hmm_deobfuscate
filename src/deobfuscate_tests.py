##################################################
# Title: deobfuscate_tests.py
# Descr: Tests for deobfuscate.py
# Author: Justin Cullen
##################################################
from deobfuscate import deobfuscate

# Run deobfuscate(string) on various obfuscated test strings... that's
# all

training_data = [x[:-1] for x in open('../data/training_data.txt').readlines()]
training_data = ''.join(training_data)
training_data = training_data[26:]

def test_deobfuscate_word(s):
  print 'Decoding: ' + s
  result = deobfuscate(s)
  print 'Result: ' + result
  print
  return result

viagra_tests = ["viagorea", "viagdrha", "v l a g r a", "vyagra", "via---gra", "viagrga", "via-gra", "v 1 @ g' ra", "viagzra", "viagdra", "via_gra", "viazugra", "viargvra", "viagrya", "vii-agra", "viagwra", "vi(@)gr@", "viagvra", "v-i-a-g-r-a", "vi-ag.ra", "vigra", "vkiagra", "via.gra", "v-ii-a=g-ra", "v l a g r a", "via7gra", "v/i/a/g/r/a", "vixagra", "viaggra", "vi@gr|@|", "viatagra", "viaverga", "viagr(a", "viagr^a", "viagara", "viag@ra", "viag&ra", "vi@g*r@", "v-i.a-g*r-a", "v1@gra", "viaaprga", "vi$agra", "viaj1gra", "viag$ra", "via---gra", "vi.ag.ra", "viaoygra", "vi/agra", "viag%ra", "viarga", "v|i|a|g|r|a", "viag)ra", "vi@|g|r@", "viag&ra", "vi**agra", "vi@gr*@", "vi-@gr@", "v iagr", "v&iagra"] 
viagra_results = [test_deobfuscate_word(x) for x in viagra_tests]

other_tests = [ "u-n-s-u-s-c-r-i-b-e link", "re xe finance", "m/\ cromei)i", "gre4t pr1ces", "heyllooo its me chelsea", "veerrryyy cheeaapp", "u.n sabscjbe", "con. tains forwa. rdlook. ing sta. tements", "pa, rty for sen, ding this re.port.", "get your u n iversi t y d i pl0 m a", "ree movee below:"] 
other_results = [test_deobfuscate_word(x) for x in other_tests]
