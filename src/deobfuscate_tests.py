from deobfuscate import deobfuscate

# Run deobfuscate(string) on various obfuscated test strings... that's
# all

training_data = open('../data/training_data.txt').readlines()
training_data = [x[:-1] for x in training_data]
training_data = ''.join(training_data)
training_data = training_data[26:]
