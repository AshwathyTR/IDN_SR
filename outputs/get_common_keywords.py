from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

ps = PorterStemmer()

with open(file1, 'r') as f:
    text1 = f.read()
    words1 = [ps.stem(word) for word in  word_tokenize(text1)]

with open(file2, 'r') as f:
    text2 = f.read()
    words2 = [ps.stem(word) for word in  word_tokenize(text2)]

overlap = list(set(words1) & set(words2))
text = ''
for line in text1.split('\n'):
    for word in word_tokenize(line):
        if ps.stem(word) in overlap:
            word = "\033[44;33m"+word+"\033[m"
        text = text +' '+ word
    text = text+'\n'

print(text)

