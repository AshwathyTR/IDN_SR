import json
import sys
import os

dpath = sys.argv[1]
split = sys.argv[2]
look_behind = int(sys.argv[3])
look_ahead  = int(sys.argv[3])
path = os.path.join(dpath, split+'.json')

with open(path,'r') as f:
    h = f.readlines()


with open(path[:-5]+'_w_'+str(look_behind)+'.json','w') as f:
    pass


for text in h:
    j = json.loads(text)
    s = j['doc'].split('\n')
    w = sum([sent.split(' ') for sent in s],[])
    r = []
    word_index = 0
    for index, sentence in enumerate(s):
        sentence_r = []
        for w_index, word in enumerate(sentence.split(' ')):
            word_index = word_index +1
            window = ' '.join(w[max(0,word_index - look_behind) : word_index + look_ahead])
        #print(index)
        #print(window)
        #print('..............')
        
            if 'choice :' in window.lower():
                sentence_r.append('1')
            #print(index)
            else:
                sentence_r.append('0')
        sentence_r = ' '.join(sentence_r)
        r.append(sentence_r)
    r = '\n'.join(r)
    j['rationale'] = r
    #print(r)
    with open(path[:-5]+'_w_'+str(look_behind)+'.json','a') as f:
        json.dump(j,f)
        f.write('\n')

