import json
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
dpath = sys.argv[1]
split = sys.argv[2]
look_behind = int(sys.argv[3])
look_ahead  = int(sys.argv[3])
path = os.path.join(dpath, split+'.json')

with open(path,'r') as f:
    h = f.readlines()



with open(path[:-5]+'_tfidf_'+str(look_behind)+'.json','w') as f:
    pass


corpus = []
for text in h:
    j = json.loads(text)
    d = j['doc']
    corpus.append(d)

X = vectorizer.fit_transform(corpus)
    
for doc_id, text in enumerate(h):
    j = json.loads(text)
    d = j['doc']
    s= d.split('\n')
    r = []
    w = sum([sent.split(' ') for sent in s],[])
    word_index = 0
    tf_idf = dict(zip(vectorizer.get_feature_names(), X.toarray()[doc_id]))
    for index, sentence in enumerate(s):
        sentence_r = []
        for w_index, word in enumerate(sentence.split(' ')):
            word_index = word_index +1
            window = ' '.join(w[max(0,word_index - look_behind) : word_index + look_ahead])
             
            if 'choice :' in window.lower():
                sentence_r.append(str(tf_idf[word]))
                #print(index)
            else:
                sentence_r.append(str(0.0))
        sentence_r = ' '.join(sentence_r)
        r.append(sentence_r)
    r = '\n'.join(r)
    j['rationale'] = r
    #print(r)
    with open(path[:-5]+'_tfidf_'+str(look_behind)+'.json','a') as f:
        json.dump(j,f)
        f.write('\n')

