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
tf_idf = dict(zip(vectorizer.get_feature_names(), X.toarray()[index]))
    
for doc_id, text in enumerate(h):
    j = json.loads(text)
    d = j['doc']
    s= d.split('\n')
    r = []
    for index in range(0,len(s)):
        window = '\n'.join(s[max(0,index - look_behind) : index + look_ahead])
        #print(index)
        #print(window)
        #print('..............')
        
        if 'choice :' in window.lower():
            r.append(1.0)
            #print(index)
        else:
            r.append(0.0)
    tf_idf = dict(zip(vectorizer.get_feature_names(), X.toarray()[doc_id]))
    r = [str(mask * score) for mask, score in zip(r, tf_idf)]
    r = '\n'.join(r)
    j['rationale'] = r
    #print(r)
    with open(path[:-5]+'_tfidf_'+str(look_behind)+'.json','a') as f:
        json.dump(j,f)
        f.write('\n')

