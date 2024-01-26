import json
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
vectorizer = TfidfVectorizer()
dpath = sys.argv[1]
split = sys.argv[2]
look_behind_w = int(sys.argv[3])
look_ahead_w  = int(sys.argv[3])
look_behind_s = int(sys.argv[4])
look_ahead_s  = int(sys.argv[4])
path = os.path.join(dpath, split+'.json')



#Load Corpus
with open(path,'r') as f:
    jsonlines = f.readlines()
corpus = []
for text in jsonlines:
    j = json.loads(text)
    doc = j['doc']
    corpus.append(doc)

#Clear Save File
with open(path[:-5]+'_ws_'+str(look_behind_s)+'_'+str(look_behind_w)+'.json','w') as f:
    pass



# Get TF-IDF Scores
X = vectorizer.fit_transform(corpus)
    
for doc_id, doc in enumerate(corpus):

    sentences = doc.split('\n')

	# Get Words around choice points
    tf_idf = dict(zip(vectorizer.get_feature_names(), X.toarray()[doc_id]))
    choice_tf_idf = {}
	word_index = 0
    for index, sentence in enumerate(sentences):        
        for w_index, word in enumerate(sentence.split(' ')):
            word_index = word_index +1
            window = ' '.join(w[max(0,word_index - look_behind_w) : word_index + look_ahead_w])
            if 'choice :' in window.lower() and word in tf_idf.keys():
                choice_tf_idf[word] = tf_idf[word]

	# Word Rationales
    rationales = []
    for index, sentence in enumerate(s):
        word_r = []
        for w_index, word in enumerate(sentence.split(' ')): 
            if word not in choice_tf_idf.keys():
                    word_r.append("0.0")
            else:
                    word_r.append(str(choice_tf_idf[word]))            
        word_r = ' '.join(word_r)
        rationales.append(word_r)
    rationales = '\n'.join(rationales)
    j['rationale_w'] = rationales
	
	# Sentence Rationales
    r = []
    for index in range(0,len(sentences)):
        window = '\n'.join(sentences[max(0,index - look_behind_s) : index + look_ahead_s])

        if 'choice :' in window.lower():
            r.append('1')
        else:
            r.append('0')
    r = '\n'.join(r)
    j['rationale_s'] = r
	
	# Save Dataset with Rationales
    with open(path[:-5]+'_ws_'+str(look_behind_s)+'_'+str(look_behind_w)+'.json','a') as f:
        json.dump(j,f)
        f.write('\n')

