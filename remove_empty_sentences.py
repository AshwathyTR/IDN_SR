import sys
import json
import os
path = sys.argv[1]

folders = ['sr_3/data']
splits = ['train' , 'test', 'val']

for folder in folders:
    for split in splits:
        fpath = os.path.join(path, folder, split)+'.json'
        nfpath = fpath.replace('.json','_corrected.json')
        with open(nfpath, 'w') as f:
            pass
        with open(fpath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            sentences = data['doc'].split('\n')
            labels = data['labels'].split('\n')
            #print(len(sentences))
            #print(len(labels))
            #print(sentences[:-1])
            #print(labels[:-1])
            new_sentences = [sentence for sentence in sentences if len(sentence.split())>0 and sentence.strip()]
            new_labels = [labels[sentences.index(sentence)] for sentence in sentences if len(sentence.split())>0 and sentence.strip()]
            new_doc = '\n'.join(new_sentences)
            data['doc'] = new_doc
            data['labels'] = '\n'.join(new_labels)
            with open(nfpath, 'a') as f:
                 json.dump(data, f)
                 f.write('\n')
                
           

