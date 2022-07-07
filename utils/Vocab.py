import torch

class Vocab():
    def __init__(self,embed,word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
    
    def __len__(self):
        return len(word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX
    
    def make_features(self,batch,sent_trunc=50,doc_trunc=100,split_token='\n'):
        sents_list,targets,rationale_targets,doc_lens = [],[],[],[]
        # trunc document
        for doc,rationale,label in zip(batch['doc'],batch['rationale'],batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            rationale_labels = rationale.split(split_token)
            rationale_labels = [int(l) for l in rationale_labels]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            rationale_labels = rationale_labels[:max_sent_num]
            sents_list += sents
            targets += labels
            rationale_targets += rationale_labels
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            if len(words) == 0:
                print(sents_list)
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        
        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)
        
        features = torch.LongTensor(features)    
        targets = torch.LongTensor(targets)
        rationales =  torch.LongTensor(rationale_targets)
        summaries = batch['summaries']

        return features,targets,rationales,summaries,doc_lens

    def make_predict_features(self, batch, sent_trunc=150, doc_trunc=100, split_token='. '):
        sents_list, doc_lens = [],[]
        for doc in batch:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)

        return features, doc_lens
