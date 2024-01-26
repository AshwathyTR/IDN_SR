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
        sents_list,targets,rationale_s_targets,rationale_w_targets, doc_lens,word_rationales = [],[],[],[],[],[]
        # trunc document
        #print(rationale_type)
        for doc,rationale_s,rationale_w,label in zip(batch['doc'],batch['rationale_s'], batch['rationale_w'],batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            assert len(sents) == len(labels)
            rationale_s_labels = rationale_s.split(split_token)
            rationale_s_labels = [float(l) for l in rationale_s_labels]
            assert len(rationale_s_labels) == len(sents)
            rationale_w_labels = rationale_w.split(split_token)
            assert len(rationale_w_labels) == len(sents)
            max_sent_num = doc_trunc
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            #assert len(sents) == len(labels)
            rationale_s_labels = rationale_s_labels[:max_sent_num]
            rationale_w_labels = rationale_w_labels[:max_sent_num]
            #assert len(sents) == len(rationale_labels)
            if len(sents) < doc_trunc:
                #print('sentlen'+str(len(sents)))
                pad = ['.']* (doc_trunc - len(sents))
                #print('padsentlen'+str(len(sents)))
                #print('labellen'+str(len(labels)))
                labels_pad = [0]*(doc_trunc - len(sents))
                rationale_w_pad =  ['0.0']*(doc_trunc - len(sents))
                rationale_w_labels = rationale_w_labels + rationale_w_pad
                rationale_s_pad =  [0]*(doc_trunc - len(sents))
                rationale_s_labels = rationale_s_labels + rationale_s_pad
                sents = sents+pad
                labels = labels + labels_pad
                #print('labelpadlen'+str(labels))
                #print(labels_pad)
                assert len(sents) == len(labels)
                assert len(sents) == len(rationale_s_labels)
                assert len(sents) == len(rationale_w_labels)
                #    rationale_pad =  ['0']*(doc_trunc - len(sents))
                #    rationale_labels = rationale_labels + rationale_pad
            sents_list += sents
            targets += labels
            rationale_s_targets += rationale_s_labels
            rationale_w_targets += rationale_w_labels
            doc_lens.append(len(sents))
            
           
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        max_rationale_len =0 
        s_rationales = rationale_s_targets
        batch_rationales = []
        for sent_rationale in rationale_w_targets:
                word_rationales = sent_rationale.split(' ')
                #print('r_len'+str(len(word_rationales)))
                if len(word_rationales) > sent_trunc:
                    word_rationales = word_rationales[:sent_trunc]
                max_rationale_len = len(word_rationales) if len(word_rationales) > max_rationale_len else max_rationale_len
                #print(word_rationales)
                word_rationales = [float(word_rationale) for word_rationale in word_rationales]
                batch_rationales.append(word_rationales) 
        #print('b_len'+str(len(batch_rationales)))
        for sent in sents_list:
            words = sent.split(' ')
            #print('w'+str(len(words)))
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            if len(words) == 0:
                print(sents_list)
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        #print('max sent:'+str(max_sent_len))
        #print('max r:'+str(max_rationale_len))
        #if rationale_type:
        #    if not  max_sent_len == max_rationale_len:
        #         print(max_sent_len)
        #         print(max_rationale_len)
        #    assert max_sent_len == max_rationale_len
        #print('max:'+str(max_sent_len))
        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)
        
        w_rationales = []
        for sent in batch_rationales:
                #print('sentence length:'+str(len(sent)))
                
                rationale = sent + [0.0 for _ in range(max_sent_len-len(sent))]
                #print('rationale length:'+str(len(rationale)))
                
                w_rationales.append(rationale)
        #print('w_rationales:'+str(len(w_rationales)))
        features = torch.LongTensor(features)    
        targets = torch.LongTensor(targets)
        w_rationales =  torch.FloatTensor(w_rationales) 
        s_rationales = torch.LongTensor(s_rationales)
        summaries = batch['summaries']

        return features,targets,s_rationales,w_rationales, summaries,doc_lens

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
