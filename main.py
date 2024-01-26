#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
import pickle
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=66)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
parser.add_argument('-dropout',type=float,default=0.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=15)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')

parser.add_argument('-alpha_loss',type=float, default=0.5)
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 
#torch.backends.cudnn.deterministic = True    
#torch.backends.cudnn.benchmark = False
def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    
    for batch in data_iter:
        features,targets,rationale_s, rationale_w,_,doc_lens = vocab.make_features(batch,  doc_trunc = args.pos_num)
        features,targets,rationale_s, rationale_w = Variable(features), Variable(targets.float()), Variable(rationale_s.float()), Variable(rationale_w.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
            rationale_s = rationale_s.cuda()
            rationale_w = rationale_w.cuda()
        
        if 'AttnRNNRW' in args.model:
                probs, alpha_s, alpha_w = net(features,doc_lens)
                alpha_s = alpha_s.view(rationale_s.shape)
                alpha_w = alpha_w.view(rationale_w.shape)
        elif 'AttnRNNR' in args.model:
                probs, alpha = net(features,doc_lens)
                alpha = alpha.view(rationale_s.shape)
                rationale = rationale_s
        elif 'AttnRNNW' in args.model:
                probs, alpha = net(features,doc_lens)
                alpha = alpha.view(rationale_w.shape)
                rationale = rationale_w
        
        else:
                probs = net(features,doc_lens)
        
        if 'AttnRNNRW' in args.model:
                 Ll = criterion(probs,targets)
                 Ls =  criterion(alpha_s, rationale_s)
                 Lw = criterion(alpha_w, rationale_w)
                 #Ll = 10 * Ll
                 #Lw = 3 * Lw
                 #logging.info("Ll :" + str(Ll))
                 #logging.info("Ls :" + str(Ls))
                 #logging.info("Lw :" + str(Lw))
                 loss = args.alpha_loss * Ll+ (1 - args.alpha_loss)/2 *Ls + (1 - args.alpha_loss)/2 * Lw

        elif 'AttnRNNR' in args.model or 'AttnRNNW' in args.model:
                Ll = criterion(probs,targets)
                Lr =  criterion(alpha, rationale)
                #Ll = 10 * Ll
                #Lr = 3 * Lr if 'AttnRNNW' in args.model else Lr
                #Lw = criterion(alpha_w, rationale_w)
                #logging.info("Ll :" + str(Ll))
                #logging.info("Lr :" + str(Lr))
                #print("Lw :" + str(Lw))
                loss = args.alpha_loss * Ll+ (1 - args.alpha_loss) * Lr
        
        else:
                loss = criterion(probs,targets)

            
        total_loss += loss.data
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args,embed)
    if use_gpu:
        net = nn.DataParallel(net,device_ids=list(range(torch.cuda.device_count()))).cuda()
        #net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    t1 = time() 
    for epoch in range(1,args.epochs+1):
        for i,batch in enumerate(train_iter):
            logging.info(str(i))
            features,targets,rationale_s, rationale_w,_,doc_lens = vocab.make_features(batch, doc_trunc = args.pos_num)
            features,targets,rationale_s, rationale_w = Variable(features), Variable(targets.float()), Variable(rationale_s.float()),  Variable(rationale_w.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
                rationale_s= rationale_s.cuda()
                rationale_w = rationale_w.cuda()
            
            if 'AttnRNNRW' in args.model :
                probs, alpha_s, alpha_w = net(features,doc_lens)
                alpha_s = alpha_s.view(rationale_s.shape)
                alpha_w = alpha_w.view(rationale_w.shape)
            elif 'AttnRNNR' in args.model :
                probs, alpha = net(features,doc_lens)
                alpha = alpha.view(rationale_s.shape)
                rationale = rationale_s
            elif 'AttnRNNW' in args.model :
                probs, alpha = net(features,doc_lens)
                alpha = alpha.view(rationale_w.shape)
                rationale = rationale_w
            else:
                probs = net(features,doc_lens)
            if 'AttnRNNRW' in args.model:
                Ll = criterion(probs,targets)
                Ls =  criterion(alpha_s, rationale_s)
                Lw = criterion(alpha_w, rationale_w)
                loss = args.alpha_loss * Ll+ (1 - args.alpha_loss)/2 * Ls + (1 - args.alpha_loss)/2 * Lw
            elif 'AttnRNNR' in args.model or 'AttnRNNW' in args.model:
                Ll = criterion(probs,targets)
                Lr =  criterion(alpha, rationale)
                loss = args.alpha_loss * Ll+ (1 - args.alpha_loss) * Lr
            
            else:
                loss = criterion(probs,targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            if args.debug:
                print('Batch ID:%d Loss:%f' %(i,loss.data[0]))
                continue
            if i % args.report_every == 0:
                cur_loss = eval(net,vocab,val_iter,criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.module.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
    t2 = time()
    #writer.flush()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))

def test():
     
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    print("before eval")
    net.eval()
    print("after eval")
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
   
    with open(os.path.join(args.ref,'ref.txt'), 'w') as f:
         pass
    with open(os.path.join(args.hyp,'hyp.txt'), 'w') as f:
         pass
    at = []
    for batch in tqdm(test_iter):
        
        features,rationale_s, rationale_w,_,summaries,doc_lens = vocab.make_features(batch, doc_trunc = args.pos_num)
        t1 = time()
        if use_gpu:
            features = Variable(features).cuda()
        else:
            features = Variable(features)
        if 'AttnRNNRW' in args.model:
            probs, alpha_s, alpha_w = net(features, doc_lens)
        elif 'AttnRNNR' in args.model or 'AttnRNNW' in args.model:
            probs, alpha = net(features, doc_lens)
        
        else:
            probs = net(features, doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
    
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            print(len(doc))
            print(topk_indices)
            hyp = [doc[index] for index in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,'ref.txt'), 'a') as f:
                f.write(ref+'<<END>>')
            with open(os.path.join(args.hyp,'hyp.txt'), 'a') as f:
                f.write('\n'.join(hyp)+'<<END>>')
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict(examples):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    pred_dataset = utils.Dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        features, doc_lens = vocab.make_predict_features(batch, doc_trunc = args.pos_num)
        print(doc_lens)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        probs = probs[0]
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    if args.test:
        test()
    elif args.predict:
        with open(args.filename) as file:
            bod = [file.read()]
        predict(bod)
    else:
        logging.info('GPUs: '+str(torch.cuda.device_count()))
        train()
    #writer.close()
