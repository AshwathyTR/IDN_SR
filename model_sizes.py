import torch
from models import RNN_RNN, AttnRNN, AttnRNNRW, sentonlyAttnRNNR, wordonlyAttnRNNW, sentonlyAttnRNN, wordonlyAttnRNN
import argparse
#from .AttnRNN import AttnRNN
#from .AttnRNNRW import AttnRNNRW
#from .sentonlyAttnRNNR import sentonlyAttnRNNR
#from .wordonlyAttnRNNW import wordonlyAttnRNNW
#from .sentonlyAttnRNN import sentonlyAttnRNN
#from .wordonlyAttnRNN import wordonlyAttnRNN
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
#parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
parser.add_argument('-dropout',type=float,default=0.0)
# test
#parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
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

# Create an instance of the model
rnn = RNN_RNN(args)
attn = AttnRNN(args)
sattn = sentonlyAttnRNN(args)
wattn = wordonlyAttnRNN(args)
sattnr = sentonlyAttnRNNR(args)
wattnr = wordonlyAttnRNNW(args)
attnr = AttnRNNRW(args)

# Get the model state dict
rnn_sd = rnn.state_dict()
attn_sd = attn.state_dict()
sattn_sd = sattn.state_dict()
wattn_sd = wattn.state_dict()
sattnr_sd = sattnr.state_dict()
wattnr_sd = wattnr.state_dict()
attnr_sd = attnr.state_dict()

# Count the number of parameters
rnn_num  = sum(p.numel() for p in rnn_sd.values())
attn_num  = sum(p.numel() for p in attn_sd.values())
sattn_num  = sum(p.numel() for p in sattn_sd.values())
wattn_num  = sum(p.numel() for p in wattn_sd.values())
sattnr_num  = sum(p.numel() for p in sattnr_sd.values())
wattnr_num  = sum(p.numel() for p in wattnr_sd.values())
attnr_num  = sum(p.numel() for p in attnr_sd.values())

print(f'RNN : {rnn_num}')
print(f'Attn : {attn_num}')
print(f'SAttn : {sattn_num}')
print(f'WAttn : {wattn_num}')
print(f'SAttnR : {sattnr_num}')
print(f'WAttnR : {wattnr_num}')
print(f'AttnR : {attnr_num}')
