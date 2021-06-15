import os
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from SincNet import SincNet
from utils import create_batches_rnd

#variables
fold_train = 'data_lists/TIMIT_train.scp'
fold_test = 'data_lists/TIMIT_test.scp'
class_dict_file = 'data_lists/TIMIT_labels.npy'
data_folder = 'data_new/'
seed = 1234
fs=16000
cw_len=200
cw_shift=10
N_epochs = 300
N_batches = 800
N_eval_epoch = 8
batch_size = 128
output_folder = ''

# getting train and test list of data
wav_train = []
with open(fold_train, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        wav_train.append(line.rstrip())
n_train = len(wav_train)

wav_test = []
with open(fold_test, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        wav_test.append(line.rstrip())
n_test = len(wav_test)    
    
torch.manual_seed(seed)
np.random.seed(seed)

wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

lab_dict = np.load(class_dict_file, allow_pickle=True).item()

mNN = SincNet()
mNN.cuda()

optimizer_mNN = optim.RMSprop(mNN.parameters(), lr=0.001, alpha=0.95, eps=1e-8) 
loss = nn.NLLLoss()


err_train = []
err_test = []
for epoch in range(N_epochs):
  
  test_flag=0
  mNN.train()
 
  loss_sum=0
  err_sum=0

  for i in range(N_batches):

    [x, y]=create_batches_rnd(batch_size, data_folder, wav_train, n_train, wlen, lab_dict, 0.2)
    y_pred_prob=mNN(x)
    
    y_pred=torch.max(y_pred_prob, dim=1)[1]
    loss_ = loss(y_pred_prob, y.long())
    err = torch.mean((y_pred != y.long()).float())
   
    optimizer_mNN.zero_grad()
    
    loss_.backward()
    optimizer_mNN.step()
    
    loss_sum=loss_sum+loss_.detach()
    err_sum=err_sum+err.detach()
 
  loss_tot=loss_sum/N_batches
  err_tot=err_sum/N_batches
    
  if epoch%N_eval_epoch==0:
      
   mNN.eval()
   test_flag=1 
   loss_sum=0
   err_sum=0
   err_sum_snt=0
   
   with torch.no_grad():  
    for i in range(n_test):

     [signal, fs] = sf.read(data_folder+wav_test[i].upper())

     signal=torch.from_numpy(signal).float().cuda().contiguous()
     lab_batch=lab_dict[wav_test[i]]
    
     beg_samp=0
     end_samp=wlen
     
     N_fr=int((signal.shape[0]-wlen)/(wshift))
     

     sig_arr=torch.zeros([batch_size,wlen]).float().cuda().contiguous()
     lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
     pout=Variable(torch.zeros(N_fr+1, 462).float().cuda().contiguous())
     count_fr=0
     count_fr_tot=0
     while end_samp<signal.shape[0]:
         sig_arr[count_fr,:]=signal[beg_samp:end_samp]
         beg_samp=beg_samp+wshift
         end_samp=beg_samp+wlen
         count_fr=count_fr+1
         count_fr_tot=count_fr_tot+1
         if count_fr==batch_size:
             inp=Variable(sig_arr)
             pout[count_fr_tot-batch_size:count_fr_tot,:]=mNN(inp)
             count_fr=0
             sig_arr=torch.zeros([batch_size,wlen]).float().cuda().contiguous()
   
     if count_fr>0:
      inp=Variable(sig_arr[0:count_fr])
      pout[count_fr_tot-count_fr:count_fr_tot,:]=mNN(inp)

    
     pred=torch.max(pout,dim=1)[1]
     loss_ = loss(pout, lab.long())
     err = torch.mean((pred!=lab.long()).float())
    
     [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
     err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
    
    
     loss_sum=loss_sum+loss_.detach()
     err_sum=err_sum+err.detach()
    
    err_tot_dev_snt=err_sum_snt/n_test
    loss_tot_dev=loss_sum/n_test
    err_tot_dev=err_sum/n_test

  
   print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))
   err_train.append(err_tot)
   err_test.append(err_tot_dev)
   checkpoint={'Mnn_model_par': mNN.state_dict()
               }
   torch.save(checkpoint,output_folder+'/model.pkl')
  
  else:
   print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))