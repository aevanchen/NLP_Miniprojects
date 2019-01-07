import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class model(nn.Module):

     def __init__(self, vocab_size, method=None,hidden_dim=128):
        super(model, self).__init__()
        self.method=method
        self.hidden_dim = hidden_dim
      
        if self.method == 'mul':
           # self.attn = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W=torch.nn.Parameter(torch.FloatTensor(self.hidden_dim,self.hidden_dim))
            #self.W = torch.nn.Parameter( nn.init.xavier_uniform_(torch.FloatTensor(self.hidden_dim,self.hidden_dim),
                                                               # gain=nn.init.calculate_gain('relu')))
            
        elif self.method=='add':
           # self.attn = torch.nn.Linear(2*self.hidden_dim, self.hidden_dim)
            self.W1 = torch.nn.Parameter( nn.init.xavier_uniform_(torch.FloatTensor(self.hidden_dim,self.hidden_dim),
                                                                gain=nn.init.calculate_gain('relu')))
            self.W2 = torch.nn.Parameter( nn.init.xavier_uniform_(torch.FloatTensor(self.hidden_dim,self.hidden_dim),
                                                                gain=nn.init.calculate_gain('relu')))
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_dim,1))
           
        elif self.method is None:
            pass
        else:
            raise ValueError('Attention type should either "mul" or "add" or blank!')
        

      
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        if self.method is None:
            self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.decoder = nn.LSTM(2*hidden_dim, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax=nn.Softmax(dim=0)
       



     def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)
        
     def forward(self, input_seq, gold_seq=None):
        if self.method is None:
            input_vectors = self.embeds(torch.tensor(input_seq))
            input_vectors = input_vectors.unsqueeze(1)
            outputs, hidden = self.encoder(input_vectors)

            # Technique used to train RNNs: 
            # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
            teacher_force = True

            # This condition tells us whether we are in training or inference phase 
            if gold_seq is not None and teacher_force:
                gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
                gold_vectors = gold_vectors.unsqueeze(1)
                gold_vectors = torch.nn.functional.relu(gold_vectors)
                outputs, hidden = self.decoder(gold_vectors, hidden)
                predictions = self.out(outputs)
                predictions = predictions.squeeze()
                vals, idxs = torch.max(predictions, 1)
                return predictions, list(np.array(idxs))
            else:
                prev = torch.zeros(1, 1, self.hidden_dim)
                predictions = []
                predicted_seq = []
                for i in range(len(input_seq)):
                    prev = torch.nn.functional.relu(prev)
                    outputs, hidden = self.decoder(prev, hidden)
                    pred_i = self.out(outputs)
                    pred_i = pred_i.squeeze()
                    _, idx = torch.max(pred_i, 0)
                    idx = idx.item()
                    predictions.append(pred_i)
                    predicted_seq.append(idx)
                    prev = self.embeds(torch.tensor([idx]))
                    prev = prev.unsqueeze(1)
                return torch.stack(predictions), predicted_seq

        #this forward pass is for using attention
        else:
            input_vectors = self.embeds(torch.tensor(input_seq))
            input_vectors = input_vectors.unsqueeze(1)
            encoder_outputs, hidden = self.encoder(input_vectors)
            
            # Technique used to train RNNs: 
            # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
            teacher_force = True
            #get embedding vector
            gold_vectors=torch.zeros(1, self.hidden_dim)
            gold_vectors=gold_vectors.unsqueeze(1)
          
            predictions=[]
            predicted_seq=[] 
            
            decoder_hidden=hidden[0] #this is same
      
            for t in range (len(input_seq)):
                if(self.method=='mul'):
                    energy=torch.mm(encoder_outputs.squeeze(1),self.W)
               # energy=self.attn(encoder_outputs.squeeze(1))  
                    attn_weights=torch.mm(decoder_hidden.squeeze(1),energy.t())
                    attn_weights=attn_weights.t()
                    
                else:
                    
                    s=decoder_hidden.expand(len(input_seq),-1,-1)
                
                    attn_weights=torch.add(torch.mm(encoder_outputs.squeeze(1),self.W1),torch.mm(s.squeeze(1),self.W2)).tanh()
                    

                    attn_weights=torch.mm(attn_weights.squeeze(1),self.v)
                    
              
                attn_weights=self.softmax(attn_weights)  

                #generte context vector by multiplying weights with encoder outputs
                context=torch.sum(attn_weights*encoder_outputs.squeeze(1),dim=0).view(1,1,-1)
               
                decoder_input=torch.cat((gold_vectors, context), 2)

                decoder_input = torch.nn.functional.relu(decoder_input)

                decoder_outputs, hidden = self.decoder(decoder_input, hidden)

                pred=self.out(decoder_outputs)
                pred = pred.squeeze()

                _, idx = torch.max(pred, 0)
                idx = idx.item()
                predictions.append(pred)
                predicted_seq.append(idx)
                if gold_seq is not None and teacher_force:

                    gold_vectors=self.embeds(torch.tensor(gold_seq[t:t+1])).unsqueeze(1)
                   # print(gold_vectors)
                else:
                    gold_vectors=self.embeds(torch.tensor([idx])).unsqueeze(1)

                decoder_hidden=hidden[0]
            return torch.stack(predictions),predicted_seq


