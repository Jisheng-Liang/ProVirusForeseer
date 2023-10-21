import torch
import pandas as pd
import torch.nn as nn
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from datasets import load_dataset, load_metric
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
import re

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='Rostlab/prot_bert'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        seq = str(self.data.iloc[index,0])
        seq = list(seq)
        seq = " ".join(seq)
        seq = re.sub(r"[UZOB]", "X", seq)
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_input = self.tokenizer(seq,
                                      padding='max_length',  # Pad to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_input['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_input['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_input['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.iloc[index,1] * self.data.iloc[index,2]
            label = torch.FloatTensor([label])
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids
        
class ProteinRegressor(nn.Module):

    def __init__(self, bert_model, freeze_bert=False):
        super(ProteinRegressor, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = BertModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "Rostlab/prot_bert":  # 12M parameters
            hidden_size = 1024

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids, return_dict=False)
        pooler_output = self.act(pooler_output)
        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            outputs = net(seq, attn_masks, token_type_ids)
            loss = criterion(outputs,labels)
            mean_loss += loss.item()
            count += 1

    return mean_loss / count

class trainer:
    def __init__(self, model, criterion, optimizer, scheduler, epochs, 
                 gradient_accumulation_steps,early_stopping,
                 fp16=True,
                 ):
        self.early_stopping = early_stopping
        self.model = model
        self.fp16 = fp16
        if self.fp16:
            self.scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.criterion = criterion
    def step(self, dataset, bs, device):
        best_loss = np.Inf
        best_ep = 1
        nb_iterations = int(len(dataset)*0.8)
        print_every = nb_iterations // 5
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for ep in range(self.epochs):

            # kFold cross-validation
            for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
                if fold >= 1:
                    break
                train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
                val_fold = torch.utils.data.dataset.Subset(dataset, val_index)

                train_loader = DataLoader(train_fold, batch_size=bs, num_workers=5)
                val_loader = DataLoader(val_fold, batch_size=bs, num_workers=5)
                self.model.train()
                train_loss = 0.0
                for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
                    # Converting to cuda tensors
                    seq, attn_masks, token_type_ids, labels = \
                        seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
                    # Enables autocasting for the forward pass (model + loss)
                    # Obtaining the logits from the model
                    with autocast():
                        outputs = self.model(seq, attn_masks, token_type_ids)
                        loss = self.criterion(outputs,labels)
                        # # Computing loss
                        loss = loss / self.gradient_accumulation_steps

                    if self.fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    train_loss += loss.item()
                    if (it + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    if (it + 1) % print_every == 0:  # Print training loss information
                        print("No.{}Fold/{} of epoch {} complete. Loss : {} "
                            .format(fold, nb_iterations, ep+1, train_loss / print_every))
                        train_loss = 0.0


                val_loss = evaluate_loss(self.model, device, self.criterion, val_loader)  # Compute validation loss
                print()
                print("Epoch {} Fold {} complete! Validation Loss : {}".format(ep+1, fold, val_loss))

                if val_loss < best_loss:
                    print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
                    print()
                    net_copy = copy.deepcopy(self.model)  # save a copy of the model
                    best_loss = val_loss
                    best_ep = ep + 1
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early Stopping!")
                break
        # Saving the model
        path_to_model='models/{}_val_loss_{}_ep_{}.pt'.format('ProtBert', round(best_loss, 5), best_ep)
        torch.save(net_copy.state_dict(), path_to_model)
        print("The model has been saved in {}".format(path_to_model))

        del loss
        torch.cuda.empty_cache()

def main():
    bert_model = "Rostlab/prot_t5_xl_uniref50"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
    maxlen = 1000  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = 8  # batch size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 1e-5  # learning rate
    epochs = 40  # number of training epochs
    num_labels = 1
    set_seed(1)

    # datasets
    data = pd.read_csv('data/train_seq.csv')
    dataset = CustomDataset(data,maxlen)
    print('dataset loaded!')
    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ProteinRegressor(bert_model)
    opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    early_stopping = EarlyStopping(patience=1, verbose=True)
    model.to(device)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    t_total = int((len(dataset) * 0.8) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    
    train = trainer(model, criterion, opti, lr_scheduler, epochs, iters_to_accumulate, early_stopping)
    train.step(dataset, bs, device)


if __name__ == "__main__":
    main()