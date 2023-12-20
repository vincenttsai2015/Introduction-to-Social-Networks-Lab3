#!/usr/bin/env python
# coding: utf-8

# import general tools
import numpy as np
import pandas as pd
import json
import re
import nltk
import string
from tqdm import tqdm

# import DL tools
import torch
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# self-defined dataset module
class KaggleSentimentDataset(Dataset):
    def __init__(self, df, tokenizer):
        texts = df.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self.texts = [tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt") for text in texts]
        if 'emotion' in df:
            self.labels = [all_labels[label] for label in df['emotion']]
        else:
            self.labels = [-1] * len(df)
    
    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)
        text = self._remove_punctuation(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()

    def _remove_amp(self, text):
        return text.replace("&amp;", " ")
    
    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)
    
    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)
    
    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')
        return [token for token in text_tokens if token not in stop_words]

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]
        return text, label

# model
class BertSentimentClassifier(Module):
    def __init__(self, base_model, dropout=0.5):
        super(BertSentimentClassifier, self).__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 8)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output_1 = self.fc1(dropout_output)
        linear_output_2 = self.fc2(linear_output_1)
        final_layer = self.relu(linear_output_2)
        return final_layer

# training
def train(model, train_ldr, val_ldr, learning_rate, epochs):
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0
    
    # GPU usage determination
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Defining loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # total_steps = len(train_ldr) * 1
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    # Send the model to GPU memory
    model = model.to(device)
    criterion = criterion.to(device)
    # training iteration
    for epoch_num in range(epochs):
        print(f'Epoch: {epoch_num}')
        # training accuracy and loss
        train_acc = []
        train_loss = 0
        
        model.train()
        # tqdm
        for train_input, train_label in tqdm(train_ldr):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
            optimizer.zero_grad()
            # prediction result
            output = model(input_id, mask)
            # loss
            batch_loss = criterion(output, train_label)
            train_loss += batch_loss.item()
            
            # model update
            model.zero_grad()
            batch_loss.backward()            
            optimizer.step()
            # scheduler.step()
            
            # accuracy
            output_index = output.argmax(axis=1)
            acc = (output_index == train_label)
            train_acc += acc

        train_accuracy = (sum(train_acc)/len(train_acc)).item()
        print(f'Train Loss: {train_loss / len(train_ldr): .3f} | Train Accuracy: {train_accuracy: 10.3%}')
        
        # Model validation
        model.eval()
        # validation accuracy and loss
        val_acc = []
        val_loss = 0        
        with torch.no_grad(): # no need to compute gradient
            # validate with trained model
            for val_input, val_label in tqdm(val_ldr):
                # same process with training
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                # prediction result
                output = model(input_id, mask)
                # loss
                batch_loss = criterion(output, val_label)
                val_loss += batch_loss.item()
                
                # accuracy
                output_index = output.argmax(axis=1)
                acc = (output_index == val_label)
                val_acc += acc
            val_accuracy = (sum(val_acc)/len(val_acc)).item()            
            print(f'Val Loss: {val_loss / len(val_ldr): .3f} | Val Accuracy: {val_accuracy: 10.3%}')
            
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(model, f"best_model.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1
            if early_stopping_threshold_count >= 3:
                print("Early stopping")
                break

def get_text_predictions(model, test_ldr):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)    
    
    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(test_ldr):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)            
            output_index = output.argmax(axis=1)
            results_predictions.append(output_index)
    
    return torch.cat(results_predictions).cpu().detach().numpy()

print('Raw data parsing...')
# groundtruth
df_groundtruth = pd.read_csv('emotion.csv')

# train-test-splitting
df_identification = pd.read_csv('data_identification.csv')

# raw data parsing
tweet_ids = []
textual_data = []
with open("tweets_DM.json","r") as jsfile:
    for line in jsfile.readlines():
        dic = json.loads(line)
        tweet_ids.append(dic['_source']['tweet']['tweet_id'])
        textual_data.append(dic['_source']['tweet']['text'])

# merging with identification for splitting train/test data
df_raw = pd.DataFrame({'tweet_id': tweet_ids, 'text': textual_data})
df_raw_merge = pd.merge(df_raw, df_identification, on='tweet_id', how='left')

print('Tidying up training data...')
# preparing training data and labels
df_identification_train = df_identification[df_identification['identification']=='train']
df_raw_merge_train = df_raw_merge[df_raw_merge['identification']=='train']
df_raw_merge_train = pd.merge(df_raw_merge_train, df_groundtruth, on='tweet_id', how='left')

print('Tidying up testing data...')
# preparing testing data 
df_submit_samples = pd.read_csv("sampleSubmission.csv",encoding="utf-8")
df_test = df_submit_samples.drop(columns=['emotion'])
df_submit_samples.rename(columns = {'id':'tweet_id'}, inplace = True)
df_test.rename(columns = {'id':'tweet_id'}, inplace = True)
df_identification_test = df_identification[df_identification['identification']=='test']
df_raw_merge_test = df_raw_merge[df_raw_merge['identification']=='test']
df_raw_merge_test = pd.merge(df_test, df_raw_merge_test, on='tweet_id', how='left')
# df_raw_merge_test['emotion'] = -1

print('Saving data...')
# saving data
df_train = df_raw_merge_train.drop(columns=['tweet_id', 'identification'])
df_test = df_raw_merge_test.drop(columns=['tweet_id', 'identification'])
train_csv = df_train.to_csv("./dataset/dataset.train.csv", index=False, columns=["text","emotion"], encoding="utf-8")
test_csv = df_test.to_csv("./dataset/dataset.test.csv", index=False, columns=["text"], encoding="utf-8")

print('Sampling validation data from training data...')
# split the training data into training part and validation part
df_val = df_train.sample(frac=0.2, random_state=30)
df_train = df_train.drop(df_val.index)

print('Tokenizing...')
# available labels and tokenizer
available_labels = df_raw_merge_train['emotion'].unique().tolist()
all_labels = {element: count for count, element in enumerate(available_labels)}
id2label = {count: element for count, element in enumerate(available_labels)}
# {'anticipation':0, 'sadness':1, 'fear':2, 'joy':3, 'anger':4, 'trust':5, 'disgust':6, 'surprise':7}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# construct the datasets
print('Training dataset construction...')
training_dataset = KaggleSentimentDataset(df_train, tokenizer)
print('Validation dataset construction...')
validation_dataset = KaggleSentimentDataset(df_val, tokenizer)
print('Testing dataset construction...')
testing_dataset = KaggleSentimentDataset(df_test, tokenizer)

# dataloaders
print('Building dataloaders...')
train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(validation_dataset, batch_size=16, num_workers=0)
test_dataloader = DataLoader(testing_dataset, batch_size=16, shuffle=False, num_workers=0)

# model initialization
print('Initializing BERT model...')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(all_labels))
base_model = BertModel.from_pretrained("roberta-base")
model = BertSentimentClassifier(base_model)

# training and testing
print('Training...')
EPOCHS = 3
LR = 2e-5
train(model, train_dataloader, val_dataloader, LR, EPOCHS)

print('Testing...')
pred_model = torch.load("best_model.pt")
sample_submission = pd.read_csv("sampleSubmission.csv")
pred_results = get_text_predictions(pred_model, test_dataloader)
id2label = {count: element for count, element in enumerate(available_labels)}
sample_submission["emotion"] = pred_results
sample_submission["emotion"] = sample_submission["emotion"].map(id2label)
sample_submission.to_csv("final_submission_112065802.csv", index=False)
