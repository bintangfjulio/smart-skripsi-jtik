import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT_CNN(nn.Module):
    def __init__(self, labels, pretrained_bert, dropout, window_sizes, in_channels, out_channels, num_bert_states):
        super(BERT_CNN, self).__init__()
        self.pretrained_bert = pretrained_bert

        conv_layers = []
        for window_size in window_sizes:
            conv_layer = nn.Conv2d(in_channels, out_channels, (window_size, pretrained_bert.embeddings.word_embeddings.weight.size(1)))
            conv_layers.append(conv_layer)
            
        self.cnn = nn.ModuleList(conv_layers)

        self.dropout = nn.Dropout(dropout) 
        self.num_bert_states = num_bert_states

        self.output_layer = nn.Linear(len(window_sizes) * out_channels, len(labels))

    def forward(self, input_ids, attention_mask):
        bert_output = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask)
        stacked_hidden_states = torch.stack(bert_output.hidden_states[-self.num_bert_states:], dim=1)

        pooling = []
        for layer in self.cnn:
            hidden_states = layer(stacked_hidden_states)
            relu_output = F.relu(hidden_states.squeeze(3))
            pooling.append(relu_output)

        max_pooling = []
        for features in pooling:
            pooled_features = F.max_pool1d(features, features.size(2)).squeeze(2)
            max_pooling.append(pooled_features)
        
        concatenated = torch.cat(max_pooling, dim=1)
        preds = self.dropout(concatenated)

        preds = self.output_layer(preds)
        
        return preds