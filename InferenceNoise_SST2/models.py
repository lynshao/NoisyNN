import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)
  
class BertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2',num_labels = 2)
        self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
