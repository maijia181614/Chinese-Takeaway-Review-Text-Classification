import torch
import pandas
from transformers import BertModel, BertTokenizer
import numpy as np
import torch.nn as nn
import sklearn

class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = './chinese_wwm_ext_pytorch'
        self.bert_model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fullC = nn.Linear(768, 2).to(self.device)

    def forward(self, inputText):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_tokenized = self.tokenizer.batch_encode_plus(inputText, add_special_tokens=True,
                                                          max_length=200, pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        last_hidden_outputs = bert_outputs[0]
        print("hidden:", last_hidden_outputs)
        outputs = last_hidden_outputs[:, 0, :]
        print("predict:", outputs)
        fc_output = self.fullC(outputs)
        return fc_output, last_hidden_outputs


