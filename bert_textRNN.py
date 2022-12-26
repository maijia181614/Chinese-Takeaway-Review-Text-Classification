import torch.nn as nn
import torch.nn.functional as F
import torch

data_path = './dataset/'
save_path = './model/'

BERT_MODEL = torch.load(save_path + 'bert_model_train.pth')

EMBEDDING_DIM = 768
DROPOUT = 0.1
RNN_HIDDEN_DIM = 356
NUM_LAYERS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BERT_MODEL
        self.bert.to(device)
        for name ,param in self.bert.named_parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(EMBEDDING_DIM, RNN_HIDDEN_DIM, NUM_LAYERS, batch_first=True,
                            bias=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(RNN_HIDDEN_DIM * 2, 2)

    def forward(self, input):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        out = self.bert(input)[1]
        print("forward bert out:")
        print(out)
        lstm_out, (hidden_last, cn_last) = self.lstm(out)
        hidden_last_l = hidden_last[-2]
        hidden_last_r = hidden_last[-1]
        hidden_last_out = torch.cat([hidden_last_l, hidden_last_r], dim=-1)
        hidden_last_out = self.dropout(hidden_last_out)
        pred = self.fc(hidden_last_out)
        print("forward rnn pred:")
        print(pred)
        tempOutput = []
        return pred, tempOutput