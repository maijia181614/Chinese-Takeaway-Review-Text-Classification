import torch.nn as nn
import torch.nn.functional as F
import torch

data_path = './dataset/'
save_path = './model/'

BERT_MODEL = torch.load(save_path + 'bert_model_train.pth')
NUM_FILTERS = 256
EMBEDDING_DIM = 768
DROPOUT = 0.1
NUM_CLASSES = 2
FILTER_SIZES = [2, 3, 4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BERT_MODEL
        self.bert.to(device)
        for name ,param in self.bert.named_parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        out = self.bert(input)[1].unsqueeze(1)
        print("forward out:")
        print(out)
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        out = self.dropout(out)
        pred = self.linear(out)
        print(pred)
        tempOutput = []
        return self.linear(out), tempOutput