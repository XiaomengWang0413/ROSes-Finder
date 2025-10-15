import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

# 模型参数
MAX_SEQUENCE_LENGTH = 6269
batch_size = 20
class_num = 2
embedding_dim = 64
kernel_size = 20
channel_size = 2048
dropout_value = 0.5
att_dropout_val = 0.5
d_a = 100
r = 10
drop_con1 = 0.5
drop_con2 = 0.5
drop_con3 = 0.5

# 定义模型结构
class embedding_CNN_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_size, dropout_value=0.5,
                 MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, kernel_size=20, pool_kernel_size=0, stride=1,
                 channel_size=2048, att_dropout_val=0.5, d_a=100, r=100, drop_con1=0.0, drop_con2=0.0, drop_con3=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_value)
        self.drop_con1 = nn.Dropout(drop_con1)
        self.drop_con2 = nn.Dropout(drop_con2)
        self.drop_con3 = nn.Dropout(drop_con3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, channel_size, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=1)
        self.conv3 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=1)

        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.pool_kernel_size = pool_kernel_size

        self.fc = nn.Linear(channel_size * r, class_size)
        self.linear_first = nn.Linear(channel_size, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.att_dropout = nn.Dropout(att_dropout_val)
        self.r = r

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        xb = self.drop_con1(F.relu(self.conv1(x)))
        out = xb
        out = out.permute(0, 2, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.att_dropout(x)
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ out
        out = self.fc(sentence_embeddings.view(x.size(0), -1))
        return out

# 创建模型实例
MAX_NB_CHARS = 21
model = embedding_CNN_attention(MAX_NB_CHARS, embedding_dim, class_num, dropout_value=dropout_value,
                               MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, kernel_size=kernel_size,
                               pool_kernel_size=0, stride=1, channel_size=channel_size,
                               att_dropout_val=att_dropout_val, d_a=d_a, r=r,
                               drop_con1=drop_con1, drop_con2=drop_con2, drop_con3=drop_con3)

# 加载模型权重
model_path = '4_1yesORno.h5'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 加载数据
fake = torch.load("fa.pt")
query_dataset = TensorDataset(fake, torch.zeros(len(fake)))
query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

@torch.no_grad()
def get_probas(model, valid_dl):
    model.eval()
    scores = []
    F_softmax = torch.nn.Softmax(dim=1)
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        scores.append(F_softmax(y_hat.cpu()).numpy())
    return np.concatenate(scores)

# 进行预测
predict_proba = get_probas(model, query_dataloader)
print(predict_proba)

# 获取预测类别
clsid_to_class_name = {0: '0', 1: '1'}
prediction_list = []
for i in range(len(predict_proba)):
    prediction_list.append(np.argmax(predict_proba[i]))
pred_label = [clsid_to_class_name[i] for i in prediction_list]

# 保存结果
f1 = pd.DataFrame(pred_label)
f1.to_csv("cnn_2class.out", sep=" ", index=False, header=False)
