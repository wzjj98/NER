#!/usr/bin/env python3

import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from bert4torch.models import  BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, EarlyStopping, get_pool_emb
from bert4torch.callbacks import AdversarialTraining
from bert4torch.optimizers import extend_with_exponential_moving_average, get_linear_schedule_with_warmup, Lion
from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer
from transformers import BertModel,AutoTokenizer
from PromptModel import PromptTable


maxlen = 150
batch_size = 16

# 加载标签字典
categories_label2id = {"PER": 1, "BOOK": 2, "OFI": 3}
categories_id2label = dict((value, key) for key, value in categories_label2id.items())
ner_vocab_size = len(categories_label2id)
ner_head_size = 64

# BERT base
config_path = './GujiBERT/config.json'
checkpoint_path = './GujiBERT/pytorch_model.bin'
dict_path = './GujiBERT/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
#seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

#生产格式数据
def labelgen(text, categories_label2id):
    ind = 3
    label = []
    flag = 1
    for i, char in enumerate(text):
        if text[i] == '{':
            s = i + 1
            flag = 0
        elif flag == 1 and text[i] != '{':
            ind += 1
        elif flag == 0 and text[i] != '}':
            continue
        elif text[i] == '}':
            e = i
            span, lab = text[s:e].split('|')
            # print(t,lab)
            lab = categories_label2id[lab]
            label.append([ind, ind + len(span) - 1, lab])
            ind += len(span)
            flag = 1
    t = '人书职'
    for i, char in enumerate(text):
        if char == '{' or char == '}' or char == '|' or char == 'P' or char == 'E' \
                or char == 'E' or char == 'R' or char == 'O' or char == 'F' or char == 'I' \
                or char == 'B' or char == 'K':
            continue
        else:
            t += char
    return t, label

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        data = []
        d = [item.strip() for item in open(filename, 'r').readlines()]
        for index, content in enumerate(d, 1):
            if content.startswith(u'\ufeff'):
                content = content.encode('utf8')[3:].decode('utf8')
            text, label = labelgen(content, categories_label2id)
            data.append((text, label))  # label为[[start, end, entity], ...]
        return data
    
def collate_fn(batch):
    batch_token_ids = []
    batch_entity_labels = []
    for i, (senttext, text_labels) in enumerate(batch):
        tokens = tokenizer.tokenize(senttext, maxlen=maxlen)
        mapping = tokenizer.rematch(senttext, tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros((250, 250))
        for start, end, label in text_labels:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start, end] = 1
                labels[end, start] = 1
                labels[start, label] = 1
                labels[label, end] = 1
        batch_token_ids.append(token_ids)
        batch_entity_labels.append(labels[:len(token_ids), :len(token_ids)])
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2), dtype=torch.float,
                                       device=device)
    return batch_token_ids, batch_entity_labels

# 转换数据集
train_dataloader = DataLoader(MyDataset('./GuNERdata/train4.txt'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset('./GuNERdata/valid4.txt'), batch_size=batch_size, collate_fn=collate_fn)

# 定义模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("./GujiBERT")
        self.prompttable = PromptTable(hidden_size=768,head_size=ner_head_size)
        
    def forward(self, token_ids):
        sequence_output = self.bert(input_ids=token_ids, output_hidden_states=True)['last_hidden_state']
        logit = self.prompttable(sequence_output, token_ids.gt(0).long())
        return logit

model = Model().to(device)

class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


optimizer = optim.Adam(model.parameters(), lr=1e-5)
ema_schedule = extend_with_exponential_moving_average(model, decay=0.99)
model.compile(
    loss=MyLoss(),
    optimizer=optimizer,
    scheduler=[ema_schedule],
)


def extract(output, threshold=0):
    S = set()
    for start, end in zip(*np.where(output.cpu() > threshold)):
        if start <= end:
            for rel in range(1, 4):
                if output[start, rel] > threshold and output[rel, end] > threshold and output[end, start] > threshold and output[start, end] > threshold:
                    S.add((start, end, rel))
    return S

def evaluate(data):
    X, Y, Z = 0, 1e-10, 1e-10
    for x_true, label in data:
        scores = model.predict(x_true)
        for i, score in enumerate(scores):
            R = extract(score, threshold=0)
            T = extract(label[i], threshold=0)
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(Callback):
    """
    评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        ema_schedule.apply_ema_weights()  # 使用滑动平均的ema权重
        f1, precision, recall = evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./bestmodel/model4.pt')
        ema_schedule.restore_raw_weights()  # 恢复原来模型的参数
        logs['f1'] = f1  # 这个要设置，否则EarlyStopping不生效
        print(f'[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_val_f1:.5f}')


if __name__ == '__main__':
    callbacks = [
        Evaluator(),
        EarlyStopping(monitor='f1', patience=5, verbose=1, mode='max'),  # 需要在Evaluator后面
        AdversarialTraining('fgm')  # fgm, pgd, vat, gradient_penalty
    ]
    model.fit(train_dataloader, steps_per_epoch=None, epochs=80, callbacks=callbacks)
else:

    model.load_weights('best_model.pt')
    