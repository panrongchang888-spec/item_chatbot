from datasets import Dataset
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
from tqdm import *
from sklearn.metrics import classification_report
import os

BASE_DIR = os.path.dirname(__file__)
print("Preparing data...")
# all-MiniLM-L6-v2
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
label_name = ['機能相談','無間']
# 准备数据
def data_prepare():
    texts = []
    labels = []

    with open('japanese_dataset_clean.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    for idx, i in enumerate(original_data):
        texts.extend(i['examples'])
        labels.extend([idx]*len(i['examples']))

    print(texts)
    print(labels)

    # 转成 HuggingFace Dataset
    dataset = Dataset.from_dict({'text': texts, 'label': labels})

    # 划分训练/验证集
    dataset = dataset.train_test_split(test_size=0.2)

    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=10, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=1)
    return train_loader, valid_loader

# モデル定義
class IntentClassifier(nn.Module):
    def __init__(self):
        super(IntentClassifier, self).__init__()
        self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = nn.Linear(384, 2)
       
    def forward(self,texts):
        outputs = self.base_model.encode(texts)
        outputs = torch.tensor(outputs).to(torch.float32)
        logits = self.classifier(outputs)
        return logits
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# トレーニング
def train(train_loader, valid_loader):
    
    model = IntentClassifier().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for epoch in tqdm(range(10)):  # 演示只跑2个epoch
        
        for batch in train_loader:
            text_list = batch["text"]
            label_list = batch["label"]
            labels = torch.tensor(label_list)
            logits = model(text_list)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    # 验证集的准确度
    preds = [] # 予測値
    labels = [] # 正解値
    for batch in valid_loader:
        text_list = batch["text"]
        label_list = batch["label"]
        labels.extend(torch.tensor(label_list))
        with torch.no_grad():
            logits = model(text_list)
        preds.extend(torch.argmax(logits, dim=1))

    print(classification_report(labels, preds, target_names=label_name))

    # ========== 5. 保存模型 ==========
    torch.save(model.state_dict(), "intent_model.pt")
    print("模型已保存。")


# ========== 6. 加载模型并预测 ==========
# 加载

# 预测
def predict(text):
    loaded_model = IntentClassifier()
    loaded_model.load_state_dict(torch.load(f"{BASE_DIR}/intent_model.pt", map_location=device))
    loaded_model.eval()
  
    with torch.no_grad():
        logits = loaded_model(text)
        print("Logits:", logits)

    pred = torch.argmax(logits, dim=1).item()
    return pred

# test_text = ["この機械の使い方を教えてください。"]
# print("预测结果:", predict(test_text))  # 可能输出 0 (退货)

if __name__ == "__main__":
    train_loader, valid_loader = data_prepare()
    train(train_loader=train_loader, valid_loader=valid_loader)

    text1 = 'この機械の使い方を教えてください。'
    text2 = 'こんにちは、元気ですか？'
    print("预测结果:", predict([text1]))  
    print("预测结果:", predict([text2]))  
    