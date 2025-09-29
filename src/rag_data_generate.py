import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from tqdm import *
import json
import os

BASE_DIR = os.path.dirname(__file__)
class DataProcess():
    def __init__(self, text_data,name,token=100):
        self.original_data = text_data
        self.token = token
        self.json_data = []
        self.file_name = name

    # テキストを分割
    def text_split(self):
        str_len = len(self.original_data)
        result_list = []
        if str_len % self.token == 0:
            block = str_len // self.token
        else:
            block = str_len // self.token + 1
        
        for i in range(block):
            start = i * self.token
            end = (i + 1) * self.token
            if end > str_len:
                end = str_len
            # block log
            print(f'block {i+1}: from {start} to {end}')
            print(self.original_data[start:end])
            result_list.append(self.original_data[start:end])
    
        return result_list
    
    #　ベクトル化と索引保存
    def embedding_process(self, text_list):
        json_data = []
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        out = model.encode('test')
        #print(out, out.shape)
        dimension = out.shape[0]  # all-MiniLM-L6-v2 の　embedding 次元数
        index = faiss.IndexFlatIP(dimension)
        for idx, i in tqdm(enumerate(text_list)):
        
            out = model.encode([i])#.shape[1]
            #out = faiss.normalize_L2(out)
            index.add(np.array(out, dtype=np.float32))
            # オリジナルデータ保存
            json_data.append({"index":idx,"text":i,"product":self.file_name})
        # index save
        try:
            faiss.write_index(index, f"{BASE_DIR}/data/index_data/{self.file_name}.index")
        except:
            faiss.write_index(index, f"./data/{self.file_name}.index")
        print('index saved!')
        # json save
        try:
            with open(f'{BASE_DIR}/data/index_original_data/{self.file_name}.json', 'w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        except:
            with open(f'./data/{self.file_name}.json', 'w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        print('json saved!')
   
    def run(self):
        text_list = self.text_split()
        self.embedding_process(text_list)
      
if __name__ == "__main__":
    # すべてのファイルを処理
    for file in os.listdir('./data/item_original_data/'):
        with open(f'./data/item_original_data/{file}', 'r', encoding='utf-8') as f:
            text_data = f.read()
        data_process = DataProcess(text_data, file.split('.')[0],token=200)
        data_process.run()