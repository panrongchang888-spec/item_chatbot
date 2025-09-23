import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from tqdm import *
import json

class DataProcess():
    def __init__(self, text_data,name,token=100):
        self.original_data = text_data
        self.token = token
        self.json_data = []
        self.file_name = name

    def text_split(self):
        str_len = len(self.original_data)
        result_list = []
        if str_len % 100 == 0:
            block = str_len // 100
        else:
            block = str_len // 100 + 1
        
        for i in range(block):
            start = i * self.token
            end = (i + 1) * self.token
            if end > str_len:
                end = str_len
            # block log
            print(f'block {i+1}: from {start} to {end}')
            print(self.original_data[start:end])
            result_list.append(self.original_data[start:end])
        # 返回分块后的文本列表
        return result_list
    

    def embedding_process(self, text_list):
        json_data = []
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        out = model.encode('test')
        #print(out, out.shape)
        dimension = out.shape[0]  # all-MiniLM-L6-v2 的 embedding 维度
        index = faiss.IndexFlatL2(dimension)
        for idx, i in tqdm(enumerate(text_list)):
        
            out = model.encode([i])#.shape[1]
            # 用来存储原始文本
            index.add(np.array(out, dtype=np.float32))
            json_data.append({"index":idx,"text":i,"product":self.file_name})
        # index save
        faiss.write_index(index, f"./data/index_data/{self.file_name}.index")
        print('index saved!')
        # json save
        with open(f'./data/index_original_data/{self.file_name}.json', 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print('json saved!')
   
    def run(self):
        text_list = self.text_split()
        self.embedding_process(text_list)
      
