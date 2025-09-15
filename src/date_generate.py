import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from tqdm import *


def data_process(token=100, data_name='./data/test_data.csv'):
    data = pd.read_csv(data_name)
    data['intro'].to_list()[0]
    token = 100
    intro = data['intro'].to_list()[0]
    #print(intro)
    print(len(intro))
    result_list = []
    if len(intro) % 100 == 0:
        block = len(intro) // 100
    else:
        block = len(intro) // 100 + 1
    
    for i in range(block):
        start = i * token
        end = (i + 1) * token
        if end > len(intro):
            end = len(intro)
        print(f'block {i+1}: from {start} to {end}')
        print(intro[start:end])
        result_list.append(intro[start:end])
    # 返回分块后的文本列表
    return result_list

def embedding_process(text_list):
    stored_texts = []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out = model.encode('test')
    print(out, out.shape)
    dimension = out.shape[0]  # all-MiniLM-L6-v2 的 embedding 维度
    index = faiss.IndexFlatL2(dimension)
    for i in tqdm(text_list):
      
        out = model.encode([i])#.shape[1]
          # 用来存储原始文本
        index.add(np.array(out, dtype=np.float32))
    faiss.write_index(index, "./data/i17_vector.index")
    print('index saved!')
    # 返回 embedding 结果
    # return out

item='iphone17'

result_list = data_process()
embedding_process(result_list)
