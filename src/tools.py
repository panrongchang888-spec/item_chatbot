import faiss
import json
import torch
from zai import ZhipuAiClient
import os

BASE_DIR = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api_key = os.environ.get('GLM_API_KEY')
client = ZhipuAiClient(api_key=api_key)  # API Key

# chatglmの出力
def Chat_GLM(messages):
    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=messages,
        thinking={
            "type": "false",    # 思考モジュール
        },
        stream=False,
        max_tokens=4096,          # 最大 tokens
        temperature=0.6           # ランダム性調整
    )

    response = response.choices[0].message.content
    return response


# RAGでコンテキスト取得
def Rag_data_get(user_input,product_name, embed_model):
    # 元データの読み込み
    try:

        try:
            
            with open(f'{BASE_DIR}/data/index_original_data/{product_name}.json', 'r') as f:
                json_data = json.load(f)
            # rag用のベクトルデータベースの読み込み
            index_load = faiss.read_index(f"{BASE_DIR}/data/index_data/{product_name}.index")

        except:

            with open(f'./data/{product_name}.json', 'r') as f:
                json_data = json.load(f)
            # rag用のベクトルデータベースの読み込み
            index_load = faiss.read_index(f"./data/{product_name}.index")

    except FileNotFoundError:
        return "指定された商品データが見つかりません。別の商品を選択するか、商品データをアップロードしてください。", [], []
    # ユーザークエリのベクトル化と類似度検索
    user_question_emb = embed_model.encode([user_input])
    # D: score, I: index
    #faiss.normalize_L2(user_question_emb)
    D, I = index_load.search(user_question_emb, k=5)
    print('I--------->',I)
    print('D--------->',D)
    new_I = []
    new_D = []
    # 閾値0.４以上のものだけを抽出
    for idx, score in enumerate(D[0]):
        if score > 0.4:
            new_I.append(I[0][idx])
            new_D.append(D[0][idx])

    context = ''.join([json_data[i]['text'] for i in new_I])

    return context, new_I, new_D