import faiss
import json
import torch
from zai import ZhipuAiClient
import os

BASE_DIR = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api_key = os.environ.get('GLM_API_KEY')
client = ZhipuAiClient(api_key=api_key)  # 请填写您自己的 API Key
# chatglmの出力
def Chat_GLM(messages):
    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=messages,
        thinking={
            "type": "false",    # 启用深度思考模式
        },
        stream=False,
        max_tokens=4096,          # 最大输出 tokens
        temperature=0.6           # 控制输出的随机性
    )
    #print('response--->',response)
    response = response.choices[0].message.content
    #print('response--->',response)
     # 获取完整回复
                #print(response.choices[0].message)

                # for chunk in response:
                #     if chunk.choices[0].delta.content:
                #         print(chunk.choices[0].delta.content, end='')

               
                #answer = gemini_chat(messages.format(user_input,context))
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
            with open(f'/tmp/{product_name}.json', 'r') as f:
                json_data = json.load(f)
            # rag用のベクトルデータベースの読み込み
            index_load = faiss.read_index(f"/tmp/{product_name}.index")
    except FileNotFoundError:
        return "指定された商品データが見つかりません。別の商品を選択するか、商品データをアップロードしてください。", [], []
    # rag用のベクトルデータベースの読み込み
    #index_load = faiss.read_index(f"{BASE_DIR}/data/index_data/{product_name}.index")
    # ユーザークエリのベクトル化と類似度検索
    user_question_emb = embed_model.encode([user_input])
    # D: score, I: index
    D, I = index_load.search(user_question_emb, k=2)
    print('I--------->',I)
    print('D--------->',D)
    context = ''.join([json_data[i]['text'] for i in I[0]])
    return context, I, D