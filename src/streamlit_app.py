import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from rag_data_generate import DataProcess
from intention_train import predict
import torch
from tools import Tools
import time
import os
import json
import random

class MyChatbot():
    def __init__(self):
        self.tools = Tools()
        self.BASE_DIR = os.path.dirname(__file__)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.label_name = ['機能相談','無間']  
        #self.products = [i.split('.')[0] for i in os.listdir(f'{self.BASE_DIR}/data/item_original_data/') if i.endswith('.txt') or i.endswith('.pdf')]
        self.products = ['iphone17','iphone17pro','switch2']
        #self.product_question = {"iphone17pro":["カメラの特徴","新たな特徴"],"switch2":["Joy-Con 2は何だ"]}
    
    # 会話の初期化
    def chat_init(self,product_name):
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'intentions' not in st.session_state:
            st.session_state['intentions'] = []
        if 'rag_data' not in st.session_state:
            st.session_state['rag_data'] = []
        if 'last_product_name' not in st.session_state:
            st.session_state['last_product_name'] = product_name

    # 会話の再初期化
    def chat_init_second(self, product_name):
        
        st.session_state['messages'] = []
        st.session_state['intentions'] = []
        st.session_state['rag_data'] = []
        st.session_state['last_product_name'] = product_name

    # モデルのロード
    @st.cache_resource
    def load_model(self):
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        return embed_model
    
    # ファイルアップロードと処理
    def file_upload(self):
        st.sidebar.header("商品ファイルアップロード")
        uploaded_file = st.sidebar.file_uploader(
            "ファイルをアップロード (TXT/PDF)", type=["txt", "pdf"]
        )
        if uploaded_file:
            with st.spinner("ファイルを処理中..."):
                name = uploaded_file.name.split('.')[0]
                if name not in self.products:
                    self.products.append(name)
                uploaded_file.seek(0)
                # テキスト抽出
                if uploaded_file.type == "text/plain":
                    text_data = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    text_data = "\n".join([page.extract_text() for page in reader.pages])
                # データをベクトルデータベースに書き込む
                data_process = DataProcess(text_data, name)
                data_process.run()
            st.success("ファイルが正常にアップロードされ、処理されました！")

    # サイドバーに意図とRAGデータを表示
    def st_sliderbar(self):
        with st.sidebar:
            st.header("意図&検索データ")    
            for idx, (intent, rag) in enumerate(zip(st.session_state.intentions, st.session_state.rag_data)):
                st.subheader(f"会話 {idx+1}")
                st.write(f"意図: {self.label_name[intent['intention']]}")
                st.write(f"データ索引: {rag['index']}")
                st.write(f"データスコア: {rag['score']}")
                for i, line in enumerate(rag["data"].split('\n')):
                    st.write(f"検索データ {i+1}: {line}")
    
    # 会話履歴リセットボタン
    def clean_buttion(self):
        if st.sidebar.button("会話履歴リセット"):
            self.chat_init_second(self.products[0])
            st.session_state.selected = self.products[0]
            st.sidebar.success("会話履歴がリセットされました！")
            st.rerun()
    
    # チャット機能
    def chat(self,product_name, user_input, messages):
        # 会話の初期化
        self.chat_init(product_name=product_name)
        # 商品が変わった場合、会話履歴をリセット
        if product_name != st.session_state['last_product_name']:
            self.chat_init_second(product_name)
        # 会話の履歴表示
        for msg in st.session_state.messages:
            if msg['role'] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        
        if user_input:
            # ユーザーの入力を会話履歴に追加
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            # 意図の予測(モデルのストレス減少のため)
            intention, intent_score = predict(user_input)
            st.session_state.intentions.append({"intention": intention, "score": intent_score})
            # RAGデータの取得　
            rag_data, rag_index, rag_score = self.tools.get_rag_data(user_input, product_name, self.embed_model)
            st.session_state.rag_data.append({"data": rag_data, "index": rag_index, "score": rag_score})
            # システムメッセージの作成
            try:
                if rag_data:
                    if intention == 0: # 機能相談
                        messages.append({"role": "user", "content":f"以下は参考内容です【{rag_data}】。これを参考にして、ユーザーの質問に答えてください。質問：{user_input}"})
                        answer = self.tools.GLM_chat(messages)
                    else: # 無間
                        messages.append({"role": "user", "content":f"以下は参考内容です【{rag_data}】。これを参考にして、ユーザーの質問に答えてください。もし参考内容とユーザーの質問が関係ないなら、回答の内容は以下通り「申し訳ございません、私は商品に関する質問にのみ対応しています」質問：{user_input}"})
                        answer = self.tools.GLM_chat(messages)
                        #answer = '申し訳ございません、私は商品に関する質問にのみ対応しています。'
                else:
                    if intention == 0:
                        answer = "申し訳ございません,その質問に関する情報が見つかりませんでした。別の質問を試してみてください。"
                    else:
                        answer = '申し訳ございません、私は商品に関する質問にのみ対応しています。'

            except Exception as e:
                print("Error in Chat_GLM:", e)
                answer = "申し訳ございません、現在システムが混雑しています。後ほどもう一度お試しください。"
            # 回答の表示、ストリーミング
            with st.chat_message("assistant"):
                #st.write(f"user intent{label_nname[intent_label]}")
                placeholder = st.empty()
                text = ''
                for char in answer:
                    text += char
                    placeholder.write(text) 
                    time.sleep(0.05)   

            # サイドバーに意図と検索内容を表示
            self.st_sliderbar()
            # アシスタントの回答を会話履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "assistant", "content": answer})

    # アプリの実行
    def run(self):
        print('start app...')
        st.set_page_config(page_title="RAGチャットボット", page_icon=":robot_face:")
        st.title("商品説明のチャットボット")
        self.file_upload()
        self.clean_buttion()
        # サイドバーで商品を選択
        product_name = st.selectbox("商品を選択", self.products, key='selected')
        if product_name:
            product_question = json.loads(open(f'./data/question/{product_name}.json', 'r', encoding='utf-8').read())['questions']
            print('product_question-->',product_question)
            product_question = '\r'.join(random.sample(product_question, 3))
            print('product_question-->',product_question)
            st.write(f"この商品について質問してください: **{product_name}**")
            st.write(f"おそらくご質問されたい内容:**\\\n{product_question}**")
        # ユーザーの入力
        user_input = st.chat_input("質問を入力してください。")

        messages = [{"role":"system","content":"あなたは今、人工カスタマーサポートとして振る舞います。ユーザーの質問に対して、\
        私が提供する内容を完全に参考にして人間らしい応答を行ってください。関連する情報がない場合は、\
            「申し訳ございません」と回答してください。"}
            ]
        self.chat(product_name, user_input, messages)
    
if __name__ == "__main__":
    chatbot = MyChatbot()
    chatbot.run()