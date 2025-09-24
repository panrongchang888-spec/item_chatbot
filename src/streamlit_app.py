import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from rag_data_generate import DataProcess
from intention_train import predict
import torch
from tools import Chat_GLM, Rag_data_get
import time
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Starting the application...")


# 在庫の商品
products = []
for i in os.listdir('/src/data/item_original_data/'):
    if i.endswith('.txt') or i.endswith('.pdf'):
        products.append(i.split('.')[0])

# 意図ラベル
label_nname = ['機能相談','無間']
# ファイルアップロード
st.sidebar.header("商品ファイルアップロード")

uploaded_file = st.sidebar.file_uploader(
    "ファイルをアップロード (TXT/PDF)", type=["txt", "pdf"]
)

if uploaded_file and uploaded_file.name.split('.')[0] not in products:
    with st.spinner("ファイルを処理中..."):
        name = uploaded_file.name.split('.')[0]
        products.append(name)
        with open(f'/src/data/item_original_data/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.getbuffer())
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

def chat(user_input,product_name,messages, embed_model):
    # 会話歴史の変数
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'intentions' not in st.session_state:
        st.session_state['intentions'] = []
    if 'rag_data' not in st.session_state:
        st.session_state['rag_data'] = []
    # 会話履歴の表示
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    
    if user_input:
        
        # ユーザーのメッセージ
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        # ユーザーの意図を予測
        intent_label = predict([user_input])
        st.session_state.intentions.append(intent_label)
        context, indexs, scores = Rag_data_get(user_input,product_name, embed_model)
        print("scores--->",scores)
        # 意図が機能相談かどうかを再確認
        for i in scores[0]:
            if i > 0.8:
                intent_label = 0
        if intent_label == 0:  # 機能相談
            # RAGで回答生成
            st.session_state.rag_data.append({"content":context, "index":indexs.tolist()})
            try:
                messages.append({"role": "user", "content":f"以下は参考内容です【{context}】。これを参考にして、ユーザーの質問に答えてください。質問：{user_input}"})
                
                answer = Chat_GLM(messages)
               
            except Exception as e:
                answer = "申し訳ございません、現在モデルがオーバーロード。もう一度お試しください。"
                print("Error :", e)
                st.chat_message("assistant").write(answer)
                return 
        
        elif intent_label == 1: # 雑談
            answer = '申し訳ございません、私は商品に関する質問にのみ対応しています。'
        
        else: # 無間
            answer = '申し訳ございません、私は商品に関する質問にのみ対応しています。'
        
        with st.sidebar:
            st.header("意図＆RAG内容")
            for idx, (intent, rag) in enumerate(zip(st.session_state.intentions, st.session_state.rag_data)):
                st.subheader(f"会話 {idx+1}")
                st.write(f"意図: {label_nname[intent]}")
                st.write(f"RAG索引: {rag['index']}")
                for i, line in enumerate(rag["content"].split('\n')):
                    st.write(f"RAG内容 {i+1}: {line}")

        
        # 回答の表示、ストリーミング
        with st.chat_message("assistant"):
            st.write(f"user intent{label_nname[intent_label]}")
            placeholder = st.empty()
            text = ''
            for char in answer:
                text += char
                placeholder.write(text) 
                time.sleep(0.05)   
            
            #st.chat_message("assistant").write(i)
        #st.chat_message("assistant").write(f"user intent{label_nname[intent_label]}\\\nanswer:{answer}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "assistant", "content": answer})


def main():

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    st.set_page_config(page_title="知能会話", layout="centered")
    st.title("🛒 商品問題助手")

    #conn = init_db()
    text = 'Unibodyの筐体。Proにふさわしい鍛え抜かれたボディ。\
        iPhone 17 ProとiPhone 17 Pro Max、登場。史上最もパワフルなiPhoneを生み出すために細部まで徹底的に設計しました。新しいデザインの中心は、熱間鍛造アルミニウムUnibody。この筐体が、パフォーマンス、バッテリー容量、そして耐久性の限界を打ち破りますiPhone 17 Proのカメラシステムには、イノベーションが惜しみなく注ぎ込まれています。その一つが、iPhone史上最高の8倍望遠。最大200mmの焦点距離に相当するこの望遠機能は、次世代のテトラプリズムデザインと56%大きくなったセンサーによって実現しました。'
    # prompt
    messages = [{"role":"system","content":"あなたは今、人工カスタマーサポートとして振る舞います。ユーザーの質問に対して、\
        私が提供する内容を完全に参考にして人間らしい応答を行ってください。関連する情報がない場合は、\
            「申し訳ございません」と回答してください。"}
            ]
    
    # 选择商品
    product_name = st.selectbox("商品を選んでください", products)

    if product_name:
        #description = get_product_info(conn, product_name)
        st.write(f"この商品について質問してください: **{product_name}**")
        user_input = st.chat_input("質問：")
        print("user_input--->",user_input)
        chat(user_input,product_name,messages, embed_model)

if __name__ == "__main__":
    main()