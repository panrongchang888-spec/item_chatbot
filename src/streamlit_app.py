import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from gemini import gemini_chat
import faiss
from sentence_transformers import SentenceTransformer

# Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:


# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame({
#     "x": x,
#     "y": y,
#     "idx": indices,
#     "rand": np.random.randn(num_points),
# })

# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with open('/app/src/data/iphone17_data.txt', 'r') as f:
        lines = f.readlines()
    index_load = faiss.read_index("/app/src/data/i17_vector.index")
    st.set_page_config(page_title="商品智能客服", layout="centered")
    st.title("🛒 商品智能问答助手")

    #conn = init_db()
    text = 'Unibodyの筐体。Proにふさわしい鍛え抜かれたボディ。iPhone 17 ProとiPhone 17 Pro Max、登場。史上最もパワフルなiPhoneを生み出すために細部まで徹底的に設計しました。新しいデザインの中心は、熱間鍛造アルミニウムUnibody。この筐体が、パフォーマンス、バッテリー容量、そして耐久性の限界を打ち破りますiPhone 17 Proのカメラシステムには、イノベーションが惜しみなく注ぎ込まれています。その一つが、iPhone史上最高の8倍望遠。最大200mmの焦点距離に相当するこの望遠機能は、次世代のテトラプリズムデザインと56%大きくなったセンサーによって実現しました。'
    
    messages = "ユーザーの問題：{}，あなたは今、人工カスタマーサポートとして振る舞います。ユーザーの質問に対して、私が提供する内容を完全に参考にして人間らしい応答を行ってください。関連する情報がない場合は、「申し訳ございません」と回答してください。以下は参考内容です【{}】"
    # 选择商品
    products = ["iPhone 17", "MacBook Pro", "AirPods Pro"]
    product_name = st.selectbox("请选择商品：", products)

    if product_name:
        #description = get_product_info(conn, product_name)
        st.write(f"この商品について質問してください: **{product_name}**")

        # 聊天区域
        user_question = st.text_input("質問：")
        if user_question:
            #product = {"name": product_name, "description": description}
            #answer = ask_llm(product, user_question)
            user_question_emb = model.encode([user_question])
            D, I = index_load.search(user_question_emb, k=2)
            context = ''.join([lines[i] for i in I[0]])
            answer = gemini_chat(messages.format(user_question,context))
            st.markdown(f"**🤖 回答：** {answer}")

if __name__ == "__main__":
    main()