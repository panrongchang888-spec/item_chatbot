---
title: Ai Project
emoji: 🚀
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: product assistant
---

太棒了 👍 你已经把流程图和文件说明都准备好了，我来帮你整理成一个 日语版 README 文件，结构清晰、符合日本职场/研究习惯，可以直接放到 GitHub 或 Hugging Face Space。

⸻



# 商品知能助手

以下の図は、商品の知能助手システムの全体フローを示しています。  

<img src="./src/images/flowchart.png" alt="システムフロー図" width="600">

## 概要
本プロジェクトは、ユーザーのクエリに基づき、商品関連の質問に対して適切な回答を生成するAIシステムです。  
意図分類、ベクトル検索（Faiss）、および大規模言語モデル（LLM）を組み合わせることで、効率的かつ正確な情報提供を実現します。  

## 特徴
- 意図分類によるクエリの自動仕分け  
- 商品データをベクトル化してインデックスを構築（Faiss使用）  
- 類似度検索に基づく関連情報の取得  
- 大規模モデル API を用いた自然言語応答生成  
- Streamlit による簡易UI提供  

## ファイル構成

├─ streamlit_app.py        # メインアプリ（StreamlitによるUI）
├─ intention_train.py      # 意図認識モデルの学習スクリプト
├─ rag_data_generate.py    # 商品データ処理およびベクトルインデックス生成
├─ tools.py                # 大規模モデルAPI呼び出し、ベクトル検索関連処理
├─ requirements.txt        # 必要な依存パッケージ
└─ README.md               # 本ファイル

## 動作環境
- Python >= 3.10
- Streamlit
- Faiss
- Transformers
- そのほか `requirements.txt` に記載のライブラリ  

## インストール方法
```bash
git clone https://github.com/username/project.git
cd project
pip install -r requirements.txt

実行方法

ローカル環境での実行
	1.	ChatGLM の API キーを取得する
	2.	環境変数に設定するか、tools.py 内で指定する
	3.	以下を実行

streamlit run streamlit_app.py

Hugging Face Space 上での実行
	•	追加の設定は不要で、そのまま利用可能です。

今後の改善方針
	1.	より適切な元データの分割方法を検討
	2.	新しいモジュールの追加
	•	商品推薦
	•	アフターサービス関連のサポート
	3.	自前のデプロイ環境が整った場合、大規模モデルAPIを自前モデルに置き換え

