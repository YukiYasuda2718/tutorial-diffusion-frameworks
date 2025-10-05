# tutorial-diffusion-frameworks

## セットアップ

### VSCode ユーザー

1. [Docker](https://www.docker.com/ja-jp/) をインストール
2. 拡張機能 [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) を VSCode に追加
3. コマンドパレットから `Dev Container: Rebuild Container` を選択
4. GPU が Docker から使える場合 `tutorial_gpu` を選択，そうでない場合 `tutorial_cpu` を選択

### Google Colab ユーザー

1. 以下をクリックして Google Colab を開く: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YukiYasuda2718/tutorial-diffusion-frameworks/blob/main/notebooks/run_on_your_google_colab.ipynb)
2. 開いたノートブックを Google Colab 上で実行する

## 実験方法

- Docker コンテナ内での実験方法を説明
- このチュートリアルを実行する場合，VSCode 上で Docker コンテナを立ち上げ実行する方法を推奨
- Google Colab 上でも，深層学習が可能なように作られているが，すべての機能を Colab で実行できるようにしたわけではない
- Colab ユーザーは，上で説明したように，リンクをクリックしてノートブックを実行するだけでよい

### 確率過程のデモ

### 深層学習

1. 訓練データを作成するため，[make_lorenz96_data.ipynb](./notebooks/make_lorenz96_data.ipynb) を実行する
2. 深層学習および結果の解析を行うため，[make_lorenz96_data.ipynb](./notebooks/train_and_test_ddpm.ipynb)
