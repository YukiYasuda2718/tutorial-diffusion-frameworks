# tutorial-diffusion-frameworks

## セットアップ

### VSCode ユーザー

1. [Docker](https://www.docker.com/ja-jp/) をインストール
2. 拡張機能 [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) を VSCode に追加
3. コマンドパレットから `Dev Container: Rebuild Container` を選択
4. GPU が Docker から使える場合 `tutorial_gpu` を選択，そうでない場合 `tutorial_cpu` を選択

### Google Colab ユーザー

## 実験方法

### 確率過程のデモ

### 深層学習

1. 訓練データを作成するため，[make_lorenz96_data.ipynb](./notebooks/make_lorenz96_data.ipynb) を実行する
2. 深層学習および結果の解析を行うため，[make_lorenz96_data.ipynb](./notebooks/train_and_test_ddpm.ipynb)
