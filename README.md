# tutorial-diffusion-frameworks

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-informational)](https://pytorch.org/)

深層生成モデルの一つである「拡散モデル」は，その推論精度の高さから現在盛んに応用されている．拡散モデルは確率解析と密接に結びついている．確率解析とは，確率的なノイズに駆動される時間発展を記述する体系である．確率解析を習得することで，拡散モデルの理解と実装が容易になる．本セミナーでは，伊藤の公式や確率微分方程式などの確率解析の基礎から説明をはじめ，拡散モデルの動作原理を説明する．セミナーでは簡単なコードを配布することで，確率解析や拡散モデルを身近なものに感じてもらえる工夫を行う．拡散モデルの応用例として，スコアベースデータ同化を扱う．

- [理論ノート](./docs/theoretical_note_on_diffusion_model.pdf)
- [スライド](./docs/seminar_20251009.pdf)

## セットアップ

### VSCode ユーザー

1. [Docker](https://www.docker.com/ja-jp/) をインストール
2. 拡張機能 [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) を VSCode に追加
3. コマンドパレットから `Dev Container: Rebuild Container` を選択
4. GPU が Docker から使える場合 `tutorial_gpu` を選択，そうでない場合 `tutorial_cpu` を選択

- コンテナイメージは GitHub レジストリからダウンロードするため，ローカルでのビルドは必要ない
- ローカルでビルドしたい場合，`./docker-compose.yml` を例えば以下のように書き換える
  - 必要ならば `tutorial_cpu` に対しても同じように行う
  - ネットワークの設定によっては，GitHub レジストリにアクセスできないこともある
  - その場合はローカルビルドする

<details><summary>書き換え方の例</summary>

```
# 書き換え前
tutorial_gpu:
  image: ghcr.io/yukiyasuda2718/tutorial-diffusion-frameworks:v1.0.0
```

```
# 書き換え後
tutorial_gpu:
    build:
      context: ./docker
```

</details>

### Google Colab ユーザー

1. 以下をクリックして Google Colab を開く: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YukiYasuda2718/tutorial-diffusion-frameworks/blob/main/notebooks/run_on_your_google_colab.ipynb)
2. 開いたノートブックを Google Colab 上で実行する

## 実験方法

- Docker コンテナ内での実験方法を説明
- このチュートリアルを実行する場合，VSCode 上で Docker コンテナを立ち上げ実行する方法を推奨
- Google Colab 上でも，実験が可能なように作られているが，ノートブックが巨大なので VSCode での実行を推奨
  - Colab ユーザーは，上で説明したように，リンクをクリックしてノートブックを実行するだけでよい

### 確率過程のデモ

- SDE の実行と拡散モデルの順過程と逆過程を調べる: [understand_sde_and_ddpm.ipynb](./notebooks/understand_sde_and_ddpm.ipynb)

### 拡散モデルの学習とテスト (無条件データ生成)

1. 訓練データを作成する: [make_lorenz96_data.ipynb](./notebooks/make_lorenz96_data.ipynb)
2. 深層学習および結果の解析を行う: [train_and_test_ddpm.ipynb](./notebooks/train_and_test_ddpm.ipynb)

### 条件付きデータ生成

- 無条件データ生成で学習した拡散モデルで条件付きデータ生成を行う: [perform_conditional_generation.ipynb](./notebooks/perform_conditional_generation.ipynb)
