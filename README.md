# Neuro-Symbolic-Pharyngitis-Triage
Thank you for your interest in our research paper.
This repository contains the official implementation of **"From Black-Box to Glass-Box: A Knowledge-Constrained Neuro-Symbolic Approach to Medical Triage under Data Scarcity"**, which has published as a preprint on Research Square, here is the doi: https://doi.org/10.21203/rs.3.rs-8522643/v1
The system combines deep learning (YOLOv8) for image analysis with a causal Bayesian network for diagnostic reasoning, providing explainable and age-adaptive clinical decision support.

## 🚀 Quick Start (Google Colab)
You can run the demonstration directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lug2/LM-Pharyngitis-Autonomous-Triage/blob/main/colab_demo.ipynb)

## 📂 Repository Structure
- `datasets/`: Annotated pharynx image dataset (MIT License).
- `models/`: Pre-trained YOLOv8 segmentation weights.
- `src/`: Core inference logic (Reasoning Engine).
- `experiments/`: Benchmark scripts and figure generation code.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 📊 Run Benchmark

### Basic Usage
```bash
# Standard Benchmark (N=1000)
python experiments/Benchmark/runner.py --task standard
```

### Available Tasks
You can specify the following tasks using the `--task` option:
- `standard`: Fidelity check, Sensitivity/Specificity analysis
- `stress`: Robustness stress test (Noise tolerance)
- `ablation`: Ablation study (Component importance)
- `dca`: Decision Curve Analysis (Clinical utility)
- `breaking_point`: Breaking point analysis
- `comparative`: Comparison with baseline models
- `all`: Run all tasks

### Reviewer Options (CLI Overrides)
I'd love to deeply appreciate for reviewing this paper.
You can modify key parameters directly from the command line without editing config files.
This allows exact reproduction of the paper's experimental conditions (e.g., sensitivity analysis).

```bash
# Example: Run standard benchmark with 500 samples
python experiments/Benchmark/runner.py --task standard --n_samples 500

# Example: Run robustness test with higher noise
python experiments/Benchmark/runner.py --task stress --noise 0.8

# Example: Detailed Sensitivity Analysis (Reproducing Paper Conditions)
# Run with 10 steps, 200 samples per step
python experiments/Benchmark/runner.py --task standard --steps 10 --rob_samples 200
```

## ⚖️ License
- **Code**: MIT Liscense
- **Datasets**: MIT License

## 📧 Contact
If you have any questions or need further assistance, please don't hesitate to contact us.
- **First Author / Developer, Leon Moriguchi**: a7213738@gmail.com
If you want to reach the corresponding author, please refer to the paper.
(You can also create an Issue in this repository)

---

# 咽頭炎トリアージのためのニューロシンボリックAI
閲覧いただきありがとうございます。
本リポジトリは、**「From Black-Box to Glass-Box: A Knowledge-Constrained Neuro-Symbolic Approach to Medical Triage under Data Scarcity」** の公式実装です。Reseach Squareでプレプリントとして公開されました。リンクはこちら: https://doi.org/10.21203/rs.3.rs-8522643/v1
深層学習(YOLOv8)による画像解析と、因果ベイジアンネットワークによる推論を組み合わせ、説明可能で年齢に適応した臨床意思決定支援を提供します。

## 🚀 クイックスタート (Google Colab)
以下のリンクから、ブラウザ上でデモを実行できます。
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lug2/LM-Pharyngitis-Autonomous-Triage/blob/main/colab_demo.ipynb).

## 📂 フォルダ構成
- `datasets/`: 咽頭画像データセット (MITライセンス).
- `models/`: 学習済みYOLOv8モデル.
- `src/`: 推論エンジン.
- `experiments/`: ベンチマークおよび図表生成スクリプト.

## 📦 インストール
```bash
pip install -r requirements.txt
```

## 📊 ベンチマーク実行

以下のコマンドでベンチマークを実行できます。

### 基本的な使用法
```bash
# 標準ベンチマーク (N=1000)
python experiments/Benchmark/runner.py --task standard
```

### 利用可能なタスク一覧
`--task` オプションで以下のベンチマークを指定できます:
- `standard`: 忠実度検証、感度・特異度分析
- `stress`: ロバスト性ストレステスト (ノイズ耐性)
- `ablation`: アブレーション研究 (コンポーネント重要度)
- `dca`: 決定曲線分析 (臨床的有用性)
- `breaking_point`: 限界点分析
- `comparative`: ベースラインモデルとの比較
- `all`: 全て実行

### 査読者用オプション (CLI)
設定ファイルを編集することなく、主要なパラメータをコマンドラインから変更可能です。
これにより論文の条件(感度分析など)を正確に再現できます。

```bash
# 例: サンプル数500で実行
python experiments/Benchmark/runner.py --task standard --n_samples 500

# 例: ノイズレベル0.8でストレステスト
python experiments/Benchmark/runner.py --task stress --noise 0.8

# 例: 感度分析の詳細設定 (論文条件の再現)
# ステップ数10, 各ステップ200サンプルで実行
python experiments/Benchmark/runner.py --task standard --steps 10 --rob_samples 200
```

## ⚖️ ライセンス
- **コード**: MIT Liscence
- **データセット**: MIT License

## 📧 お問い合わせ
ご質問やご不明な点等ございましたら、お気軽にお問い合わせください。
- **筆頭著者・開発者**: a7213738@gmail.com
(本リポジトリのIssueでも受け付けています)


