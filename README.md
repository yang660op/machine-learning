# Dry Bean Classification System

基于 Dry Bean Dataset 的完整机器学习工程项目，涵盖数据分析、数据处理、多算法实验对比与系统展示。

---

## 目录

- [数据说明](#数据说明)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [数据处理方法](#数据处理方法)
- [实现的算法](#实现的算法)
- [实验结果](#实验结果)

---

## 数据说明

### 数据集来源

Dry Bean Dataset 包含 13,611 个样本，每个样本通过图像处理提取 16 个几何形态特征，用于区分 7 种不同的干豆品种。

### 特征列表

| 特征 | 描述 |
|------|------|
| Area | 豆粒区域面积（像素） |
| Perimeter | 豆粒周长 |
| MajorAxisLength | 长轴长度 |
| MinorAxisLength | 短轴长度 |
| AspectRation | 长宽比 |
| Eccentricity | 离心率 |
| ConvexArea | 凸包面积 |
| EquivDiameter | 等效直径 |
| Extent | 外接矩形占比 |
| Solidity | 坚固度（面积/凸包面积） |
| roundness | 圆度 |
| Compactness | 紧密度 |
| ShapeFactor1~4 | 形状因子 1~4 |

### 目标类别（7 类）

| 类别 | 训练集样本数 | 占比 |
|------|------------|------|
| DERMASON | 2,482 | 26.05% |
| SIRA | 1,818 | 19.08% |
| SEKER | 1,395 | 14.64% |
| HOROZ | 1,326 | 13.92% |
| CALI | 1,136 | 11.92% |
| BARBUNYA | 916 | 9.61% |
| BOMBAY | 360 | 3.78% |

### 数据污染问题

原始数据存在以下质量问题：

1. **标签噪声** — 共 25 种不同标签（期望 7 类），包括大小写不一致（`DERMASON` vs `dermason`）、字符替换（`D3RMAS0N`、`S3K3R`、`H0R0Z`、`B0MBAY`）、尾部空格（`DERMASON `）等问题
2. **缺失值** — `Perimeter`（469 个）、`Solidity`（272 个）存在缺失
3. **类型异常** — `Solidity`、`Compactness` 两列存储为字符串类型而非浮点数

### 数据划分

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| 训练集 | 9,527 | 模型训练 |
| 测试集 | 2,737 | 模型评估 |
| 验证集 | 1,347 | 损失曲线监控 |

---

## 项目结构

```
期末作业/
├── main.py              # CLI 统一入口（支持 4 种运行模式）
├── app.py               # Streamlit 交互式展示系统
├── data/                # 原始数据集
│   ├── Dry_Bean_Dataset_Dirty_train.csv
│   ├── Dry_Bean_Dataset_Dirty_test.csv
│   └── Dry_Bean_Dataset_Dirty_val.csv
├── models/              # 训练产物
│   ├── best_model.pkl           # 最优模型（调参后 Random Forest）
│   ├── scaler.pkl               # 标准化器
│   ├── label_encoder.pkl        # 标签编码器
│   └── best_params.json         # 最优超参数
├── reports/             # 报告与可视化图表
│   ├── model_comparison.csv     # 3 算法对比表
│   ├── robustness_analysis.csv  # 鲁棒性分析数据
│   ├── inference_speed.csv      # 推理速度数据
│   ├── loss_curve_xgboost.png   # XGBoost 损失曲线
│   ├── overfitting_analysis.png # 过拟合分析图
│   ├── correlation_heatmap.png  # 特征相关性热力图
│   ├── cm_*.png                 # 各算法混淆矩阵
│   └── ...
└── src/                 # 模块化源码
    ├── config.py        # 路径/常量/标签映射
    ├── data_loader.py   # 数据加载
    ├── preprocessor.py  # 清洗/转换/填充/缩放
    ├── eda.py           # EDA 分析
    ├── trainer.py       # 模型训练/调参
    ├── evaluator.py     # 评估：损失曲线/速度/鲁棒性/过拟合
    └── pipeline.py      # 流程编排
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib

# 2. 运行完整流水线（EDA + 训练 + 评估）
python main.py full

# 3. 快捷模式（仅训练）
python main.py quick

# 4. 启动展示系统
streamlit run app.py
```

### CLI 命令说明

| 命令 | 功能 |
|------|------|
| `python main.py full` | 完整流程：EDA → 预处理 → 4 算法训练 → 损失曲线/速度/鲁棒性/过拟合分析 |
| `python main.py quick` | 快速训练模式，仅运行数据预处理 + 模型训练 |
| `python main.py train` | 仅执行数据预处理和模型训练，不进行后续评估 |
| `python main.py eval` | 仅执行评估流程，需已有训练好的模型 |

---

## 数据处理方法

### 1. 标签标准化

将 25 种混乱标签统一映射为 7 个标准类别：

```
DERMASON, dermason, D3RMAS0N, DERMASON(含尾随空格)  →  DERMASON
SIRA, sira, SIRA(含尾随空格)                          →  SIRA
SEKER, seker, S3K3R, SEKER(含尾随空格)                →  SEKER
HOROZ, horoz, H0R0Z, HOROZ(含尾随空格)                →  HOROZ
CALI, cali, CALI(含尾随空格)                          →  CALI
BARBUNYA, barbunya, BARBUNYA(含尾随空格)              →  BARBUNYA
BOMBAY, bombay, B0MBAY, BOMBAY(含尾随空格)            →  BOMBAY
```

### 2. 数据类型转换

- `Solidity`：`object` → `float64`，无法转换的置为 NaN
- `Compactness`：`object` → `float64`，无法转换的置为 NaN

### 3. 缺失值处理

使用训练集的中位数进行填充：

| 特征 | 填充值（中位数） |
|------|----------------|
| Perimeter | 794.336 |
| Solidity | 0.9883 |
| Compactness | 0.8011 |

### 4. 特征缩放

使用 StandardScaler（Z-score 标准化）：

$$x' = \frac{x - \mu}{\sigma}$$

使每个特征服从均值为 0、标准差为 1 的分布。距离类模型（KNN、SVM、Logistic Regression）必须使用，树模型（RF、XGB、DT）虽不受尺度影响但统一处理以保证一致性。

---

## 实现的算法

共实现 4 种多分类算法：

| 算法 | 类别 | 特点 |
|------|------|------|
| **SVM (RBF)** | 线性核 | RBF 核，擅长高维决策边界，自动概率估计 |
| **ANN (MLP)** | 神经网络 | 3 层全连接 (128→64→7)，ReLU + Adam，自动早停 |
| **XGBoost** | 集成学习（Boosting） | 200 轮迭代，梯度提升，learning_rate=0.1 |
| **Logistic Regression** | 线性模型 | 可解释性强，训练快速，天然支持概率输出 |

### 超参数调优

对 ANN (MLP) 进行 GridSearchCV 网格搜索：

```json
{
  "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
  "activation": ["relu", "tanh"],
  "learning_rate_init": [0.001, 0.01]
}
```

**最优参数**: 由 GridSearchCV 自动确定（结果保存于 `models/best_params.json`）

---

## 实验结果

### 各算法精度对比

| 模型 | 测试精度 | 训练精度 | 过拟合差距 | F1-Score | 5-Fold CV | 训练时间(s) |
|------|---------|---------|-----------|---------|-----------|-----------|
| **SVM (RBF)** | **0.9288** | 0.9299 | 0.0011 | **0.9288** | 0.9241 | 5.09 |
| XGBoost | 0.9280 | 0.9966 | 0.0686 | 0.9281 | 0.9263 | 4.54 |
| ANN (MLP) | 0.9218 | 0.9273 | 0.0054 | 0.9220 | 0.9256 | 3.02 |
| Logistic Regression | 0.9185 | 0.9250 | 0.0064 | 0.9191 | 0.9216 | 7.47 |

### 推理速度对比

| 模型 | 平均推理时间(ms) | 吞吐量(samples/s) |
|------|-----------------|-------------------|
| Logistic Regression | 0.49 | 5,590,894 |
| XGBoost | 9.68 | 280,263 |
| ANN (MLP) | 1.25 | 2,189,600 |
| SVM (RBF) | 664.59 | 4,082 |

### 鲁棒性对比

| 模型 | Clean | Gaussian σ=0.1 | Gaussian σ=0.5 | Gaussian σ=1.0 | Gaussian σ=2.0 |
|------|-------|---------------|---------------|---------------|---------------|
| SVM (RBF) | 0.9285 | 0.9244 | 0.8540 | 0.6992 | 0.2230 |
| XGBoost | 0.9263 | 0.9141 | 0.7269 | 0.5330 | 0.3550 |
| ANN (MLP) | 0.9249 | 0.9200 | 0.8350 | 0.6850 | 0.4100 |
| Logistic Regression | 0.9182 | 0.9145 | 0.8002 | 0.6086 | 0.4128 |

### 过拟合分析

| 模型 | 过拟合差距 | 判定 |
|------|-----------|------|
| SVM (RBF) | 0.0011 | ✅ 优秀泛化能力 |
| ANN (MLP) | 0.0054 | ✅ 良好拟合 |
| Logistic Regression | 0.0064 | ✅ 良好拟合 |
| XGBoost | 0.0686 | ⚠️ 过拟合（可增加正则化） |

---

## 结论

1. **最优模型**：SVM (RBF) 以 92.88% 测试精度和 0.09% 过拟合差距成为综合表现最佳的模型
2. **最佳速度**：Logistic Regression 推理速度最快（0.49ms），适合生产部署
3. **最佳平衡**：Random Forest 经过调参后（max_depth=20, n_estimators=200）兼顾精度与鲁棒性
