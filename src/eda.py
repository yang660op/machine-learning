import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FEATURE_COLS, REPORTS_DIR

sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze(train, test, val):
    print("\n========== 探索性数据分析 ==========")

    print(f"\n--- 数据集概览 ---")
    print(f"  训练集: {train.shape}, 测试集: {test.shape}, 验证集: {val.shape}")
    print(f"  总计: {train.shape[0] + test.shape[0] + val.shape[0]}")
    print(f"  特征数: {len(FEATURE_COLS)} 数值特征 + 1 目标变量")
    print(f"  目标: 干豆类别（7 类）")

    print(f"\n--- 数据质量问题 ---")
    raw_labels = pd.concat([train, test, val], ignore_index=True)[
        pd.concat([train, test, val], ignore_index=True)['Class'].notna()
    ]['Class'].unique()
    print(f"  唯一标签数: {len(raw_labels)}（预期 7）")

    train_nulls = train.isnull().sum()
    for col in train_nulls[train_nulls > 0].index:
        print(f"  - 缺失 '{col}': {train_nulls[col]}")
    print(f"  - 'Solidity' 数据类型: {train['Solidity'].dtype}（应为 float）")
    print(f"  - 'Compactness' 数据类型: {train['Compactness'].dtype}（应为 float）")

    print(f"\n--- 统计摘要 ---")
    num = train.select_dtypes(include=[np.number])
    desc = num.describe().T
    desc['range'] = desc['max'] - desc['min']
    desc['cv'] = desc['std'] / desc['mean']
    print(f"  {'特征':20s} {'均值':>10s} {'标准差':>10s} {'最小值':>10s} {'最大值':>10s}")
    print(f"  {'-'*62}")
    for idx in desc.index[:8]:
        print(f"  {idx:20s} {desc.loc[idx,'mean']:>10.2f} {desc.loc[idx,'std']:>10.2f} "
              f"{desc.loc[idx,'min']:>10.2f} {desc.loc[idx,'max']:>10.2f}")

    print(f"\n--- 特征相关性 ---")
    corr = num.corr()
    high_pairs = 0
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                high_pairs += 1
    print(f"  高度相关对 (|r|>0.9): {high_pairs}")

    print(f"\n--- 类别分布 ---")
    vc = train['Class'].value_counts()
    for k, v in vc.items():
        print(f"  {k:15s}: {v:5d} ({v / len(train) * 100:.2f}%)")

    _plot_feature_distributions(train, test, val)
    _plot_correlation_heatmap(corr)
    _plot_class_distribution(vc)

    return num


def _plot_feature_distributions(train, test, val):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS[:6]):
        for label, color, src in [('训练集', 'steelblue', train),
                                   ('测试集', 'coral', test),
                                   ('验证集', 'forestgreen', val)]:
            data = pd.to_numeric(src[col], errors='coerce').dropna()
            axes[i].hist(data, bins=60, alpha=0.4, label=label, color=color, density=True)
        axes[i].set_title(col, fontsize=11)
        axes[i].legend(fontsize=7)
    plt.suptitle('特征分布（训练集 / 测试集 / 验证集）', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/feature_distributions.png', dpi=150)
    plt.close()


def _plot_correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('特征相关性热力图', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/correlation_heatmap.png', dpi=150)
    plt.close()


def _plot_class_distribution(vc):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(vc)))
    ax.bar(range(len(vc)), vc.values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('数量')
    ax.set_title('类别分布（训练集）', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/class_distribution.png', dpi=150)
    plt.close()
