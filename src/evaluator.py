import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.config import REPORTS_DIR, RANDOM_SEED

sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss_curves(X_train, y_train, X_val, y_val):
    print("\n========== 损失曲线分析 ==========")

    eval_set = [(X_train, y_train), (X_val, y_val)]
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.1,
                        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
                        eval_metric='mlogloss', early_stopping_rounds=20)
    xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    results = xgb.evals_result()
    train_loss = results['validation_0']['mlogloss']
    val_loss = results['validation_1']['mlogloss']

    best_iter = np.argmin(val_loss) + 1
    print(f"  XGBoost: 最佳验证损失={val_loss[best_iter - 1]:.4f} @ 迭代 {best_iter}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_loss) + 1), train_loss, label='训练集', color='steelblue', linewidth=2)
    ax.plot(range(1, len(val_loss) + 1), val_loss, label='验证集', color='coral', linewidth=2)
    ax.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7, label=f'最佳 @ 迭代 {best_iter}')
    ax.set_xlabel('提升迭代次数')
    ax.set_ylabel('对数损失')
    ax.set_title('XGBoost 训练与验证损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/loss_curve_xgboost.png', dpi=150)
    plt.close()


def analyze_inference_speed(models_dict, X_test, n_runs=50):
    print("\n========== 推理速度分析 ==========")
    results = []
    for name, model in models_dict.items():
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            _ = model.predict(X_test)
            times.append(time.time() - t0)
        avg = np.mean(times)
        std = np.std(times)
        results.append({
            'Model': name,
            'Avg Time (s)': round(avg, 5),
            'Std Time (s)': round(std, 5),
            'Throughput (samples/s)': int(len(X_test) / avg),
        })
        print(f"  {name:25s}: {avg * 1000:.2f}ms +- {std * 1000:.2f}ms（{int(len(X_test) / avg)} 样本/秒）")

    pd.DataFrame(results).to_csv(f'{REPORTS_DIR}/inference_speed.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(results)), [r['Avg Time (s)'] * 1000 for r in results], color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r['Model'] for r in results], fontsize=10)
    ax.set_xlabel('推理时间 (ms)')
    ax.set_title(f'推理速度（{n_runs} 次平均）')
    for bar, r in zip(ax.patches, results):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{r["Avg Time (s)"] * 1000:.2f}ms', va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/inference_speed.png', dpi=150)
    plt.close()


def analyze_robustness(models_dict, X_test, y_test):
    print("\n========== 鲁棒性分析（噪声注入） ==========")
    noise_types = {
        '高斯 s=0.1': 0.1,
        '高斯 s=0.5': 0.5,
        '高斯 s=1.0': 1.0,
        '高斯 s=2.0': 2.0,
        '椒盐 5%': 0.05,
        '椒盐 10%': 0.10,
    }
    results = []
    for name, model in models_dict.items():
        base_acc = accuracy_score(y_test, model.predict(X_test))
        row = {'Model': name, 'Clean Acc': round(base_acc, 4)}
        for desc, param in noise_types.items():
            X_noisy = X_test.copy()
            if desc.startswith('椒盐'):
                n = int(param * X_noisy.size)
                idx = np.random.choice(X_noisy.size, n, replace=False)
                flat = X_noisy.flatten()
                flat[idx] = np.random.choice([-3, 3], n) * np.std(X_test) * 0.5
                X_noisy = flat.reshape(X_noisy.shape)
            else:
                X_noisy += np.random.normal(0, param, X_noisy.shape)
            noisy_acc = accuracy_score(y_test, model.predict(X_noisy))
            row[f'Acc {desc}'] = round(noisy_acc, 4)
            row[f'Drop {desc}'] = round(base_acc - noisy_acc, 4)
        results.append(row)
        print(f"  {name:25s}: clean={base_acc:.4f} | s=0.1->{row['Acc 高斯 s=0.1']:.4f} "
              f"s=0.5->{row['Acc 高斯 s=0.5']:.4f} s=1.0->{row['Acc 高斯 s=1.0']:.4f}")

    ndf = pd.DataFrame(results)
    ndf.to_csv(f'{REPORTS_DIR}/robustness_analysis.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    noise_keys = ['Clean Acc', 'Acc 高斯 s=0.1', 'Acc 高斯 s=0.5',
                  'Acc 高斯 s=1.0', 'Acc 高斯 s=2.0']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    x = np.arange(len(results))
    w = 0.15
    for i, nk in enumerate(noise_keys):
        axes[0].bar(x + i * w, [r.get(nk, 0) for r in results], w, label=nk, color=colors[i])
    axes[0].set_xticks(x + w * 2)
    axes[0].set_xticklabels([r['Model'] for r in results], rotation=15, fontsize=9)
    axes[0].set_ylabel('准确率')
    axes[0].set_title('噪声下的准确率', fontsize=13)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.2, 1.0)

    for i, nk in enumerate(noise_keys[1:]):
        axes[1].bar(x + i * w, [r['Clean Acc'] - r.get(nk, 0) for r in results],
                    w, label=f'下降 ({nk})', color=colors[i + 1], alpha=0.7)
    axes[1].set_xticks(x + w * 2)
    axes[1].set_xticklabels([r['Model'] for r in results], rotation=15, fontsize=9)
    axes[1].set_ylabel('准确率下降')
    axes[1].set_title('噪声下的准确率退化', fontsize=13)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/robustness_analysis.png', dpi=150)
    plt.close()


def analyze_overfitting(results_df):
    print("\n========== 过拟合分析 ==========")
    print(f"  {'模型':25s} {'训练准确率':>10s} {'测试准确率':>10s} {'差距':>8s} {'结论':>12s}")
    print(f"  {'-'*65}")
    for _, row in results_df.iterrows():
        gap = row['Overfit Gap']
        verdict = ('拟合良好' if gap < 0.01 else
                   '轻微过拟合' if gap < 0.05 else
                   '过拟合' if gap < 0.10 else '严重过拟合')
        print(f"  {row['Model']:25s} {row['Train Acc']:>10.4f} {row['Test Acc']:>10.4f} "
              f"{gap:>8.4f} {verdict:>12s}")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    ax.bar(x - 0.15, results_df['Train Acc'].values, 0.3, label='训练准确率', color='steelblue')
    ax.bar(x + 0.15, results_df['Test Acc'].values, 0.3, label='测试准确率', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'].values, rotation=15, fontsize=10)
    ax.set_ylabel('准确率')
    ax.set_title('过拟合：训练集 vs 测试集准确率')
    ax.legend()
    ax.set_ylim(0.7, 1.02)
    for i in range(len(results_df)):
        gap = results_df['Overfit Gap'].iloc[i]
        mid = (results_df['Train Acc'].iloc[i] + results_df['Test Acc'].iloc[i]) / 2
        ax.annotate(f'差距={gap:.3f}', xy=(i, mid), ha='center', fontsize=8, color='darkred')
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/overfitting_analysis.png', dpi=150)
    plt.close()
