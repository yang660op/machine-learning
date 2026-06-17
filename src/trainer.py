import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier




from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_SEED

sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, n_jobs=-1),
        'SVM (RBF)': SVC(kernel='rbf', random_state=RANDOM_SEED, probability=True),
        'ANN (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                    solver='adam', max_iter=500, random_state=RANDOM_SEED,
                                    early_stopping=True, validation_fraction=0.1),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1,
                                 random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
                                 eval_metric='mlogloss'),
    }


def train_and_evaluate_one(name, model, X_train, y_train, X_test, y_test, le):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) / len(y_test) * 1000

    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'{name} - 混淆矩阵', fontsize=12)
    ax.set_xlabel('预测值')
    ax.set_ylabel('真实值')
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/cm_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    return {
        'Model': name,
        'Train Acc': round(train_acc, 4),
        'Test Acc': round(test_acc, 4),
        'Overfit Gap': round(train_acc - test_acc, 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'CV Mean': round(cv_scores.mean(), 4),
        'CV Std': round(cv_scores.std(), 4),
        'Train Time (s)': round(train_time, 3),
        'Infer/1000(samples)': round(infer_time, 4),
    }, model


def train_all(X_train, y_train, X_test, y_test, le):
    print("\n========== 多算法训练 ==========")
    models = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"  [{name}] 训练中 ... ", end='', flush=True)
        res, fitted = train_and_evaluate_one(name, model, X_train, y_train, X_test, y_test, le)
        results.append(res)
        trained[name] = fitted
        print(f"完成 | 准确率={res['Test Acc']:.4f} | F1={res['F1-Score']:.4f} | "
              f"交叉验证={res['CV Mean']:.4f}+-{res['CV Std']:.4f} | "
              f"训练时间={res['Train Time (s)']:.2f}秒")

    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    key_cols = ['Model', 'Test Acc', 'Train Acc', 'Overfit Gap', 'F1-Score', 'CV Mean', 'Train Time (s)']
    print(f"\n  结果:\n{results_df[key_cols].to_string(index=False)}")
    results_df.to_csv(f'{REPORTS_DIR}/model_comparison.csv', index=False)

    _plot_comparison(results_df)

    return results_df, trained


def _plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results_df))
    width, metrics = 0.15, ['Test Acc', 'F1-Score', 'Precision', 'Recall', 'CV Mean']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results_df[m].values, width, label=m, color=colors[i])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df['Model'].values, rotation=15, fontsize=10)
    ax.set_ylabel('得分')
    ax.set_title('模型性能对比', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0.7, 1.0)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/model_comparison.png', dpi=150)
    plt.close()


def hyperparameter_tuning(X_train, y_train, X_test, y_test, le):
    print("\n========== 超参数调优（人工神经网络） ==========")
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
    }
    mlp = MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1,
                        random_state=RANDOM_SEED)
    grid = GridSearchCV(mlp, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  最佳参数: {grid.best_params_}")
    print(f"  最佳交叉验证 F1: {grid.best_score_:.4f}")

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print(f"  测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  测试集 F1:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title('调优 ANN - 混淆矩阵', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{REPORTS_DIR}/cm_tuned_ann.png', dpi=150)
    plt.close()

    joblib.dump(best, f'{MODELS_DIR}/best_model.pkl')
    with open(f'{MODELS_DIR}/best_params.json', 'w') as f:
        json.dump(grid.best_params_, f, indent=2)
    print(f"  模型已保存至 {MODELS_DIR}/")

    return best
