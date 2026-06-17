#!/usr/bin/env python
"""干豆 ML 流水线 - 命令行入口。

用法:
    python main.py full       完整流水线（EDA + 训练 + 评估）
    python main.py quick      快速运行（仅训练，无 EDA/损失曲线）
    python main.py train      仅训练（无 EDA，无评估）
    python main.py eval       仅评估（需要已训练的模型）
"""

import sys
import argparse
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR, FEATURE_COLS
from src.data_loader import load_data
from src.preprocessor import preprocess, prepare_features
from src.eda import analyze as eda_analysis
from src.trainer import train_all, hyperparameter_tuning
from src.evaluator import (
    plot_loss_curves,
    analyze_inference_speed,
    analyze_robustness,
    analyze_overfitting,
)


def cmd_full():
    from src.pipeline import run_full_pipeline
    run_full_pipeline()


def cmd_quick():
    from src.pipeline import run_quick
    run_quick()


def cmd_train():
    print("========== 训练模式 ==========")
    train, test, val = load_data()
    train, test, val = preprocess(train, test, val)
    X_tr_s, X_te_s, X_v_s, y_tr, y_te, y_v, le, scaler, _, _, _ = \
        prepare_features(train, test, val)
    results_df, _ = train_all(X_tr_s, y_tr, X_te_s, y_te, le)
    return results_df


def cmd_eval():
    print("========== 评估模式 ==========")
    import joblib

    train, test, val = load_data()
    train, test, val = preprocess(train, test, val)
    X_tr_s, X_te_s, X_v_s, y_tr, y_te, y_v, le, scaler, _, _, _ = \
        prepare_features(train, test, val)

    from src.trainer import get_models
    models = get_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_tr_s, y_tr)
        trained[name] = model

    plot_loss_curves(X_tr_s, y_tr, X_v_s, y_v)
    analyze_inference_speed(trained, X_te_s)
    analyze_robustness(trained, X_te_s, y_te)

    results_df = pd.read_csv(f'{REPORTS_DIR}/model_comparison.csv') if \
        any(f'{REPORTS_DIR}/model_comparison.csv') else None
    if results_df is not None:
        analyze_overfitting(results_df)


def main():
    parser = argparse.ArgumentParser(
        description='干豆 ML 流水线 - 命令行界面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py full           完整流水线 (EDA + 训练 + 评估)
  python main.py quick          快速训练（仅训练，无 EDA/损失曲线）
  python main.py train          仅训练（无 EDA，无评估）
  python main.py eval           仅评估（需要已训练的模型）
        """)
    parser.add_argument('mode', nargs='?', default='full',
                        choices=['full', 'quick', 'train', 'eval'],
                        help='流水线模式（默认: full）')
    parser.add_argument('-v', '--version', action='version', version='干豆 ML 流水线 v1.0')

    args = parser.parse_args()

    mode_map = {
        'full': cmd_full,
        'quick': cmd_quick,
        'train': cmd_train,
        'eval': cmd_eval,
    }
    mode_map[args.mode]()


if __name__ == '__main__':
    main()
