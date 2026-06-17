import joblib

from src.config import MODELS_DIR, REPORTS_DIR
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


def run_full_pipeline():
    print("=" * 60)
    print("  干豆 ML 流水线")
    print("=" * 60)

    train, test, val = load_data()
    eda_analysis(train, test, val)

    train, test, val = preprocess(train, test, val)

    X_tr_s, X_te_s, X_v_s, y_tr, y_te, y_v, le, scaler, X_tr_r, X_te_r, X_v_r = \
        prepare_features(train, test, val)

    results_df, trained_models = train_all(X_tr_s, y_tr, X_te_s, y_te, le)

    plot_loss_curves(X_tr_s, y_tr, X_v_s, y_v)

    best_model = hyperparameter_tuning(X_tr_s, y_tr, X_te_s, y_te, le)
    trained_models['调优 ANN'] = best_model

    analyze_inference_speed(trained_models, X_te_s)
    analyze_robustness(trained_models, X_te_s, y_te)
    analyze_overfitting(results_df)

    joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
    joblib.dump(le, f'{MODELS_DIR}/label_encoder.pkl')
    print(f"  模型文件已保存至 {MODELS_DIR}/")

    print("\n" + "=" * 60)
    print("  流水线完成！")
    print(f"  报告已保存至 {REPORTS_DIR}/")
    print("=" * 60)

    return results_df, trained_models


def run_quick():
    train, test, val = load_data()
    train, test, val = preprocess(train, test, val)
    X_tr_s, X_te_s, _, y_tr, y_te, _, le, scaler, _, _, _ = \
        prepare_features(train, test, val)
    results_df, trained_models = train_all(X_tr_s, y_tr, X_te_s, y_te, le)
    return results_df, trained_models
