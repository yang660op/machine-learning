import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import FEATURE_COLS, TARGET, LABEL_MAP


def clean_labels(train, test, val):
    for df, name in [(train, 'train'), (test, 'test'), (val, 'val')]:
        before = df.shape[0]
        df[TARGET] = df[TARGET].map(LABEL_MAP)
        df.dropna(subset=[TARGET], inplace=True)
        dropped = before - df.shape[0]
        if dropped:
            print(f"  [{name}] 删除了 {dropped} 行未映射标签")
    return train, test, val


def fix_dtypes(train, test, val):
    for df, name in [(train, 'train'), (test, 'test'), (val, 'val')]:
        bad_s = pd.to_numeric(df['Solidity'], errors='coerce').isna().sum() - df['Solidity'].isna().sum()
        bad_c = pd.to_numeric(df['Compactness'], errors='coerce').isna().sum() - df['Compactness'].isna().sum()
        df['Solidity'] = pd.to_numeric(df['Solidity'], errors='coerce')
        df['Compactness'] = pd.to_numeric(df['Compactness'], errors='coerce')
        if bad_s > 0 or bad_c > 0:
            print(f"  [{name}] 转换了 {bad_s} 个 Solidity + {bad_c} 个 Compactness 错误条目为 NaN")
    return train, test, val


def impute_missing(train, test, val):
    perim_med = train['Perimeter'].median()
    solidity_med = train['Solidity'].median()
    compact_med = train['Compactness'].median()
    for df, name in [(train, 'train'), (test, 'test'), (val, 'val')]:
        before = df.isnull().sum().sum()
        df['Perimeter'] = df['Perimeter'].fillna(perim_med)
        df['Solidity'] = df['Solidity'].fillna(solidity_med)
        df['Compactness'] = df['Compactness'].fillna(compact_med)
        after = df.isnull().sum().sum()
        if before > 0:
            print(f"  [{name}] 缺失值: {before} -> {after}")
    return train, test, val


def prepare_features(train, test, val):
    X_train = train[FEATURE_COLS].values.astype(np.float64)
    y_train = train[TARGET].values
    X_test = test[FEATURE_COLS].values.astype(np.float64)
    y_test = test[TARGET].values
    X_val = val[FEATURE_COLS].values.astype(np.float64)
    y_val = val[TARGET].values

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_val_enc = le.transform(y_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    return (X_train_scaled, X_test_scaled, X_val_scaled,
            y_train_enc, y_test_enc, y_val_enc, le, scaler,
            X_train, X_test, X_val)


def preprocess(train, test, val):
    print("\n========== 数据预处理 ==========")
    train, test, val = clean_labels(train, test, val)
    train, test, val = fix_dtypes(train, test, val)
    train, test, val = impute_missing(train, test, val)
    print(f"  训练集: {train.shape}, 测试集: {test.shape}, 验证集: {val.shape}")
    return train, test, val
