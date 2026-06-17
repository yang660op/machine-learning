import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

TRAIN_FILE = os.path.join(DATA_DIR, 'Dry_Bean_Dataset_Dirty_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'Dry_Bean_Dataset_Dirty_test.csv')
VAL_FILE = os.path.join(DATA_DIR, 'Dry_Bean_Dataset_Dirty_val.csv')

FEATURE_COLS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4',
]
TARGET = 'Class'

LABEL_MAP = {
    'BARBUNYA': 'BARBUNYA', 'barbunya': 'BARBUNYA', 'BARBUNYA ': 'BARBUNYA',
    'BOMBAY': 'BOMBAY', 'bombay': 'BOMBAY', 'B0MBAY': 'BOMBAY', 'BOMBAY ': 'BOMBAY',
    'CALI': 'CALI', 'cali': 'CALI', 'CALI ': 'CALI',
    'DERMASON': 'DERMASON', 'dermason': 'DERMASON', 'D3RMAS0N': 'DERMASON', 'DERMASON ': 'DERMASON',
    'HOROZ': 'HOROZ', 'horoz': 'HOROZ', 'H0R0Z': 'HOROZ', 'HOROZ ': 'HOROZ',
    'SEKER': 'SEKER', 'seker': 'SEKER', 'S3K3R': 'SEKER', 'SEKER ': 'SEKER',
    'SIRA': 'SIRA', 'sira': 'SIRA', 'SIRA ': 'SIRA',
}

CLASS_NAMES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

RANDOM_SEED = 42

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
