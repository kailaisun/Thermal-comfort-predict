import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from Transformer_class import TransformerClassifier

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from Transformer_class import TransformerClassifier

# from dnn_class import DNNClassifier # 如果需要可以取消注释

# --- 1. 数据加载与预处理 ---
file_path = 'data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 提取 Group ID (假设第一列是人员ID 0-15)
id_col_name = data.columns[0]
groups = data[id_col_name]

# 选择特征和标签
label_column = 'TSV-7 scale'
# 排除 ID 列和 Label 列
excluded_columns = ['TSV-7 scale', id_col_name]
feature_columns = [col for col in data.columns if col not in excluded_columns]

X = data[feature_columns]
y = data[label_column]
y = y - min(y)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 转换为 Numpy 并进行 Scaling
X = X.to_numpy()
y = y.to_numpy()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# --- 2. 定义 OrdinalClassifier (保持原有逻辑，稍作适配) ---
class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learner):
        self.learner = learner
        self.ordered_learners = dict()
        self.classes = []

    def fit(self, X, y):
        self.classes = np.sort(np.unique(y))
        assert self.classes.shape[
                   0] >= 3, f'OrdinalClassifier needs at least 3 classes, only {self.classes.shape[0]} found'

        for i in range(self.classes.shape[0] - 1):
            N_i = np.vectorize(int)(y > self.classes[i])

            # Clone learner
            learner = self.learner.clone().cuda()

            # smote = SMOTE(random_state=42)
            # X_resampled, N_i_resampled = smote.fit_resample(X, N_i)

            # 将 numpy 转为 tensor target，因为你的 fit 内部似乎需要 tensor
            N_i = torch.tensor(N_i, dtype=torch.long).cuda()

            # 训练子模型
            learner.fit(X, N_i)
            self.ordered_learners[i] = learner

    def predict(self, X):
        return np.vectorize(lambda i: self.classes[i])(np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self, X):

        predicted = [self.ordered_learners[k].predict_proba(X)[:, 1].reshape(-1, 1) for k in self.ordered_learners]

        predicted = [p if isinstance(p, np.ndarray) else p.cpu().detach().numpy() for p in predicted]

        N_1 = 1 - predicted[0]
        N_K = predicted[-1]
        N_i = [predicted[i] - predicted[i + 1] for i in range(len(predicted) - 1)]

        probs = np.hstack([N_1, *N_i, N_K])

        # probs=probs*[0.5,0.63, 1, 1.63, 1, 0.63, 0.5]

        # probs = probs * [1.63, 1, 0.63, 0.5, 0.63, 1, 1.63]
        return probs


# Group K-Fold

gkf = GroupKFold(n_splits=10)  #n_splits=5

results_baseline = []
results_ordinal = []

print(f"Start Group K-Fold (Total Samples: {len(X)})...")

fold = 1
for train_idx, test_idx in gkf.split(X, y, groups):
    print(f"\n--- Fold {fold} ---")


    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()


    # Baseline
    clf = TransformerClassifier().cuda()
    clf.fit(X_train_resampled, y_train_resampled)  # Fit numpy (assuming class handles it)

    # Predict
    y_pred_base = clf.predict(X_test_tensor)
    # Convert result back to cpu/numpy
    if isinstance(y_pred_base, torch.Tensor):
        y_pred_base = y_pred_base.cpu().detach().numpy()

    acc_base = accuracy_score(y_test, y_pred_base)
    results_baseline.append(acc_base)
    print(f'Baseline Acc: {acc_base:.4f}')


    # Ordinal

    model = OrdinalClassifier(TransformerClassifier().cuda())
    model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred_ord = model.predict(X_test_tensor)
    # Convert result back
    if isinstance(y_pred_ord, torch.Tensor):
        y_pred_ord = y_pred_ord.cpu().detach().numpy()

    acc_ord = accuracy_score(y_test, y_pred_ord)
    results_ordinal.append(acc_ord)
    print(f'Ordinal Acc:  {acc_ord:.4f}')

    fold += 1

print(f"Average Baseline Accuracy: {np.mean(results_baseline):.4f}")
print(f"Average Ordinal Accuracy:  {np.mean(results_ordinal):.4f}")
