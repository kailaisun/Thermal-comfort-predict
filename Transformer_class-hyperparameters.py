import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=12, num_classes=5, num_heads=4, num_layers=2, hidden_dim=192):
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 保持之前的架构
        self.seq_len = 8
        self.token_dim = hidden_dim // self.seq_len

        if self.token_dim % num_heads != 0:
            raise ValueError(f"token_dim ({self.token_dim}) 必须能被 num_heads ({num_heads}) 整除。")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=self.token_dim * 4,
            dropout=0.1,  # 保持低 Dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.bn(x)
        x = self.relu(x)

        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.token_dim)
        x = self.transformer(x)
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)
        return x

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=64, learning_rate=0.001):
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(y, dtype=torch.long).to(self.device)

        if X_val is not None:
            tensor_x_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            tensor_y_val = torch.tensor(y_val, dtype=torch.long).to(self.device)
            val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            print(">>> 启用验证集监控 <<<")

        # --- 核心修改：回归温和权重 ---
        # 既然 Class 2 (样本最多) 掉分严重，我们就把它的权重从 0.6 提回 0.8
        # Class 0 从 2.0 降回 1.5，不再过度溺爱
        manual_weights = [1.5, 1.0, 0.8, 1.0, 1.2]
        class_weights = torch.tensor(manual_weights, dtype=torch.float32).to(self.device)
        print(f"应用温和权重: {manual_weights}")

        # 回归 CrossEntropy
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.state_dict())

        self.train()
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_acc = 0.0
            if val_loader is not None:
                self.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.forward(batch_x)
                        predicted = torch.argmax(outputs, dim=1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                val_acc = correct / total

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = copy.deepcopy(self.state_dict())

                scheduler.step(val_acc)
            else:
                scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")

        if val_loader is not None:
            print(f"Done! Loading Best Model with Val Acc: {best_val_acc:.4f}")
            self.load_state_dict(best_model_wts)

        return self

    def predict(self, X):
        """
        预测类别标签

        参数:
        X -- 需要预测的样本特征 (numpy数组或pandas DataFrame)

        返回:
        预测的类别标签 (张量)
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.forward(tensor_x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, X):
        """
        预测类别的概率

        参数:
        X -- 需要预测的样本特征 (numpy数组或pandas DataFrame)

        返回:
        每个类别的预测概率 (张量)
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.forward(tensor_x)
            return logits


    def clone(self):
        return TransformerClassifier()

