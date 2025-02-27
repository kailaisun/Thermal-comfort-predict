import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=12, num_classes=7, num_heads=4, num_layers=1, hidden_dim=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # x = x.mean(dim=1)
        x = self.classifier(x)
        x= self.softmax(x)
        return x

    def fit(self, X, y, epochs=20, batch_size=4, learning_rate=0.0003):
        """
        训练模型

        参数:
        X -- 训练数据特征 (numpy数组或pandas DataFrame)
        y -- 训练数据标签 (numpy数组或pandas Series)
        epochs -- 训练轮数 (默认10)
        batch_size -- 每个batch的样本数 (默认32)
        learning_rate -- 学习率 (默认0.001)
        """
        # 将数据转换为PyTorch张量
        tensor_x = torch.tensor(X, dtype=torch.float32).cuda()
        tensor_y = torch.tensor(y, dtype=torch.long).cuda()
        # tensor_y = F.one_hot(tensor_y, num_classes=self.num_classes)


        # 创建DataLoader
        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # 训练模型
        self.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        print("Training finished!")

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
            tensor_x = torch.tensor(X, dtype=torch.float32).cuda()
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
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(tensor_x)
            return logits


    def clone(self):
        return TransformerClassifier()

