import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DNNClassifier(nn.Module):
    def __init__(self, input_dim=12, num_classes=7, hidden_dims=[64], dropout=0.1): #128,256, 256,
        super(DNNClassifier, self).__init__()
        self.num_classes = num_classes

        # 定义全连接层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def fit(self, X, y, epochs=30, batch_size=16, learning_rate=0.0003):
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
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)

        # 创建DataLoader
        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)


        # class_weight_tensor = torch.tensor([0.00303,0.00312,0.0182,0.0093,0.0209,0.00950,0.06951], dtype=torch.float32)
        # criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)


        # loss = nn.CrossEntropyLoss()
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.empty(3, dtype=torch.long).random_(5)
        # output = loss(input, target)

        # 训练模型
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        print("Training finished!")
        # return self.model

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
            tensor_x = torch.tensor(X, dtype=torch.float32)
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
        return DNNClassifier()


# 测试代码
if __name__ == "__main__":
    import numpy as np

    # 生成一些随机数据
    X = np.random.rand(100, 20)  # 100 个样本，20 个特征
    y = np.random.randint(0, 3, 100)  # 3 类分类

    # 初始化模型
    model = DNNClassifier(input_dim=20, num_classes=3, hidden_dims=[128, 64], dropout=0.1)

    # 训练模型
    model.fit(X, y, epochs=10, batch_size=8, learning_rate=0.001)

    # 测试模型
    test_X = np.random.rand(10, 20)  # 10 个测试样本
    predictions = model.predict(test_X)
    probabilities = model.predict_proba(test_X)

    print("Predicted classes:", predictions)
    print("Predicted probabilities:", probabilities)
