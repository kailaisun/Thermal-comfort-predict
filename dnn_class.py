import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DNNClassifier(nn.Module):
    def __init__(self, input_dim=12, num_classes=7, hidden_dims=[128, 64], dropout=0.1): #128,256, 256,
        super(DNNClassifier, self).__init__()
        self.num_classes = num_classes

        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

      
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def fit(self, X, y, epochs=30, batch_size=16, learning_rate=0.0003):
        
       
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)

        
        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)


        # class_weight_tensor = torch.tensor([0.00303,0.00312,0.0182,0.0093,0.0209,0.00950,0.06951], dtype=torch.float32)
        # criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)


        # loss = nn.CrossEntropyLoss()
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.empty(3, dtype=torch.long).random_(5)
        # output = loss(input, target)

        
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

        self.eval()  
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(tensor_x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, X):

        self.eval()  
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(tensor_x)
            return logits

    def clone(self):
        return DNNClassifier()


# 测试代码
if __name__ == "__main__":
    print(1)

