import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(y_test, y_pred):
    """
    计算各种指标，包括准确率、调整后的准确率、宏平均召回率、加权召回率等。

    参数:
    y_test: 测试集的真实标签
    y_pred: 测试集的预测标签

    返回:
    包含各种指标的字典
    """
    # 初始化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred) * 100

    # 计算调整后的准确率
    adj_accuracy = (np.sum(np.diag(cm)) + cm[1, 2] + cm[2, 1] + cm[2, 3] +
                    cm[3, 2] + cm[3, 4] + cm[4, 3] + cm[4, 5] +
                    cm[5, 4] + cm[5, 6] + cm[6, 5]) / len(y_test) * 100

    # 封装计算召回率的函数
    def recall(class_idx):
        return cm[class_idx, class_idx] / np.sum(cm[class_idx, :]) * 100 if np.sum(cm[class_idx, :]) != 0 else 0

    # 计算各个类别的召回率
    recalls = [recall(i) for i in range(cm.shape[0])]

    # 计算宏平均召回率
    macro_orecall = np.mean(recalls)

    # 计算加权召回率
    weighted_orecall = np.sum([recalls[i] * np.sum(cm[i, :]) for i in range(cm.shape[0])]) / len(y_test)

    # 打印结果
    print(f"Accuracy: {accuracy:.3f}%")
    print(f"Adjusted Accuracy: {adj_accuracy:.3f}%")
    print("Confusion Matrix:")
    print(cm)
    print(f"Macro Recall: {macro_orecall:.3f}%")
    print(f"Weighted Recall: {weighted_orecall:.3f}%")

    # 返回各个指标
    return {
        "accuracy": accuracy,
        "adj_accuracy": adj_accuracy,
        "macro_recall": macro_orecall,
        "weighted_recall": weighted_orecall,
        "recalls": recalls
    }



