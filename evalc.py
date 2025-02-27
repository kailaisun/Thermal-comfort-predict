import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(y_test, y_pred):
 
   
    cm = confusion_matrix(y_test, y_pred)


    accuracy = accuracy_score(y_test, y_pred) * 100


    adj_accuracy = (np.sum(np.diag(cm)) + cm[1, 2] + cm[2, 1] + cm[2, 3] +
                    cm[3, 2] + cm[3, 4] + cm[4, 3] + cm[4, 5] +
                    cm[5, 4] + cm[5, 6] + cm[6, 5]) / len(y_test) * 100


    def recall(class_idx):
        return cm[class_idx, class_idx] / np.sum(cm[class_idx, :]) * 100 if np.sum(cm[class_idx, :]) != 0 else 0


    recalls = [recall(i) for i in range(cm.shape[0])]


    macro_orecall = np.mean(recalls)


    weighted_orecall = np.sum([recalls[i] * np.sum(cm[i, :]) for i in range(cm.shape[0])]) / len(y_test)


    print(f"Accuracy: {accuracy:.3f}%")
    print(f"Adjusted Accuracy: {adj_accuracy:.3f}%")
    print("Confusion Matrix:")
    print(cm)
    print(f"Macro Recall: {macro_orecall:.3f}%")
    print(f"Weighted Recall: {weighted_orecall:.3f}%")

 
    return {
        "accuracy": accuracy,
        "adj_accuracy": adj_accuracy,
        "macro_recall": macro_orecall,
        "weighted_recall": weighted_orecall,
        "recalls": recalls
    }



