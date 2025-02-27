
import pandas as pd
from evalc import evaluate_model
### StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Transformer_class import TransformerClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr



file_path='data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# # Define the PMV mapping
# pmv_mapping = {
#     'Cold': -3,
#     'Cool': -2,
#     'Slightly cool': -1,
#     'Neutral': 0,
#     'Slightly warm': 1,
#     'Warm': 2,
#     'Hot': 3
# }
#
# # Map the TSV-upper body column to PMV values
# data['TSV'] = data['TSV-upper body'].map(pmv_mapping)
# data.drop(columns=['TSV-upper body'], inplace=True)
# data['Air velocity'].fillna(data['Air velocity'].mean(), inplace=True)
## Step 1: 
spearman_corr = data.corr(method='spearman')


print("Spearman Correlation Matrix:")
print(spearman_corr)


plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Spearman Correlation Matrix')
plt.show()


pearson_corr = data.corr(method='pearson')


print("\nPearson Correlation Matrix:")
print(pearson_corr)


plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.show()

# Step 3:
kendall_corr = data.corr(method='kendall')


print("\nKendall Correlation Matrix:")
print(kendall_corr)


plt.figure(figsize=(8, 6))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Kendall Correlation Matrix')
plt.show()

# Step 4: 
correlation_methods = ['spearman', 'pearson', 'kendall']
correlation_results = {method: {} for method in correlation_methods}

for col in data.columns[:-1]:  #
    spearman_corr, spearman_p = spearmanr(data[col], data['TSV'])
    correlation_results['spearman'][col] = (spearman_corr, spearman_p)
    pearson_corr, pearson_p = pearsonr(data[col], data['TSV'])
    correlation_results['pearson'][col] = (pearson_corr, pearson_p)
    kendall_corr, kendall_p = kendalltau(data[col], data['TSV'])
    correlation_results['kendall'][col] = (kendall_corr, kendall_p)

# Step 5: 
correlation_dfs = {}
for method in correlation_methods:
    df = pd.DataFrame(
        correlation_results[method],
        index=['Correlation', 'p_value']
    ).T.sort_values(by='Correlation', ascending=False)
    correlation_dfs[method] = df


for method in correlation_methods:
    print(f"\n{method.capitalize()} Correlation with TSV:")
    print(correlation_dfs[method])

# Step 6: 
for method in correlation_methods:
    plt.figure(figsize=(8, 5))
    correlation_dfs[method]['Correlation'].plot(kind='bar', color='skyblue')
    plt.title(f'{method.capitalize()} Correlation of Features with TSV')
    plt.ylabel(f'{method.capitalize()} Correlation')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

exit(0)





# # Display the updated dataset to the user
# import ace_tools as tools; tools.display_dataframe_to_user(name="TSV-upper body mapped to PMV", dataframe=data)



# Select features and label
label_column = 'TSV-upper body (PMV)'
excluded_columns = ['TSV-upper body','TSV-upper body (PMV)']
feature_columns = [col for col in data.columns if col not in excluded_columns]




data['Air velocity'].fillna(data['Air velocity'].mean(), inplace=True)


