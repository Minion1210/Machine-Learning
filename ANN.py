import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt





# 讀取 CSV 檔案
file_path = "C:/Users/ASUS/Desktop/introduction/Sleep_health_and_lifestyle_dataset.csv"
data = pd.read_csv(file_path)

#print(data)
#檢查 DataFrame 的列名
#print(data.columns)
# 刪除 'Person.ID' 欄位
data = data.drop(columns=['Person ID'])
# 將 'Gender' 列的值轉換
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)
# 將 'Sleep.Disorder' 列的值轉換
data['Sleep Disorder'] = data['Sleep Disorder'].apply(lambda x: 1 if x == 'Insomnia' else 0)
# 將 'Blood.Pressure' 列的值轉換
data['Blood Pressure'] = data['Blood Pressure'].apply(
    lambda x: 0 if 'normal blood pressure' in x else (1 if 'mild hypertension' in x else 2))
# 將 'BMI.Category' 列的值轉換
data['BMI Category'] = data['BMI Category'].apply(
    lambda x: 0 if x == 'Normal' else (1 if x == 'Underweight' else (2 if x == 'Obese' else 3)))
# 將 'Occupation' 列的值轉換
data['Occupation'] = data['Occupation'].apply(
    lambda x: 0 if x == 'Doctor' else (1 if x == 'Nurse' else 2))
# 將 'Daily Steps' 列的值轉換
data['Daily Steps'] = data['Daily Steps'].apply(lambda x: 0 if 7000 <= x <= 10000 else 1)

# 將數據分為有病和沒病兩個子集
has_disease = data[data['Sleep Disorder'] == 1]
no_disease = data[data['Sleep Disorder'] == 0]
# 從有病和沒病兩個子集中分別抽取80%作為訓練集
train_has_disease, test_has_disease = train_test_split(has_disease, test_size=0.2, random_state=42)
train_no_disease, test_no_disease = train_test_split(no_disease, test_size=0.2, random_state=42)
# 合併訓練集和測試集
train_data = pd.concat([train_has_disease, train_no_disease])
test_data = pd.concat([test_has_disease, test_no_disease])
# 將訓練集和測試集的特徵和標籤分開
X_train = train_data.drop(columns=['Sleep Disorder'])
y_train = train_data['Sleep Disorder']
X_test = test_data.drop(columns=['Sleep Disorder'])
y_test = test_data['Sleep Disorder']

# 定义神经网络模型 
model = Sequential([
    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
   #  Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 訓練模型並保存訓練歷史
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 在測試集上進行預測
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# 計算評估指標
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
# 在测试集上评估模型性能
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("Test Loss:", test_loss)

# 繪製損失圖表
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 绘制 ROC 曲线
plt.plot(fpr, tpr, label='ROC Curve (AUC = 0.85)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# 绘制准确率图
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()





#train_data.to_csv('train_data.csv', index=False)
#test_data.to_csv('test_data.csv', index=False)


#print(data)
#
print("Hello, world!")

