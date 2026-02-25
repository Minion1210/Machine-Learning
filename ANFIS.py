# Test myanfis model
import myanfis
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from tensorflow import keras



def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


if __name__ == "__main__":
    # set parameters
    param = myanfis.fis_parameters(
        n_input=6,                # no. of Regressors
        n_memb=2,                 # no. of fuzzy memberships
        batch_size=16,            # 16 / 32 / 64 / ...
        memb_func='sigmoid',      # 'gaussian' / 'gbellmf' / 'sigmoid'
        optimizer='adam',          # sgd / adam / ...
        # mse / mae / huber_loss / mean_absolute_percentage_error / ...
        #loss='huber_loss',
        loss=tf.keras.losses.Huber(),
        n_epochs=100               # 10 / 25 / 50 / 100 / ...
    )
    # create random data


    # Load the dataset
    X, y = load_data_from_csv(r"C:\Users\User\Desktop\ANFIS\Sleep_health_and_lifestyle_dataset - ANFIS.csv")
    job_to_number = {
    'Normal': 1,
    'Obese': 2,
    'Overweight': 3,
    'underweight': 4
        }

    for job, number in job_to_number.items():
        X[X == job] = number
    job_to_number = {
    'Accountant': 1,
    'Doctor': 2,
    'Engineer': 3,
    'Lawyer': 4,
    'Manager': 5,
    'Nurse': 6,
    'Sales Representative': 7,
    'Salesperson': 8,
    'Scientist': 9,
    'Software Engineer': 10,
    'Teacher': 11
    }
    for job, number in job_to_number.items():
        X[X == job] = number
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert labels to binary
    #y = (y == 'n').astype(int)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train=X_train[0:288,]
    X_test=X_test[0:64,]
    y_train=y_train[0:288,]
    y_test=y_test[0:64,]
    # Pad X_train
    #X_train = pad_to_multiple_of_32(X_train)
    
    
    fis = myanfis.ANFIS(n_input=param.n_input,
                        n_memb=param.n_memb,
                        #batch_size=param.batch_size,
                        memb_func=param.memb_func,
                        name='myanfis'
                        )

    # compile model
    fis.model.compile(optimizer=param.optimizer,
                      loss=param.loss,
                      metrics=['accuracy']  # Add accuracy as a metric
                      )

    # fit model
    history = fis.fit(X_train, y_train,
                      epochs=param.n_epochs,
                      batch_size=param.batch_size,
                      validation_data=(X_test, y_test),
                      # callbacks = [tensorboard_callback]  # for tensorboard
                      )

    # eval model
    import pandas as pd
    fis.plotmfs(show_initial_weights=True)

    loss_curves = pd.DataFrame(history.history)
    loss_curves.plot(figsize=(8, 5))

    fis.model.summary()

    # get premise parameters
    premise_parameters = fis.model.get_layer(
        'fuzzyLayer').get_weights()       # alternative

    # get consequence paramters
    bias = fis.bias
    weights = fis.weights

    # conseq_parameters = fis.model.get_layer('defuzzLayer').get_weights()# alternative
    #loss, accuracy = model.evaluate(X_test, y_test)
    #print(f'Test loss: {loss:.4f}')
    #print(f'Test accuracy: {accuracy:.4f}')

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()
    
    # Calculate ROC curve and AUC for test set
    y_test_pred_proba = fis.predict(X_test).ravel()
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve for test set
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set')
    plt.legend(loc='lower right')
    plt.show()

    # Calculate ROC curve and AUC for train set
    y_train_pred_proba = fis.predict(X_train).ravel()
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_proba)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Plot ROC curve for train set
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Train Set')
    plt.legend(loc='lower right')
    plt.show()
    # Plot confusion matrix for test set
    X_test_predict = (fis.predict(X_test)>0.5).astype(int)
    plt.figure(figsize=(8, 6))
    cm_test = confusion_matrix(y_test, X_test_predict )
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Plot confusion matrix for train set
    X_train_predict=(fis.predict(X_train) >0.5).astype(int)
    plt.figure(figsize=(8, 6))
    cm_train = confusion_matrix(y_train, X_train_predict)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Train Set')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, X_test_predict)
    test_precision = precision_score(y_test, X_test_predict)
    test_recall = recall_score(y_test, X_test_predict)
    test_f1_score = f1_score(y_test, X_test_predict)
    test_auc = roc_auc_score(y_test, X_test_predict )

    print("Test Metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-score: {test_f1_score:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    train_accuracy = accuracy_score(y_train, X_train_predict)
    train_precision = precision_score(y_train, X_train_predict)
    train_recall = recall_score(y_train, X_train_predict)
    train_f1_score = f1_score(y_train, X_train_predict)
    train_auc = roc_auc_score(y_train, X_train_predict )

    print("Train Metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  F1-score: {train_f1_score:.4f}")
    print(f"  AUC: {train_auc:.4f}")

# manually check ANFIS Layers step-by-step

    #L1 = myanfis.FuzzyLayer(n_input, n_memb)
    #L1(X) # to call build function
    #mus = fis.mus
    #sigmas = fis.sigmas
    #L1.set_weights([fis.mus, fis.sigmas])

    # op1 = np.array(L1(Xs))

    # L2 = myanfis.RuleLayer(n_input, n_memb)
    # op2 = np.array(L2(op1))

    # L3 = myanfis.NormLayer()
    # op3 = np.array(L3(op2))

    # L4 = myanfis.DefuzzLayer(n_input, n_memb)
    # L4(op3, Xs) # to call build function
    # bias = fis.bias
    # weights = fis.weights
    # L4.set_weights([fis.bias, fis.weights])
    # op4 = np.array(L4(op3, Xs))

    # L5 = myanfis.SummationLayer()
    # op5 = np.array(L5(op4))