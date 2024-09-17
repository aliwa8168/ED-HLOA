import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Get the list of available GPUs
gpus = tf.config.list_physical_devices('GPU')
# If a GPU is detected, set the first GPU to be visible to the current program and use it
if gpus:
    try:
        # Select the first GPU for training
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Set GPU memory to be allocated as needed, avoiding full memory allocation at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU: ", gpus[0].name)
    except RuntimeError as e:
        # If an error occurs, print the error message
        print("Error setting up GPU:", e)
else:
    print("No GPU found. Using CPU instead.")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import n_model as md
import datetime
from Read_picture import x_data, y_data, Read_time
from n_QIHLOA import QIHLOA

np.set_printoptions(threshold=np.inf)

def create_train_data():
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    # Perform necessary preprocessing on the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(tf.shape(x_test))
    print(tf.shape(y_test))
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # Convert label data to integers
    y_test = label_encoder.fit_transform(y_test)  # Convert label data to integers

    return x_train, y_train, x_test, y_test

# Model testing
def cnn_model_predict(cnn, model_name, it_acc, it_val_acc, it_loss, it_val_loss, test_labels, best_X):
    # Obtain various test result data
    print(type(test_labels))
    score = cnn.evaluate(test_data, test_labels, verbose=1)
    test_loss = 'Test Loss : {:.4f}'.format(score[0])
    test_accuracy = 'Test Accuracy : {:.4f}'.format(score[1])
    predicted_probabilities = cnn.predict(test_data)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    correct = np.nonzero(predicted_classes == test_labels)
    incorrect = np.nonzero(predicted_classes != test_labels)
    acc_score = accuracy_score(test_labels, predicted_classes)
    cls_report = classification_report(test_labels, predicted_classes, zero_division=1)
    print(test_loss)
    print(test_accuracy)
    # Save data to file
    current_time = datetime.datetime.now()
    time = current_time - Read_time
    sss = time.total_seconds()

    if not os.path.exists('result/' + version):
        os.mkdir('result/' + version)
    file = open("./result/" + version + '/record_' + model_name + '.txt', "a")
    file.write("Optimal hyperparameters: " + str(best_X) + '\n')
    file.write('test_loss: ' + test_loss)
    file.write('\ntest_accuracy: ' + test_accuracy)
    file.write('\ncorrect: ' + str(len(correct[0])))
    file.write('\nincorrect: ' + str(len(incorrect[0])))
    file.write('\nacc_score: ' + str(acc_score))
    file.write('\n' + cls_report)

    file.write('\n' + "it_acc=" + str(it_acc))
    file.write('\n' + "it_val_acc=" + str(it_val_acc))
    file.write('\n' + "it_loss=" + str(it_loss))
    file.write('\n' + "it_val_loss=" + str(it_val_loss))
    file.write('\n' + "Time spent: " + str(sss))
    file.close()
    # Save the model
    if not os.path.exists('models/' + version):
        os.mkdir('models/' + version)
    model_path = 'models/' + version + '/cnn_model_' + model_name + '.h5'
    cnn.save(model_path)
    print("Model training completed, saved at:", model_path)
    print("Total time spent:", sss)


if __name__ == '__main__':
    # Generate training and testing datasets
    train_data, train_label, test_data, test_label = create_train_data()
    # Model parameters
    model_param = {
        "test_data": test_data,
        "test_label": test_label,
        "data": train_data,
        "label": train_label
    }
    """
    Parameters for Quadratic Interpolation Horned Lizard Optimization Algorithm
    """
    qihloa_param = {
        "dim": 2,
        "SearchAgents_no": 10,
        "Max_iter": 10,
        #               lr,       compression_rate
        "lb": np.array([0.00001,   0.1 ]),  # lowbound
        "ub": np.array([0.1,       1 ])  # upbound
    }
    version = 'qihloa'
    import os

    folder_path = 'result/' + version

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    folder_path = 'models/' + version
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    # Optimize using quadratic interpolation + horned lizard optimization algorithm

    qihloa = QIHLOA(model_param, qihloa_param)
    best_err, best_X = qihloa.run()


    print("The optimal validation set loss after QIHLOA optimization is:", best_err)
    print("The optimal hyperparameters after QIHLOA optimization are:", best_X)
    model = md.DenseNet201(best_X)
    qihloa_model = model.model_create(best_X[0])
    qihloa_history = qihloa_model.fit(train_data, train_label, epochs=30, batch_size=16, validation_split=0.1)
    qihloa_accuracy = qihloa_history.history['accuracy']
    qihloa_val_accuracy = qihloa_history.history['val_accuracy']
    qihloa_loss = qihloa_history.history['loss']
    qihloa_val_loss = qihloa_history.history['val_loss']
    cnn_model_predict(qihloa_model, version,
                      qihloa_accuracy, qihloa_val_accuracy, qihloa_loss, qihloa_val_loss, test_label,best_X)





