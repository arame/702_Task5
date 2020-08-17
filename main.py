from __future__ import print_function, division
from matplotlib import pyplot
from builtins import range
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import requests
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from settings import Settings
from neural_net import ANN2L


def main():

    Settings.start()
    with open((Settings.pathInputFile), "rb") as fh:
        data = pickle.load(fh)
    print("data keys; ", data.keys())

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(data['training_data'][0][0].reshape((100, 100)), cmap="gray")
    fig.savefig(Settings.pathOutput + "FirstImage.png", bbox_inches='tight', dpi=150)
    print("training dataset shape; ", data['training_data'][0].shape)

    X_tr = data['training_data'][0]
    y_tr = data['training_data'][1].ravel()
    X_v = data['validation_data'][0]
    y_v =data['validation_data'][1].ravel()
    X_t= data['test_data'][0]
    y_t = data['test_data'][1].ravel()

    X= np.append(X_tr, X_v, axis = 0)
    X= np.append(X, X_t, axis = 0)
    y = np.append(y_tr, y_v, axis = 0)
    y= np.append(y, y_t, axis = 0)

    #Splitting dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    label ={}
    for n, emotion in enumerate (Settings.emotions):
        label[n] = emotion
    print(label)

    # Visualisation color 
    _rows = 4
    _cols = 5
    _size = _rows * _cols
    fig, ax = plt.subplots(nrows=_rows, ncols=_cols, sharex=True, sharey=True,)
    ax = ax.flatten()
    _labels = []
    for i in range(_size):
        img = X_train[i].reshape(100,100)
        lbl = y_train[i]
        ax[i].imshow(img)
        _labels.append(lbl)
    fig.savefig(Settings.pathOutput + "EmotionImages.png", bbox_inches='tight', dpi=150)

    ground_truth_filepath = Settings.pathOutput + "ground_truth6.txt"
    ground_truth_file = open(ground_truth_filepath,"w+")
    ground_truth_text = 'GroundTruth: ', ' '.join('%5s' % Settings.emotions[_labels[true_label]] for true_label in range(_size))
    ground_truth_file.write(ground_truth_text[0] + ground_truth_text[1])
    print(ground_truth_text)
    ann=ANN2L()

    ann.run(X_train, y_train, X_val, y_val, X_test, y_test)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(Settings.epochs), ann.eval_['train_acc'], 
         label='training')
    plt.plot(range(Settings.epochs), ann.eval_['valid_acc'], 
            label='validation', linestyle='--')
    plt.plot(range(Settings.epochs), ann.eval_['test_acc'], 
            label='test', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    fig.savefig(Settings.pathOutput + "AccuracyEpoch.png", bbox_inches='tight', dpi=150)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(Settings.epochs), ann.eval_['cost'])
    plt.ylabel('Training Cost')
    plt.xlabel('Epochs')
    fig.savefig(Settings.pathOutput + "CostEpoch.png", bbox_inches='tight', dpi=150)

    y_test_pred = ann.predict(X_test)

    # Confusion Matrix

    cm, cm_acc = ann.conf_matrix(y_test, y_test_pred)
    print(cm)
    mpl.style.use('seaborn')
    fig= plt.figure()
    plt.clf()
    sns.heatmap(cm, annot=True,  fmt='d')   # font size
    plt.title('Confusion matrix with accuracy %.2f%%' % (cm_acc*100))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(Settings.pathOutput + "ConfusionMatrix.png", bbox_inches='tight', dpi=150)

    filepath = Settings.pathOutput + "digit_metrics.txt"
    file = open(filepath, "w+")
    Settings.outputLine(file, "* Digit metrics")
    Settings.outputLine(file, "* ---------------")
    total_sample =0
    for i in range(len(Settings.emotions)):
        row_sum = np.sum(cm[i:i+1,:])
        
        total_sample = total_sample + row_sum
        message = 'label: %.0f '' |  Precision: %.2f%%'' |  Recall: %.2f%%'' |  f1-score: %.2f%%'' |  support: %.0f' % (i, ann.precision(i, cm), ann.recall(i, cm), ann.f1_score(ann.precision(i, cm),ann.recall(i, cm)),row_sum)    
        Settings.outputLine(file, message)
    message = "Total samples:" + str(total_sample)
    Settings.outputLine(file, message)

    miscl_image = X_test[y_test != y_test_pred][:24] # missclassified image
    correct_label = y_test[y_test != y_test_pred][:24]  # Correct label
    miscl_label = y_test_pred[y_test != y_test_pred][:24] # missclassified label

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True,)
    ax = ax.flatten()

    print("======================================================")
    print("T: true label, P : predicted label")
    print("0: neutral, 1: anger, 2: contempt, 3: disgust")
    print("4: fear, 5: happy ,6: sadness, 7: surprise")
    print("======================================================")
    for i in range(16):
        img = miscl_image[i].reshape(100, 100)
        ax[i].imshow(img)
        ax[i].set_title('%d) T: %d P: %d' % (i+1, correct_label[i], miscl_label[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    print("final reminder:")
    Settings.printHyperparameters()
    print("The end")
####
if __name__ == '__main__':    
    main()