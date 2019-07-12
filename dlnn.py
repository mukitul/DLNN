# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:10:48 2018

@author: Md. Mukitul Islam Ratul
"""
import csv
import pandas as pd
import numpy as np

import keras as ks
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from itertools import cycle

def loadDataset(trainFile,testFile, trainingSet=[] , testSet=[]):
    with open(trainFile, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            trainingSet.append(dataset[x])

    with open(testFile, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            testSet.append(dataset[x])


def main():
    trainingSet=[]
    testSet=[]

    testFile='csv/CSV_for_our_method/our_method_test_tomato_hist_glcm.csv'
    trainFile='csv/CSV_for_our_method/our_method_train_tomato_hist_glcm.csv'
    
    loadDataset(trainFile,testFile,trainingSet, testSet)


    train = pd.DataFrame.from_records(trainingSet)
    test = pd.DataFrame.from_records(testSet)


    
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:,-1]
    
    # feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # encode class values as integers
    encoder1 = LabelEncoder()
    encoder1.fit(y_train)
    encoded1_Y = encoder1.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train = np_utils.to_categorical(encoded1_Y)
    encoder2 = LabelEncoder()
    encoder2.fit(y_test)
    encoded2_Y = encoder2.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_test = np_utils.to_categorical(encoded2_Y)
    

    # create model
    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(units = 128, activation = 'relu', input_dim = 260))
    # Adding the second hidden layer
    model.add(Dense(units = 64, activation = 'relu'))
    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(units = 4, activation = 'softmax'))
    
    
    
    # Compile model  categorical_crossentropy mean_squared_error binary_crossentropy
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    
    
    # Fitting our model 
    model.fit(x_train, y_train, epochs = 500)
    
    
    # save this model
    model.save('model/model1.model')
    
    # print summary of our model
    #model.summary()
    
    #plot Neural Network graph
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)      
            
    
    
    

    #laod saved model
    model = ks.models.load_model('model/model1.model')
    
    y_pred = model.predict(x_test)
    
    
    #-------------------------------------------------------------------------
    #----------------------draw GRAPH-----------------------------
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Proposed Segmentation Method'+'\n'+'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    fig = plt.gcf()
    fig.savefig('plot_img/graph_ap.png')
    
    
    
    #-------------------------------------------------------------------------
    #-----------------------draw GRAPH--------------------------------
    
    
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(4), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Proposed Segmentation Method'+'\n'+'Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
    
    plt.show()
    fig.savefig('plot_img/graph_multiClass.png')
    
    
    #-------------------------------------------------------------------------
    #------------------------------------------------------------------------
    
    
    
    
    
    
    y_pred = np.where(y_pred > 0.5,1,0)

    print("------------------------------------------------")
    print("Classification Report")
    print (metrics.classification_report(y_test, y_pred))
    
    print("------------------------------------------------")
    print("Confusion Matrix")
    print (metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    
    print("------------------------------------------------")
    print("Accuracy")
    print(metrics.accuracy_score(y_test, y_pred)*100)
    val_loss,val_acc = model.evaluate(x_test,y_test);
    print('val_loss: ',val_loss, 'val_acc: ',val_acc*100);
    
main()
