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
import os
from settings import Settings

class ANN2L(object):
    """ Feedforward neural network / Multilayer perceptron classifier.
    """
    def __init__(self):
        self.random = np.random.RandomState(Settings.seed)                       

    def _forward(self, X):
        """Compute forward propagation step"""

        # step net input to hidden layer1
     
        z_h1 = np.dot(X, self.w_h1) + self.b_h1
        a_h1 = self._relu(z_h1)
        #Optional dropout
        if Settings.dropout== True:
            a_h1 = self.in_dropout(a_h1, Settings.drop_prob1)

        # step hidden1 to hidden2
        z_h2 = np.dot(a_h1, self.w_h2) + self.b_h2 
        a_h2 = self._relu(z_h2)
         #Optional dropout
        if Settings.dropout== True:
            a_h2 = self.in_dropout(a_h2, Settings.drop_prob2)   

         # step hidden2 to output
        z_out = np.dot(a_h2, self.w_out) + self.b_out
        a_out = self._softmax(z_out)

        return z_h1, a_h1, z_h2, a_h2, z_out, a_out

    def run(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        """ Learn weights from training data."""
        n_output = np.unique(y_train).shape[0]  # number of class labels     
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################
         
        if Settings.weights_init == "trunc_norm":
            print(" Weights initialised within a truncated normal distribution")
            # weights for input -> hidden
            rad_1= 1/np.sqrt(n_features)
            self.distw1 = self.truncated_normal(mean=0, sd=1, low=-rad_1, upp= rad_1)
            self.b_h1 = self.distw1.rvs((1,Settings.hidden_n[0]))
            self.w_h1 = self.distw1.rvs((n_features, Settings.hidden_n[0]))

            # weights for hidden -> hidden 2

            rad_2= 1/np.sqrt(Settings.hidden_n[0])
            self.distw2 = self.truncated_normal(mean=0, sd=1, low=-rad_2, upp=rad_2)
            self.b_h2 = self.distw2.rvs((1,Settings.hidden_n[1]))
            self.w_h2 = self.distw2.rvs((Settings.hidden_n[0], Settings.hidden_n[1]))

            # weights for hidden2 -> output

            rad_3= 1/np.sqrt(Settings.hidden_n[1])
            self.distw3 = self.truncated_normal(mean=0, sd=1, low=-rad_2, upp=rad_2)
            self.b_out = self.distw3.rvs((1,n_output))
            self.w_out = self.distw3.rvs((Settings.hidden_n[1], n_output))

        if Settings.weights_init == "rand_norm":
           
            print("Weights initialised within a normal distribution - 'He'")
            
            #  Input ---> first hidden layer
            rad_1= np.sqrt(2/(n_features))        
            self.w_h1 = self.random.normal(loc=0.0, scale= rad_1,size=(n_features, Settings.hidden_n[0]))
            self.b_h1 = np.zeros(Settings.hidden_n[0])
        
            #  first hidden --->second hidden layer

            rad_2= np.sqrt(2/Settings.hidden_n[0])
            self.w_h2 = self.random.normal(loc=0.0, scale= rad_2, size=(Settings.hidden_n[0], Settings.hidden_n[1]))
            self.b_h2 = np.zeros(Settings.hidden_n[1])
            
            #  2nd hidden -> output

            rad_3= np.sqrt(2/Settings.hidden_n[1])
            self.w_out = self.random.normal(loc=0.0, scale= rad_3, size=(Settings.hidden_n[1], n_output))
            self.b_out = np.zeros(n_output)
              
        ################################################
        # Initial velocities - momentum & adam optimizers
        ################################################

        self.vb_h1 = np.zeros_like(self.b_h1)        
        self.vw_h1 = np.zeros_like(self.w_h1)
        self.vb_h2 = np.zeros_like(self.b_h2)
        self.vw_h2 = np.zeros_like(self.w_h2)
        self.vb_out = np.zeros_like(self.b_out)
        self.vw_out = np.zeros_like(self.w_out)       
        
        #Initializatin of sq for Adam

        self.sb_h1 = np.zeros_like(self.b_h1)   
        self.sw_h1 = np.zeros_like(self.w_h1)
        self.sb_h2 = np.zeros_like(self.b_h2)
        self.sw_h2 = np.zeros_like(self.w_h2)
        self.sb_out = np.zeros_like(self.b_out)
        self.sw_out = np.zeros_like(self.w_out)

        epoch_strlen = len(str(Settings.epochs))  # for progress formatting
    
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': [], 'test_acc':[]}

        y_train_enc = self._onehot(y_train, n_output) 

        # iterate over training epochs
 
        t= 0 # time -step for adam optimizer
       
        for i in range(Settings.epochs):
            if  i == Settings.schedule:
                for i in range(Settings.schedule, Settings.epochs, Settings.schedule):
                    Settings.lr = Settings.lr*Settings.lr_d
            
                    if Settings.lr <= 0.00001:
                        break
            
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if Settings.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - Settings.minibatch_size +
                                   1, Settings.minibatch_size):
                batch_idx = indices[start_idx:start_idx + Settings.minibatch_size]

                # forward propagation
                z_h1, a_h1, z_h2, a_h2, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation -SGD
                ##################

                # [n_samples, n_classlabels]
                error_out = (a_out - y_train_enc[batch_idx])/ a_out.shape[0]
                # error_out = (a_out - y_train_enc[batch_idx])

                # [n_samples, hidden_n]

                relu_derivative_h2= self._relu_d(a_h2)
                relu_derivative_h1= self._relu_d(a_h1)

                # [n_samples, n_classlabels] dot [n_classlabels, hidden-n]
                # -> [n_samples, hidden_n]
                error_h2 = (np.dot(error_out, self.w_out.T) *relu_derivative_h2)
                error_h1 = (np.dot(error_h2, self.w_h2.T) *relu_derivative_h1)

                # -> [hidden_n, n_classlabels]
                grad_w_out = np.dot(a_h2.T, error_out)
                grad_b_out = np.sum(error_out, axis=0)

                # [n_features, n_samples] dot [n_samples, hidden_n]
                # -> [n_features, hidden_n]
                grad_w_h2 = np.dot(a_h1.T, error_h2)
                grad_b_h2 = np.sum(error_h2, axis=0)

                # [n_features, n_samples] dot [n_samples, hidden_n]
                # -> [n_features, hidden_n]
                grad_w_h1 = np.dot(X_train[batch_idx].T, error_h1)
                grad_b_h1 = np.sum(error_h1, axis=0)

                # Regularization and weight updates- if not regularized  delta = grad)

                delta_w_out = (grad_w_out + Settings.l2*self.w_out*2)
                delta_b_out = grad_b_out  # bias is not regularized
               
                delta_w_h2 = (grad_w_h2 + Settings.l2*self.w_h2*2) #
                delta_b_h2 = grad_b_h2 # bias is not regularized

                delta_w_h1 = (grad_w_h1 + Settings.l2*self.w_h1*2)
                delta_b_h1 = grad_b_h1  # bias is not regularized
                
              
                #############
                # Update weights(parameters)
                #############
                t+=1  # Increase time-step before Adam
                if Settings.optimizer == "sgd_mo":
                    
                    self.vb_h1 = Settings.momentum*self.vb_h1 - Settings.lr * delta_b_h1
                    self.vw_h1 = Settings.momentum*self.vw_h1 - Settings.lr * delta_w_h1

                    self.vb_h2 = Settings.momentum*self.vb_h2 - Settings.lr * delta_b_h2
                    self.vw_h2 = Settings.momentum*self.vw_h2 - Settings.lr * delta_w_h2

                    self.vb_out = Settings.momentum*self.vb_out - Settings.lr * delta_b_out
                    self.vw_out = Settings.momentum*self.vw_out - Settings.lr * delta_w_out
                    
                    self.w_h1 += self.vw_h1
                    self.b_h1 += self.vb_h1
                    self.w_h2 += self.vw_h2
                    self.b_h2 += self.vb_h2
                    self.w_out += self.vw_out
                    self.b_out += self.vb_out

                if Settings.optimizer == "sgd":
                    
                    self.w_h1 -= Settings.lr * delta_w_h1
                    self.b_h1 -= Settings.lr * delta_b_h1

                    self.w_h2 -= Settings.lr * delta_w_h2
                    self.b_h2 -= Settings.lr * delta_b_h2
                      
                    self.w_out -= Settings.lr * delta_w_out
                    self.b_out -= Settings.lr * delta_b_out
                    
                if Settings.optimizer == "adam":
                    """b1: decay for previous gradient, 
                    b2:decay for the prevous squared gradient"""
                    
                    eps_sta = 1e-8
                    beta1 = 0.9
                    beta2 = 0.999

                    ##############
                    # average of the gradients and updates layer h1
                    #############
                    # weights
       
                    self.vw_h1 = beta1 * self.vw_h1  + (1 - beta1) * delta_w_h1
                    self.sw_h1 = beta2 * self.sw_h1 + (1.- beta2) * np.square(delta_w_h1)
                    self.vw_h1_c = self.vw_h1 / (1 - beta1** t)
                    self.sw_h1_c = self.sw_h1 / (1. - beta2 ** t)
 
                    #bias
                    self.vb_h1 = beta1 * self.vb_h1  + (1 - beta1) * delta_b_h1
                    self.sb_h1 = beta2 * self.sb_h1 + (1.- beta2) * np.square(delta_b_h1)
            
                    self.vb_h1_c = self.vb_h1 / (1 - beta1** t)
                    self.sb_h1_c = self.sb_h1 / (1. - beta2 ** t)
                    
                    # params update
       
                    prop_wh1 = Settings.lr * self.vw_h1_c / (np.sqrt(self.sw_h1_c) + eps_sta)
                    prop_bh1 = Settings.lr * self.vb_h1_c / (np.sqrt(self.sb_h1_c) + eps_sta)
   
                    self.w_h1-= prop_wh1
                    
                    self.b_h1-= prop_bh1
                    
                    #############
                    # layer h2
                    ###########
                    
                    #weigths
                    self.vw_h2 = beta1 * self.vw_h2  + (1 - beta1) * delta_w_h2
                    self.sw_h2 = beta2 * self.sw_h2 + (1.- beta2) * np.square(delta_w_h2)
                    self.vw_h2_c = self.vw_h2 / (1 - beta1** t)
                    self.sw_h2_c = self.sw_h2 / (1. - beta2 ** t) 

                    # bias
                    
                    self.vb_h2 = beta1 * self.vb_h2  + (1 - beta1) * delta_b_h2
                    self.sb_h2 = beta2 * self.sb_h2 + (1.- beta2) * np.square(delta_b_h2)
                    self.vb_h2_c = self.vb_h2 / (1 - beta1** t)
                    self.sb_h2_c = self.sb_h2 / (1. - beta2 ** t)
                    
 
                    # params update
                    prop_wh2 = Settings.lr * self.vw_h2_c / (np.sqrt(self.sw_h2_c) + eps_sta)
                    prop_bh2 = Settings.lr * self.vb_h2_c / (np.sqrt(self.sb_h2_c) + eps_sta)

                    self.w_h2-= prop_wh2
                    self.b_h2-= prop_bh2
        
        
                    #############
                    # output layer
                    ###########

                    
                    #weigths
                    self.vw_out = beta1 * self.vw_out  + (1 - beta1) * delta_w_out
                    self.sw_out = beta2 * self.sw_out + (1.- beta2) * np.square(delta_w_out)
                    
                    self.vw_out_c = self.vw_out / (1 - beta1** t)
                    self.sw_out_c = self.sw_out / (1. - beta2 ** t)
                    
                    
                    # layer bias
                    
                    self.vb_out = beta1 * self.vb_out  + (1 - beta1) * delta_b_out
                    self.sb_out = beta2 * self.sb_out + (1.- beta2) * np.square(delta_b_out)
                    
                    self.vb_out_c = self.vb_out/ (1 - beta1** t)
                    self.sb_out_c = self.sb_out / (1. - beta2 ** t)
                    
                     # params update
                    prop_w_out = Settings.lr * self.vw_out_c / (np.sqrt(self.sw_out_c) + eps_sta)
                    prop_b_out = Settings.lr * self.vb_out_c / (np.sqrt(self.sb_out_c) + eps_sta)

                    self.w_out-= prop_w_out
                    self.b_out-= prop_b_out


            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h1, a_h1, z_h2, a_h2, z_out, a_out = self._forward(X_train)
  
            cost = self._compute_cost(y_enc=y_train_enc, output = a_out)
        
            if np.isnan(cost) == True:
                break

            y_train_pred = self.predict(X_train)
            if Settings.dropout:
                y_valid_pred = self.predict_val_test(X_valid)
                y_test_pred = self.predict_val_test(X_test)
            else:
                y_valid_pred = self.predict(X_valid)
                y_test_pred = self.predict(X_test)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                            X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                            X_valid.shape[0])
            test_acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
            
            if test_acc <= 0.80* train_acc:
                
                print("Early stopping")
                return self

            print('%0*d/%d | Train cost: %.2f ''| Train/Valid Acc.: %.2f%%/%.2f%% '
                              '|Test Acc.: %.2f%%''| L rate.: %.5f '%
                            (epoch_strlen, i+1, Settings.epochs, cost, train_acc*100, valid_acc*100, test_acc*100, Settings.lr))

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            self.eval_['test_acc'].append(test_acc)
        
        return self

    def truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        """weights value distribution"""
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def conf_matrix(self, y, y_pred):
        """Computes the confusion matrix cm target vs prediction"""      
        dim= len(np.unique(y))
        cm = np.zeros((dim, dim),int) 
        for i in range (len(y_pred)):
            target = y[i]
            pred = y_pred[i]
            cm[target, pred] += 1
        cm=(np.array(cm))
        return cm  
            
    def precision(self, y, conf_matrix):
        col = conf_matrix[:, y]
        return conf_matrix[y, y] / col.sum()
      
    def recall(self, y, conf_matrix):
        row = conf_matrix[y, :]
        return conf_matrix[y, y] / row.sum()

    def f1_score(self, precision, recall):
        F1 = 2 * (precision * recall) / (precision + recall)
        return F1

    def _compute_cost(self,y_enc, output):
        """ If no regularisation the L2 term becomes zero"""
        L2_reg = Settings.l2*(np.sum(self.w_h1 ** 2.) +np.sum(self.w_h2 ** 2.) + 
                               np.sum(self.w_out ** 2.)) 
        ind = np.argmax(y_enc, axis = 1).astype(int)
        pred_probability = output[np.arange(len(output)), ind]
        log_preds = np.log(pred_probability)
        loss = -1.0 * np.sum(log_preds) / len(log_preds)
        loss+= L2_reg
        return loss

    def predict(self, X):
        """Predict class labels, X is the train, test or val dataset
        """
        z_h1, a_h1, z_h2, a_h2, z_out, a_out = self._forward(X)
        y_pred = np.argmax(a_out, axis=1)
        return y_pred

    def predict_val_test(self, X):
        """Predict class labels, X is the train, test or val dataset
        no dropout aplied to the test and validation dataset - probabilities 0 hard coded
        """
        dropout = Settings.dropout
        Settings.dropout = False
        z_h1, a_h1, z_h2, a_h2, z_out, a_out = self._forward(X)
        Settings.dropout = dropout
        y_pred = np.argmax(a_out, axis=1)
        return y_pred

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _relu(self, z):
        """Implements leakyrelu using a negative slope alpha""" 
        
        return np.maximum(z, 0.01*z)
        #  return np.maximum(z, np.zeros_like(z))
         
    def _relu_d(self, z):
        """Computes relu derivative with alpha to prevent nan"""  
        
        return np.where(z <= 0, z*0.01, 1)
  
    def _softmax(self,z):
        exp = np.exp(z )
        partition = np.sum(np.exp(z), axis=1, keepdims = True )
        
        return exp/partition

    def in_dropout(self, a, drop_prob):
        """Computes inverted dropout, 1-dropout
        keeps_prob is the maintained probability."""
        assert 0 <= drop_prob <= 1
        keep_prob = 1 -  drop_prob
        mask = np.random.uniform(0, 1.0, a.shape) < keep_prob
        
        if keep_prob > 0.0:
            scale = (1/keep_prob)
            return mask * a * scale
        #All elements kept
        if keep_prob == 1:
            return a
        #All elements dropped
        else:       
            return np.zeros_like(a)