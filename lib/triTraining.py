# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:11:09 2022

@author: talha
"""

import sklearn
import numpy as np
import keras.backend as K  
import matplotlib.pyplot as plt
import time

class TriTraining():
    def __init__(self,model1,model2,model3,data1,ratio=4,epoch = 5,path_output1="model1_best.h5",path_output2="model2_best.h5",path_output3="model3_best.h5",activation="sigmoid"):
        """
        
        Parameters
        ----------
        model : Keras Model 
        data: DataSet 
        class_number : int, optional, default=2
        perEpoch_label : int, optional, default=200
        path_output : PATH of model (save), optional, default="model_best.h5"

        Returns
        -------
        None.

        """
        self.X_train1 = np.array(data1.train_x)
        self.X_val1 = np.array(data1.valid_x)
        self.X_test1 = np.array(data1.test_x)
        self.y_train1 = np.array(data1.train_y)
        self.y_val1 = np.array(data1.valid_y)
        self.y_test1 = np.array(data1.test_y)
        self.X_unlabel1 = np.array(data1.unlabel_x)
        self.y_unlabel1 = np.array(data1.unlabel_y)
        self.lenUnlabel_y = len(self.y_unlabel1)

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3        
        self.models = [model1,model2,model3]
        
        self.limit = int(len(self.X_unlabel1) / 4 * ratio)
        self.ratio = ratio
        self.epoch = epoch
        self.activation = activation
        self.path_output1 = path_output1
        self.path_output2 = path_output2
        self.path_output3 = path_output3
        self.history1 = []
        self.f1Score1 = []
        self.accuarcy1 = []
        self.history2 = []
        self.f1Score2 = []
        self.accuarcy2 = []
        self.history3 = []
        self.f1Score3 = []
        self.accuarcy3 = []
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.endTime = 0
        self.startTime = 0
        self.Time = 0
        self.trueCount = 0
        self.falseCount = 0
        self.unlabel_accuarcy = 0
        
        self.epochCount = 0
        
        self.class_0_true = []
        self.class_1_true = []
        self.class_0_false = []
        self.class_1_false = []
        
        for i in range(self.epoch):
            self.class_0_true.append(0)
            self.class_1_true.append(0)
            self.class_0_false.append(0)
            self.class_1_false.append(0)
        
        
    def moveClass(self,prediction):
        result = []
        for i in prediction:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return np.asarray(result)
                                   
    def f1_score(self,precision, recall):
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val    
    
    def run(self):
        try:
            len_unlabel = len(self.X_unlabel1)
            len_train1 = len(self.X_train1)
            len_val1 = len(self.X_val1)
            len_test1 = len(self.X_test1)
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train1))
            print("Validation Dataset1 size: " + str(len_val1))
            print("Test Dataset1 size: " + str(len_test1))
        except Exception as e:
            print("ERROR3: " + str(e))
            return
        self.epochCount = 0
        changeF1 = 0
        maxF1_1 = 0
        maxF1_2 = 0
        maxF1_3 = 0
        self.startTime = time.time()
        
        print("\nTraining Bootstrap Sample")
        for i in range(3):
            sample = sklearn.utils.resample(self.X_train1, self.y_train1)  
            self.models[i].fit(*sample,validation_data=(self.X_val1, self.y_val1))  
        print("Finished Bootstrap\n")

        #status = True
        while  self.epochCount < self.epoch:
            #try:
            for i in range(3):
                L_x = []
                L_y = []
                j, k = np.delete(np.array([0,1,2]),i)
                prediction_j = self.models[j].predict(self.X_unlabel1)
                prediction_k = self.models[k].predict(self.X_unlabel1)
                if self.activation == "sigmoid":
                    index = 0
                    while index < self.limit:
                        if prediction_j[index] >= 0.5 and prediction_k[index] >= 0.5:
                            L_x.append(self.X_unlabel1[index])
                            L_y.append(1)
                            if not self.lenUnlabel_y < 2:
                                if self.y_unlabel1[index] == 1:
                                    self.trueCount += 1
                                    self.class_1_true[self.epochCount] += 1        
                                else:
                                    self.falseCount += 1
                                    self.class_1_false[self.epochCount] += 1   
                        elif prediction_j[index] <= 0.5 and prediction_k[index] <= 0.5:
                            L_x.append(self.X_unlabel1[index])
                            L_y.append(0)
                            if not self.lenUnlabel_y < 2:
                                if self.y_unlabel1[index] == 0:
                                    self.trueCount += 1
                                    self.class_0_true[self.epochCount] += 1
                                else:
                                    self.class_0_false[self.epochCount] += 1
                                    self.falseCount += 1
                        index += 1
                elif self.activation == "softmax":
                    index = 0
                    for index_x in self.X_unlabel1:
                        temp = prediction_j[index]
                        temp[np.argmax(prediction_j[index])] = 1
                        for t in range(len(temp)):
                            if not temp[t] == 1:
                                temp[t] = 0
                        class_number_j = temp
                        temp = prediction_k[index]
                        temp[np.argmax(prediction_k[index])] = 1
                        for t in range(len(temp)):
                            if not temp[t] == 1:
                                temp[t] = 0
                        class_number_k = temp
                        if np.argmax(class_number_j) == np.argmax(class_number_k):
                            L_x.append(index_x)
                            L_y.append(class_number_j)
                            if not self.lenUnlabel_y < 2:
                                if np.argmax(self.y_unlabel1[index]) == np.argmax(class_number_j):
                                    self.trueCount += 1
                                else:
                                    self.falseCount += 1
                        index += 1
                        if index > self.limit:
                            break
                L_x = np.array(L_x)
                L_y = np.array(L_y)
                if not len(L_x) < 2:
                    train_x = np.concatenate([self.X_train1, L_x])
                    train_y = np.concatenate([self.y_train1, L_y])
                else:
                    train_x = self.X_train1
                    train_y = self.y_train1
                self.history1.append(self.models[i].fit(train_x, train_y, epochs=1, 
                          validation_data=(self.X_val1, self.y_val1)))
            #except Exception as e:
            #    print("ERROR4: " + str(e))
            #    return
            try:
                status = False
                loss1, accuracy1, precision1, recall1 = self.models[0].evaluate(self.X_test1, self.y_test1, verbose=0)
                loss2, accuracy2, precision2, recall2 = self.models[1].evaluate(self.X_test1, self.y_test1, verbose=0)
                loss3, accuracy3, precision3, recall3 = self.models[2].evaluate(self.X_test1, self.y_test1, verbose=0)
                self.f1Score1.append(self.f1_score(precision1, recall1))
                self.f1Score2.append(self.f1_score(precision2, recall2))
                self.f1Score3.append(self.f1_score(precision3, recall3))
                self.accuarcy1.append(accuracy1)
                self.accuarcy2.append(accuracy2)
                self.accuarcy3.append(accuracy3)
                self.loss1.append(loss1)
                self.loss2.append(loss2)
                self.loss3.append(loss3)
                if maxF1_1 < max(self.f1Score1):
                    maxF1_1 = max(self.f1Score1)
                    changeF1 = 0
                    self.saveMaxModel(self.model1,self.path_output1)
                else:
                    changeF1 += 1
                if maxF1_2 < max(self.f1Score2):
                    maxF1_2 = max(self.f1Score2)
                    changeF1 = 0
                    self.saveMaxModel(self.model2,self.path_output2)
                else:
                    changeF1 += 1
                if maxF1_3 < max(self.f1Score3):
                    maxF1_3 = max(self.f1Score3)
                    changeF1 = 0
                    self.saveMaxModel(self.model3,self.path_output3)
                else:
                    changeF1 += 1
                self.epochCount += 1
                print("EPOCH " + str(self.epochCount) + "\nMODEL 1\nF1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
                print("MODEL 2\nF1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
                print("MODEL 3\nF1 : " + str(self.f1Score3[len(self.f1Score3)-1]) + "\nMax F1 : " + str(maxF1_3))
            except Exception as e:
                print("ERROR5: " + str(e))
                return
        try:
            print("\nRESULT\nMODEL1\n2. Epoch F1 : " + str(self.f1Score1[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
            print("MODEL 2\n2. Epoch F1 : " + str(self.f1Score2[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
            print("MODEL 3\n2. Epoch F1 : " + str(self.f1Score3[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score3[len(self.f1Score3)-1]) + "\nMax F1 : " + str(maxF1_3))
            if not self.lenUnlabel_y < 2:
                self.unlabel_accuarcy = self.trueCount / (self.trueCount + self.falseCount)
                print("\nUnlabel accuarcy : " + str(self.unlabel_accuarcy))
            self.endTime = time.time()
            self.Time = self.endTime - self.startTime
            print("Training Time: " + str(self.Time) + " sc")
            self.showF1(self.f1Score1,"Model 1")
            self.showF1(self.f1Score2,"Model 2")
            self.showF1(self.f1Score3,"Model 3")
            self.showUnlabel()
            
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train1))
            print("Validation Dataset1 size: " + str(len_val1))
            print("Test Dataset1 size: " + str(len_test1))
        except Exception as e:
            print("ERROR6: " +str(e))

    def saveMaxModel(self,model,path_output):
        try:
            model.save(path_output)
        except Exception as e:
            print("ERROR7: " + str(e))
            
    def showF1(self,f1Score,title):
        plt.figure(figsize=(20,20))
        score = [l for l in f1Score]
        plt.plot(score)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')    

    def showUnlabel(self):
        barWidth = 0.15
        fig = plt.subplots(figsize =(16, 8))
         
        # set height of bar
         
        # Set position of bar on X axis
        br1 = np.arange(len(self.class_0_false))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
         
        # Make the plot
        plt.bar(br1, self.class_0_false, color ='r', width = barWidth,
                edgecolor ='grey', label ='False 0')
        plt.bar(br2, self.class_0_true, color ='lime', width = barWidth,
                edgecolor ='grey', label ='True 0')
        plt.bar(br3, self.class_1_false, color ='darkred', width = barWidth,
                edgecolor ='grey', label ='False 1')
        plt.bar(br4, self.class_1_true, color ='darkgreen', width = barWidth,
                edgecolor ='grey', label ='True 1')
        label = []
        for i in range(self.epochCount):
            label.append(i+1)
        
        # Adding Xticks
        plt.xlabel('Epoch', fontweight ='bold', fontsize = 15)
        plt.ylabel('Number', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(self.class_0_false))], label)
        plt.title("Tri-training")
         
        plt.legend()
        plt.show()
        
        
        
class TriTrainingwDisagreement():
    def __init__(self,model1,model2,model3,data1,ratio=4,epoch=5,path_output1="model1_best.h5",path_output2="model2_best.h5",path_output3="model3_best.h5",activation="sigmoid"):
        """
        
        Parameters
        ----------
        model : Keras Model 
        data: DataSet 
        class_number : int, optional, default=2
        perEpoch_label : int, optional, default=200
        path_output : PATH of model (save), optional, default="model_best.h5"

        Returns
        -------
        None.

        """
        self.X_train1 = np.array(data1.train_x)
        self.X_val1 = np.array(data1.valid_x)
        self.X_test1 = np.array(data1.test_x)
        self.y_train1 = np.array(data1.train_y)
        self.y_val1 = np.array(data1.valid_y)
        self.y_test1 = np.array(data1.test_y)
        self.X_unlabel1 = np.array(data1.unlabel_x)
        self.y_unlabel1 = np.array(data1.unlabel_y)
        self.lenUnlabel_y = len(self.y_unlabel1)

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3        
        self.models = [model1,model2,model3]
        
        self.limit = len(self.X_unlabel1) / 4 * ratio
        self.ratio = ratio
        self.epoch = epoch
        self.activation = activation
        self.path_output1 = path_output1
        self.path_output2 = path_output2
        self.path_output3 = path_output3
        self.history1 = []
        self.f1Score1 = []
        self.accuarcy1 = []
        self.history2 = []
        self.f1Score2 = []
        self.accuarcy2 = []
        self.history3 = []
        self.f1Score3 = []
        self.accuarcy3 = []
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.pseudoLabel_number = 0
        self.trueCount = 0
        self.falseCount = 0
        self.endTime = 0
        self.startTime = 0
        self.Time = 0
        self.unlabel_accuarcy = 0
        
        self.epochCount = 0
        
        self.class_0_true = []
        self.class_1_true = []
        self.class_0_false = []
        self.class_1_false = []
        
        for i in range(self.epoch):
            self.class_0_true.append(0)
            self.class_1_true.append(0)
            self.class_0_false.append(0)
            self.class_1_false.append(0)
        
        
    def moveClass(self,prediction):
        result = []
        for i in prediction:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return np.asarray(result)
                                   
    def f1_score(self,precision, recall):
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val    
    
    def run(self):
        try:
            len_unlabel = len(self.X_unlabel1)
            len_train1 = len(self.X_train1)
            len_val1 = len(self.X_val1)
            len_test1 = len(self.X_test1)
        except Exception as e:
            print("ERROR3: " + str(e))
            return
        self.epochCount = 0
        changeF1 = 0
        maxF1_1 = 0
        maxF1_2 = 0
        maxF1_3 = 0
        status = True
        self.startTime = time.time()
        
        print("\nTraining Bootstrap Sample")
        for i in range(3):
            sample = sklearn.utils.resample(self.X_train1, self.y_train1)  
            self.models[i].fit(*sample,validation_data=(self.X_val1, self.y_val1))  
        print("Finished Bootstrap\n")
        
        #status = True
        while  self.epochCount < self.epoch:
            #try:
            for i in range(3):
                L_x = []
                L_y = []
                j, k = np.delete(np.array([0,1,2]),i)
                prediction_i = self.models[i].predict(self.X_unlabel1)
                prediction_j = self.models[j].predict(self.X_unlabel1)
                prediction_k = self.models[k].predict(self.X_unlabel1)
                if self.activation == "sigmoid":
                    index = 0
                    while index < self.limit:
                        if prediction_j[index] >= 0.5 and prediction_k[index] >= 0.5 and prediction_i[index] <= 0.5:
                            L_x.append(self.X_unlabel1[index])
                            L_y.append(1)
                            if not self.lenUnlabel_y < 2:
                                if self.y_unlabel1[index] == 1:
                                    self.trueCount += 1
                                    self.class_1_true[self.epochCount] += 1        
                                else:
                                    self.falseCount += 1
                                    self.class_1_false[self.epochCount] += 1   
                        elif prediction_j[index] <= 0.5 and prediction_k[index] <= 0.5 and prediction_i[index] >= 0.5:
                            L_x.append(self.X_unlabel1[index])
                            L_y.append(0)
                            if not self.lenUnlabel_y < 2:
                                if self.y_unlabel1[index] == 0:
                                    self.trueCount += 1
                                    self.class_0_true[self.epochCount] += 1
                                else:
                                    self.class_0_false[self.epochCount] += 1
                                    self.falseCount += 1
                        index += 1
                elif self.activation == "softmax":
                    index = 0
                    for index_x in self.X_unlabel1:
                        temp = prediction_i[index]
                        temp[np.argmax(prediction_i[index])] = 1
                        for t in range(len(temp)):
                            if not temp[t] == 1:
                                temp[t] = 0
                        class_number_i = temp
                        temp = prediction_j[index]
                        temp[np.argmax(prediction_j[index])] = 1
                        for t in range(len(temp)):
                            if not temp[t] == 1:
                                temp[t] = 0
                        class_number_j = temp
                        temp = prediction_k[index]
                        temp[np.argmax(prediction_k[index])] = 1
                        for t in range(len(temp)):
                            if not temp[t] == 1:
                                temp[t] = 0
                        class_number_k = temp
                        if np.argmax(class_number_j) == np.argmax(class_number_k) and not (np.argmax(class_number_j) == np.argmax(class_number_i)):
                            L_x.append(index_x)
                            L_y.append(class_number_j)
                            if not self.lenUnlabel_y < 2:
                                print(self.y_unlabel1[index])
                                print(np.argmax(class_number_j))
                                if np.argmax(self.y_unlabel1[index]) == np.argmax(class_number_j):
                                    self.trueCount += 1
                                else:
                                    self.falseCount += 1
                        index += 1
                        if index > self.limit:
                            break
                L_x = np.array(L_x)
                L_y = np.array(L_y)
                if not len(L_x) < 2:
                    train_x = np.concatenate([self.X_train1, L_x])
                    train_y = np.concatenate([self.y_train1, L_y])
                else:
                    train_x = self.X_train1
                    train_y = self.y_train1
                self.history1.append(self.models[i].fit(train_x, train_y, epochs=1, 
                          validation_data=(self.X_val1, self.y_val1)))
            #except Exception as e:
            #    print("ERROR4: " + str(e))
            #    return
            try:
                status = False
                loss1, accuracy1, precision1, recall1 = self.models[0].evaluate(self.X_test1, self.y_test1, verbose=0)
                loss2, accuracy2, precision2, recall2 = self.models[1].evaluate(self.X_test1, self.y_test1, verbose=0)
                loss3, accuracy3, precision3, recall3 = self.models[2].evaluate(self.X_test1, self.y_test1, verbose=0)
                self.f1Score1.append(self.f1_score(precision1, recall1))
                self.f1Score2.append(self.f1_score(precision2, recall2))
                self.f1Score3.append(self.f1_score(precision3, recall3))
                self.accuarcy1.append(accuracy1)
                self.accuarcy2.append(accuracy2)
                self.accuarcy3.append(accuracy3)
                self.loss1.append(loss1)
                self.loss2.append(loss2)
                self.loss3.append(loss3)
                if maxF1_1 < max(self.f1Score1):
                    maxF1_1 = max(self.f1Score1)
                    changeF1 = 0
                    self.saveMaxModel(self.model1,self.path_output1)
                else:
                    changeF1 += 1
                if maxF1_2 < max(self.f1Score2):
                    maxF1_2 = max(self.f1Score2)
                    changeF1 = 0
                    self.saveMaxModel(self.model2,self.path_output2)
                else:
                    changeF1 += 1
                if maxF1_3 < max(self.f1Score3):
                    maxF1_3 = max(self.f1Score3)
                    changeF1 = 0
                    self.saveMaxModel(self.model3,self.path_output3)
                else:
                    changeF1 += 1
                self.epochCount += 1
                print("EPOCH " + str(self.epochCount) + "\nMODEL 1\nF1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
                print("MODEL 2\nF1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
                print("MODEL 3\nF1 : " + str(self.f1Score3[len(self.f1Score3)-1]) + "\nMax F1 : " + str(maxF1_3))
            except Exception as e:
                print("ERROR5: " + str(e))
                return
        try:
            print("\nRESULT\nMODEL1\n2. Epoch F1 : " + str(self.f1Score1[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
            print("MODEL 2\n2. Epoch F1 : " + str(self.f1Score2[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
            print("MODEL 3\n2. Epoch F1 : " + str(self.f1Score3[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score3[len(self.f1Score3)-1]) + "\nMax F1 : " + str(maxF1_3))
            if not self.lenUnlabel_y < 2:
                self.unlabel_accuarcy = self.trueCount / (self.trueCount + self.falseCount)
                print("\nUnlabel accuarcy : " + str(self.unlabel_accuarcy))
            self.endTime = time.time()
            self.Time = self.endTime - self.startTime
            print("Training Time: " + str(self.Time) + " sc")
            self.showF1(self.f1Score1,"Model 1")
            self.showF1(self.f1Score2,"Model 2")
            self.showF1(self.f1Score3,"Model 3")
            self.showUnlabel()
            
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train1))
            print("Validation Dataset1 size: " + str(len_val1))
            print("Test Dataset1 size: " + str(len_test1))
        except Exception as e:
            print("ERROR6: " +str(e))

    def saveMaxModel(self,model,path_output):
        try:
            model.save(path_output)
        except Exception as e:
            print("ERROR7: " + str(e))
            
    def showF1(self,f1Score,title):
        plt.figure(figsize=(20,20))
        score = [l for l in f1Score]
        plt.plot(score)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')    
    
    def showUnlabel(self):
        barWidth = 0.15
        fig = plt.subplots(figsize =(16, 8))
         
        # set height of bar
         
        # Set position of bar on X axis
        br1 = np.arange(len(self.class_0_false))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
         
        # Make the plot
        plt.bar(br1, self.class_0_false, color ='r', width = barWidth,
                edgecolor ='grey', label ='False 0')
        plt.bar(br2, self.class_0_true, color ='lime', width = barWidth,
                edgecolor ='grey', label ='True 0')
        plt.bar(br3, self.class_1_false, color ='darkred', width = barWidth,
                edgecolor ='grey', label ='False 1')
        plt.bar(br4, self.class_1_true, color ='darkgreen', width = barWidth,
                edgecolor ='grey', label ='True 1')
        label = []
        for i in range(self.epochCount):
            label.append(i+1)
        
        # Adding Xticks
        plt.xlabel('Epoch', fontweight ='bold', fontsize = 15)
        plt.ylabel('Number', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(self.class_0_false))], label)
        plt.title("Tri-training with disagreement")
         
        plt.legend()
        plt.show()