# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 06:38:35 2022

@author: talha
"""

import numpy as np
import keras.backend as K  
import matplotlib.pyplot as plt
import time

class SelfTraining():
    def __init__(self,model,data,ratio=4,epoch=5,class_number=2,path_output="model_best.h5",activation="sigmoid"):
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
        self.model = model
        self.X_train = np.array(data.train_x)
        self.X_val = np.array(data.valid_x)
        self.X_test = np.array(data.test_x)
        self.y_train = np.array(data.train_y)
        self.y_val = np.array(data.valid_y)
        self.y_test = np.array(data.test_y)
        self.X_unlabel = np.array(data.unlabel_x)
        self.y_unlabel = np.array(data.unlabel_y)
        self.lenUnlabel_y = len(self.y_unlabel)

        self.ratio = ratio
        self.limit = int(len(self.X_unlabel) / 4 * ratio)
        self.class_number = class_number
        self.activation = activation
        self.epoch = epoch
        self.perEpoch_label = int((self.limit + 1) / self.epoch)
        self.path_output = path_output
        self.history = []
        self.f1Score = []
        self.loss = []
        self.pseudoLabel_number = 0
        self.trueCount = 0
        self.falseCount = 0
        self.endTime = 0
        self.startTime = 0
        self.Time = 0
        self.unlabel_accuarcy = 0
        
        self.class_0_true = []
        self.class_1_true = []
        self.class_0_false = []
        self.class_1_false = []

        self.epochCount = 0
        
        for i in range(self.epoch):
            self.class_0_true.append(0)
            self.class_1_true.append(0)
            self.class_0_false.append(0)
            self.class_1_false.append(0)
        
        self.X_unlabel = self.X_unlabel[ : self.limit]
        if not self.lenUnlabel_y < 2:
            self.y_unlabel = self.y_unlabel[ : self.limit]

    def moveClass(self,prediction):
        result = []
        for i in prediction:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return np.asarray(result)
        
    def moveData(self,y_pred,index):
        #try:
            np.append(self.X_train, self.X_unlabel[index])
            np.append(self.y_train, y_pred)
            if not self.lenUnlabel_y < 2:
                if self.y_unlabel[index] == y_pred:
                    self.trueCount += 1
                    if y_pred == 0:
                        self.class_0_true[self.epochCount] += 1
                    else:
                        self.class_1_true[self.epochCount] += 1                        
                else:
                    if y_pred == 0:
                        self.class_0_false[self.epochCount] += 1
                    else:
                        self.class_1_false[self.epochCount] += 1                        
                    self.falseCount += 1
                self.y_unlabel = np.delete(self.y_unlabel, index, axis=0)
            self.X_unlabel = np.delete(self.X_unlabel, index, axis=0)
        #except Exception as e:
        #    print("ERROR1: " + str(e))        

    def moveData2(self,y_pred,index):
        #try:
            np.append(self.X_train, self.X_unlabel[index])
            np.append(self.y_train, y_pred)
            if not self.lenUnlabel_y < 2:
                if self.y_unlabel[index][y_pred] == 1:
                    self.trueCount += 1
                else:
                    self.falseCount += 1
                self.y_unlabel = np.delete(self.y_unlabel, index, axis=0)
            self.X_unlabel = np.delete(self.X_unlabel, index, axis=0)
        #except Exception as e:
        #    print("ERROR1: " + str(e))      
    
    def getPseudoLabel_sigmoid(self):
        """
        Self Learning Algorithm

        Returns
        -------
        None.

        """
        try:
            predictions = self.model.predict(self.X_unlabel)
            y_pred = self.moveClass(predictions)
            len_predictions = len(predictions)
        except Exception as e:
            print("ERROR2_1: " + str(e))
            return
        i=0
        while i < self.perEpoch_label and i < len_predictions:
            try:
                best = np.where(predictions == np.amax(predictions))[0][0]
                if y_pred[best] == 1:
                    self.moveData(y_pred[best], best)
                    predictions = np.delete(predictions, best,axis=0)
                    y_pred = np.delete(y_pred, best,axis=0)
                    self.pseudoLabel_number += 1
                    i += 1
            except Exception as e:
                print("ERROR2_2: " + str(e))
                break
            if self.class_number == 2 and i < self.perEpoch_label and i < len_predictions:
                try:
                    worst = np.where(predictions == np.amin(predictions))[0][0]
                    if y_pred[worst] == 0:
                        self.moveData((y_pred[worst]), worst)
                        predictions = np.delete(predictions, worst,axis=0)
                        y_pred = np.delete(y_pred, worst,axis=0)
                        self.pseudoLabel_number += 1
                        i += 1
                except:
                    continue
                                   
    def getPseudoLabel_softmax(self):
        """
        Self Learning Algorithm

        Returns
        -------
        None.

        """
        try:
            predictions = self.model.predict(self.X_unlabel)
            len_predictions = len(predictions)
        except Exception as e:
            print("ERROR2_3: " + str(e))
            return
        i=0
        while i < self.perEpoch_label and i < len_predictions:
            try:
                y_pred = np.argmax(predictions)
                best1 = np.where(predictions == np.amax(predictions))[0][0]
                best2 = np.where(predictions == np.amax(predictions))[1][0]
                self.moveData2(best2, best1)
                predictions = np.delete(predictions, best1,axis=0)
                self.pseudoLabel_number += 1
                i += 1
            except Exception as e:
                print("ERROR2_4: " + str(e))
                break

                
    def f1_score(self,precision, recall):
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val    
    
    def run(self):
        try:
            len_unlabel = len(self.X_unlabel)
            len_train = len(self.X_train)
            len_val = len(self.X_val)
            len_test = len(self.X_test)
        except Exception as e:
            print("ERROR3: " + str(e))
            return
        self.pseudoLabel_number = 0
        self.epochCount = 0
        changeF1 = 0
        maxF1 = 0
        self.startTime = time.time()
        while len(self.X_unlabel) > 0:
            try:
                self.history.append(self.model.fit(self.X_train, self.y_train, epochs=1, 
                          validation_data=(self.X_val, self.y_val)))
            except Exception as e:
                print("ERROR4: " + str(e))
                return
#            print(self.history[self.epochCount].history['loss'])
            if len(self.X_unlabel) > 0:
                if self.activation == "sigmoid":
                    self.getPseudoLabel_sigmoid()
                elif self.activation == "softmax":
                    self.getPseudoLabel_softmax()
                    
            try:
                loss, accuracy, precision, recall = self.model.evaluate(self.X_test, self.y_test, verbose=0)
                self.f1Score.append(self.f1_score(precision, recall))
                self.loss.append(loss)
                if maxF1 < max(self.f1Score):
                    maxF1 = max(self.f1Score)
                    changeF1 = 0
                    self.saveMaxModel()
                else:
                    changeF1 += 1
                self.epochCount += 1
                print("EPOCH " + str(self.epochCount) + "\nF1: " + str(self.f1Score[len(self.f1Score)-1]) + " - Max F1: " + str(maxF1))
                print("# of remaining unlabeled: " + str(len(self.X_unlabel)) + "\n# of pseudo label: " + str(self.pseudoLabel_number) + "\n")
            except Exception as e:
                print("ERROR5: " + str(e))
                return
            
        try:
            print("\nRESULT\n2. Epoch Test F1 :" + str(self.f1Score[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score[len(self.f1Score)-1]) + "\nMax F1 : " + str(maxF1))
            if not self.lenUnlabel_y < 2:
                self.unlabel_accuarcy = self.trueCount / (self.trueCount + self.falseCount)
                print("\nUnlabel accuarcy : " + str(self.unlabel_accuarcy))
            self.endTime = time.time()
            self.Time = self.endTime - self.startTime
            print("Training Time : " + str(self.Time) + " sc")
            self.showF1()
            self.showUnlabel()
            
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train))
            print("Validation Dataset1 size: " + str(len_val))
            print("Test Dataset1 size: " + str(len_test))
        except Exception as e:
            print("ERROR6: " +str(e))

    def saveMaxModel(self):
        try:
            self.model.save(self.path_output)
        except Exception as e:
            print("ERROR7: " + str(e))
            
    def showF1(self):
        plt.figure(figsize=(20,20))
        score = [l for l in self.f1Score]
        plt.plot(score)
        plt.title('F1 Score')
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
        plt.title("Self-training")

        plt.legend()
        plt.show()

        
        
