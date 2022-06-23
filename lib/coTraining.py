"""
Created on Mon Apr  4 19:26:19 2022

@author: talha
"""

import numpy as np
import keras.backend as K  
import matplotlib.pyplot as plt
import time

class CoTraining():
    def __init__(self,model1,model2,data1,ratio=4,epoch=5,class_number=2,path_output1="model1_best.h5",path_output2="model2_best.h5",activation="sigmoid"):
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
        self.model1 = model1
        self.X_train1 = np.array(data1.train_x)
        self.X_val1 = np.array(data1.valid_x)
        self.X_test1 = np.array(data1.test_x)
        self.y_train1 = np.array(data1.train_y)
        self.y_val1 = np.array(data1.valid_y)
        self.y_test1 = np.array(data1.test_y)
        self.X_unlabel1 = np.array(data1.unlabel_x)
        self.y_unlabel1 = np.array(data1.unlabel_y)
        self.lenUnlabel_y = len(self.y_unlabel1)

        self.model2 = model2
        self.X_train2 = np.array(data1.train_x2)
        self.y_train2 = np.array(data1.train_y2)
        
        self.limit = int(len(self.X_unlabel1) / 4 * ratio)
        self.ratio = ratio
        self.epoch = epoch
        self.activation = activation
        self.class_number = class_number
        self.perEpoch_label = int((self.limit + 1) / self.epoch)
        self.path_output1 = path_output1
        self.path_output2 = path_output2
        self.history1 = []
        self.f1Score1 = []
        self.history2 = []
        self.f1Score2 = []
        self.loss1 = []
        self.loss2 = []
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
        
        self.X_unlabel1 = self.X_unlabel1[ : self.limit]
        if not self.lenUnlabel_y < 2:
            self.y_unlabel1 = self.y_unlabel1[ : self.limit]

        
    def moveClass(self,prediction):
        result = []
        for i in prediction:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return np.asarray(result)
        
    def moveData(self,y_pred,index,modelNumber):
        if modelNumber == 1:
                np.append(self.X_train1, self.X_unlabel1[index])
                np.append(self.y_train1, y_pred)
        elif modelNumber == 2:
                np.append(self.X_train2, self.X_unlabel1[index])
                np.append(self.y_train2, y_pred)
        if not self.lenUnlabel_y < 2:
            if self.y_unlabel1[index] == y_pred:
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
        self.y_unlabel1 = np.delete(self.y_unlabel1, index, axis=0)
        self.X_unlabel1 = np.delete(self.X_unlabel1, index, axis=0)

    def moveData2(self,y_pred,index,modelNumber):
        if modelNumber == 1:
            np.append(self.X_train1, self.X_unlabel1[index])
            np.append(self.y_train1, y_pred)
        if modelNumber == 2:
            np.append(self.X_train2, self.X_unlabel1[index])
            np.append(self.y_train2, y_pred)
        if not self.lenUnlabel_y < 2:
            if self.y_unlabel1[index][y_pred] == 1:
                self.trueCount += 1
            else:
                self.falseCount += 1
        self.y_unlabel1 = np.delete(self.y_unlabel1, index, axis=0)
        self.X_unlabel1 = np.delete(self.X_unlabel1, index, axis=0)

    def getPseudoLabel_sigmoid(self):
        """
        CO-TrainingAlgorithm

        Returns
        -------
        None.

        """
        try:
            predictions1 = self.model1.predict(self.X_unlabel1)
            y_pred1 = self.moveClass(predictions1)
            len_predictions = len(predictions1)
            predictions2 = self.model2.predict(self.X_unlabel1)
            y_pred2 = self.moveClass(predictions2)
        except Exception as e:
            print("ERROR2_1: " + str(e))
            return
        i=0
        while i < self.perEpoch_label and i < len_predictions:
            try:
                control = 0
                best1 = np.where(predictions1 == np.amax(predictions1))[0][0]
                best2 = np.where(predictions2 == np.amax(predictions2))[0][0]
                if best2 < best1:
                    if y_pred1[best1] == 1:
                        self.moveData(y_pred1[best1], best1,2)
                        predictions1 = np.delete(predictions1, best1,axis=0)
                        predictions2 = np.delete(predictions2, best1,axis=0)
                        y_pred1 = np.delete(y_pred1, best1,axis=0)
                        y_pred2 = np.delete(y_pred2, best1,axis=0)
                        self.pseudoLabel_number += 1
                        i += 1
                        control = 1
                else:
                    if y_pred2[best2] == 1:
                        self.moveData(y_pred2[best2], best2,1)
                        predictions2 = np.delete(predictions2, best2,axis=0)
                        predictions1 = np.delete(predictions1, best2,axis=0)
                        y_pred2 = np.delete(y_pred2, best2,axis=0)
                        y_pred1 = np.delete(y_pred1, best2,axis=0)
                        self.pseudoLabel_number += 1
                        i += 1
                        control = 1
            except Exception as e:
                print("ERROR2_2: " + str(e))
                break
            if i < self.perEpoch_label and i < len_predictions:
                try:
                    worst1 = np.where(predictions1 == np.amin(predictions1))[0][0]
                    worst2 = np.where(predictions2 == np.amin(predictions2))[0][0]
                    if worst2 > worst1:
                        if y_pred1[worst1] == 0:
                            self.moveData((y_pred1[worst1]), worst1,2)
                            predictions1 = np.delete(predictions1, worst1,axis=0)
                            predictions2 = np.delete(predictions2, worst1,axis=0)
                            y_pred1 = np.delete(y_pred1, worst1,axis=0)
                            y_pred2 = np.delete(y_pred2, worst1,axis=0)
                            self.pseudoLabel_number += 1
                            i += 1
                            control = 1
                    else:
                        if y_pred2[worst2] == 0:
                            self.moveData((y_pred2[worst2]), worst2,1)
                            predictions2 = np.delete(predictions2, worst2,axis=0)
                            predictions1 = np.delete(predictions1, worst2,axis=0)
                            y_pred2 = np.delete(y_pred2, worst2,axis=0)
                            y_pred1 = np.delete(y_pred1, worst2,axis=0)
                            self.pseudoLabel_number += 1
                            i += 1
                            control = 1
                except Exception as e:
                    print("ERROR2_3: " + str(e))
                    continue
                if control == 0:
                    try:
                        if y_pred1[worst1] == 1:
                            self.moveData((y_pred1[best1]), best1,2)
                            predictions1 = np.delete(predictions1, best1,axis=0)
                            predictions2 = np.delete(predictions2, best1,axis=0)
                            y_pred1 = np.delete(y_pred1, best1,axis=0)
                            y_pred2 = np.delete(y_pred2, best1,axis=0)
                            self.pseudoLabel_number += 1
                            i += 1
                        else:
                            self.moveData((y_pred1[worst1]), worst1,2)
                            predictions1 = np.delete(predictions1, worst1,axis=0)
                            predictions2 = np.delete(predictions2, worst1,axis=0)
                            y_pred1 = np.delete(y_pred1, worst1,axis=0)
                            y_pred2 = np.delete(y_pred2, worst1,axis=0)
                            self.pseudoLabel_number += 1
                            i += 1
                    except Exception as e:
                        print("ERROR2_4: " + str(e))

    def getPseudoLabel_softmax(self):
        """
        CO-TrainingAlgorithm

        Returns
        -------
        None.

        """
        try:
            predictions1 = self.model1.predict(self.X_unlabel1)
            len_predictions = len(predictions1)
            predictions2 = self.model2.predict(self.X_unlabel1)
        except Exception as e:
            print("ERROR2_3: " + str(e))
            return
        i=0
        while i < self.perEpoch_label and i < len_predictions:
            try:
                y_pred1 = np.argmax(predictions1)
                y_pred2 = np.argmax(predictions2)
                best1_1 = np.where(predictions1 == np.amax(predictions1))[0][0]
                best1_2 = np.where(predictions1 == np.amax(predictions1))[1][0]
                best2_1 = np.where(predictions2 == np.amax(predictions2))[0][0]
                best2_2 = np.where(predictions2 == np.amax(predictions2))[1][0]
                if predictions2[best2_1][best2_2] < predictions1[best1_1][best1_2]:
                    self.moveData2(best1_2, best1_1, 2)
                    predictions1 = np.delete(predictions1, best1_1,axis=0)
                    predictions2 = np.delete(predictions2, best1_1,axis=0)
                    self.pseudoLabel_number += 1
                    i += 1
                    control = 1
                else:
                    self.moveData2(best2_2,best2_1, 1)
                    predictions2 = np.delete(predictions2, best2_1,axis=0)
                    predictions1 = np.delete(predictions1, best2_1,axis=0)
                    self.pseudoLabel_number += 1
                    i += 1
                    control = 1
            except Exception as e:
                print("ERROR2_4: " + str(e))
                break
                                   
    def f1_score(self,precision, recall):
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val    
    
    def run(self):
        try:
            len_unlabel = len(self.X_unlabel1)
            len_train1 = len(self.X_train1)
            len_train1_y = len(self.y_train1)
            len_val1 = len(self.X_val1)
            len_val1_y = len(self.y_val1)
            len_test1 = len(self.X_test1)
            len_train2 = len(self.X_train2)
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train1))
            print("Train Dataset1-y size: " + str(len_train1_y))
            print("Validation Dataset1 size: " + str(len_val1))
            print("Validation Dataset1-y size: " + str(len_val1_y))
            print("Test Dataset1 size: " + str(len_test1))
            print("Train Dataset2 size: " + str(len_train2))
        except Exception as e:
            print("ERROR3: " + str(e))
            return
        self.pseudoLabel_number = 0
        self.epochCount = 0
        changeF1 = 0
        maxF1_1 = 0
        maxF1_2 = 0
        self.startTime = time.time()
        while len(self.X_unlabel1) > 0:
            #try:
            self.history1.append(self.model1.fit(self.X_train1, self.y_train1, epochs=1, 
                      validation_data=(self.X_val1, self.y_val1)))
            self.history2.append(self.model2.fit(self.X_train2, self.y_train2, epochs=1, 
                      validation_data=(self.X_val1, self.y_val1)))            
            #except Exception as e:
            #    print("ERROR4: " + str(e))
            #    return
            if len(self.X_unlabel1) > 0:
                if self.activation == "sigmoid":
                    self.getPseudoLabel_sigmoid()
                elif self.activation == "softmax":
                    self.getPseudoLabel_softmax()
            try:
                loss1, accuracy1, precision1, recall1 = self.model1.evaluate(self.X_test1, self.y_test1, verbose=0)
                loss2, accuracy2, precision2, recall2 = self.model2.evaluate(self.X_test1, self.y_test1, verbose=0)
                self.f1Score1.append(self.f1_score(precision1, recall1))
                self.f1Score2.append(self.f1_score(precision2, recall2))
                self.loss1.append(loss1)
                self.loss2.append(loss2)
                if maxF1_1 < max(self.f1Score1):
                    maxF1_1 = max(self.f1Score1)
                    changeF1 = 0
                    self.saveMaxModel(self.model1,self.path_output1)
                if maxF1_2 < max(self.f1Score2):
                    maxF1_2 = max(self.f1Score2)
                    changeF1 = 0
                    self.saveMaxModel(self.model2,self.path_output2)
                else:
                    changeF1 += 1
                self.epochCount += 1
                print("EPOCH " + str(self.epochCount) + "\nMODEL 1\nF1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
                print("MODEL2\nF1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
                print("# of remaining unlabeled: " + str(len(self.X_unlabel1)) + "\n# of pseudo label : " + str(self.pseudoLabel_number) + "\n")
            except Exception as e:
                print("ERROR5: " + str(e))
                return
        try:
            print("\nRESULT\nMODEL1\n2. Epoch F1 : " + str(self.f1Score1[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score1[len(self.f1Score1)-1]) + "\nMax F1 : " + str(maxF1_1))
            print("MODEL2\n2. Epoch F1 : " + str(self.f1Score2[1]) + "\n" + str(self.epochCount) + ". Epoch F1 : " + str(self.f1Score2[len(self.f1Score2)-1]) + "\nMax F1 : " + str(maxF1_2))
            if not self.lenUnlabel_y < 2:
                self.unlabel_accuarcy = self.trueCount / (self.trueCount + self.falseCount)
                print("\nUnlabel accuarcy : " + str(self.unlabel_accuarcy))
            self.endTime = time.time()
            self.Time = self.endTime - self.startTime
            print("Training Time: " + str(self.Time) + " sc")
            self.showF1(self.f1Score1,"Model 1")
            self.showF1(self.f1Score2,"Model 2")
            self.showUnlabel()
            
            print("\nUnlabeled Dataset size: " + str(len_unlabel))
            print("Train Dataset1 size: " + str(len_train1))
            print("Validation Dataset1 size: " + str(len_val1))
            print("Test Dataset1 size: " + str(len_test1))
            print("Train Dataset2 size: " + str(len_train2))
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
        plt.title("Co-training")
         
        plt.legend()
        plt.show()
