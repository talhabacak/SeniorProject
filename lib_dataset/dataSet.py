"""
Created on Mon Apr  4 00:31:10 2022

@author: talha
"""

import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataSet():
    def __init__(self, train, valid, test, unlabel, haveUnlabel_y=False):
        """

        Parameters
        ----------
        train : dataframe (pd)
        valid : dataframe (pd)
        test : dataframe (pd)
        unlabel : dataframe (pd)
        haveUnlabel_y : default=False 

        Returns
        -------
        None.

        """
        self.train = train
        self.train2 = train
        self.valid = valid
        self.test = test
        self.unlabel = unlabel
        self.size_of_vocabulary = 0
        self.haveUnlabel_y = haveUnlabel_y
        
    def setData(self):
        self.train.info()
        self.train2.info()
        if not len(self.valid) < 2:    
            self.valid.info()
        if not len(self.test) < 2:
            self.test.info()
        if not len(self.unlabel) < 2:
            self.unlabel.info()
        
        self.train.head(10)
        self.train2.head(10)
        if not len(self.valid) < 2:    
            self.valid.head(10)
        if not len(self.test) < 2:
            self.test.head(10)
        if not len(self.unlabel) < 2:
            self.unlabel.head(10)
    
        def Preprocessing(text):
            text = re.sub(r'[^\w\s]','',str(text))
            text = text.lower()
            text = [w for w in text.split(' ') if w not in stopwords.words('english')]
            text = [WordNetLemmatizer().lemmatize(token) for token in text]
            text = [WordNetLemmatizer().lemmatize(token,pos='v') for token in text]
            text = " ".join(text)
            return text
    
        self.train['text'] = self.train.text.apply(lambda x:Preprocessing(x))
        print("train finished")
        self.train2['text'] = self.train2.text.apply(lambda x:Preprocessing(x))
        print("train2 finished")
        if not len(self.valid) < 2:
            self.valid['text'] = self.valid.text.apply(lambda x:Preprocessing(x))
            print("valid finished")
        if not len(self.test) < 2:
            self.test['text']= self.test.text.apply(lambda x:Preprocessing(x))
            print("test finished")
        if not len(self.unlabel) < 2:
            self.unlabel['text']= self.unlabel.text.apply(lambda x:Preprocessing(x))
            print("unlabel finished")
    
        self.train['label'].value_counts()
        self.train2['label'].value_counts()
        if not len(self.valid) < 2:
            self.valid['label'].value_counts()
        if not len(self.test) < 2:
            self.test['label'].value_counts()
        if not len(self.unlabel) < 2:
            self.unlabel['label'].value_counts()
    
        self.train_x = self.train['text']
        self.train_x2 = self.train2['text']
        if not len(self.valid) < 2:
            self.valid_x = self.valid['text']
        if not len(self.test) < 2:
            self.test_x = self.test['text']
        if not len(self.unlabel) < 2:
            self.unlabel_x = self.unlabel['text']
        self.train_y = self.train['label']
        self.train_y2 = self.train2['label']
        if not len(self.valid) < 2:
            self.valid_y = self.valid['label']
        if not len(self.test) < 2:
            self.test_y = self.test['label']
        if self.haveUnlabel_y:
            self.unlabel_y = self.unlabel['label']
        else:
            self.unlabel_y = [-1]
        plt.figure(figsize=(16,20))
        plt.style.use('fivethirtyeight')
    
        plt.subplot(4,1,1)
        train_len = [len(l) for l in self.train_x]
        train_len2 = [len(l) for l in self.train_x2]
        plt.hist(train_len,bins=50)
        plt.title('Distribution of train text length')
        plt.xlabel('Length')

        plt.hist(train_len2,bins=50)
        plt.title('Distribution of train2 text length')
        plt.xlabel('Length')
    
        if not len(self.valid) < 2:
            plt.subplot(4,1,2)
            valid_len = [len(l) for l in self.valid_x]
            plt.hist(valid_len,bins=50,color='green')
            plt.title('Distribution of valid text length')
            plt.xlabel('Length')
        
        if not len(self.test) < 2:
            plt.subplot(4,1,3)
            test_len = [len(l) for l in self.test_x]
            plt.hist(test_len,bins=50,color='red')
            plt.title('Distribution of test text length')
            plt.xlabel('Length')
        
        if not len(self.unlabel) < 2:
            plt.subplot(4,1,4)
            unlabel_len = [len(l) for l in self.unlabel_x]
            plt.hist(unlabel_len,bins=50,color='orange')
            plt.title('Distribution of unlabel text length')
            plt.xlabel('Length')
    
        plt.show()
    
        plt.figure(figsize=(20,20))
        pos_freq = FreqDist(' '.join(self.train[self.train['label'] == 1].text).split(' '))
        wc = WordCloud().generate_from_frequencies(frequencies=pos_freq)
        plt.imshow(wc,interpolation='bilinear')
        plt.title('Positive Review Common Text')
        plt.axis('off')
        plt.show()
    
        plt.figure(figsize=(20,6))
        pos_freq.plot(50,cumulative=False,title='Positive Review Common Text')
        plt.show()
    
        plt.figure(figsize=(20,20))
        neg_freq = FreqDist(' '.join(self.train[self.train['label'] == 0].text).split(' '))
        wc = WordCloud().generate_from_frequencies(frequencies=neg_freq)
        plt.imshow(wc,interpolation='bilinear')
        plt.title('Negative Review Common Text')
        plt.axis('off')
        plt.show()
    
        plt.figure(figsize=(20,6))
        neg_freq.plot(50,cumulative=False,title='Negative Review Common Text',color='red')
        plt.show()
    
        plt.figure(figsize=(20,20))
        pos_freq = FreqDist(' '.join(self.train2[self.train2['label'] == 1].text).split(' '))
        wc = WordCloud().generate_from_frequencies(frequencies=pos_freq)
        plt.imshow(wc,interpolation='bilinear')
        plt.title('Positive Review Common Text2')
        plt.axis('off')
        plt.show()
    
        plt.figure(figsize=(20,6))
        pos_freq.plot(50,cumulative=False,title='Positive Review Common Text2')
        plt.show()
    
        plt.figure(figsize=(20,20))
        neg_freq = FreqDist(' '.join(self.train2[self.train2['label'] == 0].text).split(' '))
        wc = WordCloud().generate_from_frequencies(frequencies=neg_freq)
        plt.imshow(wc,interpolation='bilinear')
        plt.title('Negative Review Common Text2')
        plt.axis('off')
        plt.show()
    
        plt.figure(figsize=(20,6))
        neg_freq.plot(50,cumulative=False,title='Negative Review Common Text2',color='red')
        plt.show()

    
        #Tokenize the sentences
        tokenizer = Tokenizer()
        #preparing vocabulary
        tokenizer.fit_on_texts(self.train_x)
        #converting text into integer sequences
        self.train_x = tokenizer.texts_to_sequences(self.train_x)
        self.train_x2 = tokenizer.texts_to_sequences(self.train_x2)
        if not len(self.valid) < 2:
            self.valid_x = tokenizer.texts_to_sequences(self.valid_x)
        if not len(self.test) < 2:
            self.test_x = tokenizer.texts_to_sequences(self.test_x)
        if not len(self.unlabel) < 2:
            self.unlabel_x = tokenizer.texts_to_sequences(self.unlabel_x)
        #padding to prepare sequences of same length
        self.train_x = pad_sequences(self.train_x,maxlen=120)
        self.train_x2 = pad_sequences(self.train_x2,maxlen=120)
        if not len(self.valid) < 2:
            self.valid_x=pad_sequences(self.valid_x,maxlen=120)
        if not len(self.test) < 2:
            self.test_x=pad_sequences(self.test_x,maxlen=120)
        if not len(self.unlabel) < 2:
            self.unlabel_x=pad_sequences(self.unlabel_x,maxlen=120)
    
        self.size_of_vocabulary = len(tokenizer.word_index)+1
        print(self.size_of_vocabulary)

    
    
class DataSetGeneral():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel, y_unlabel):
        """

        Parameters
        ----------
        train : dataframe (pd)
        valid : dataframe (pd)
        test : dataframe (pd)
        unlabel : dataframe (pd)

        Returns
        -------
        None.

        """
        self.train_x = x_train
        self.train_x2 = x_train
        self.train_y = y_train
        self.train_y2 = y_train
        self.valid_x = x_val
        self.valid_y = y_val
        self.test_x = x_test
        self.test_y = y_test
        self.unlabel_x = x_unlabel
        self.unlabel_y = y_unlabel