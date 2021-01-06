import numpy as np
import pandas as pd
import pickle
import random
import tensorflow 
import preprocessing
import os
from tensorflow import keras
from collections import Counter
from nltk.util import ngrams
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt



maximum_length =50
hidden_unit =   250
embedding_size  =  64



# Helps Us to combine the one or more file together 
# it takes list of files [file1,file2,file3] type="crossvalidation" or "train"
def combinefile(list_file,type):
    if type == "train":
      type='training'
    elif type == "cross_validation":
      type='gold'
    
    #writing all the data into one file and using it for preprocessing
    with open('full_training.utf8', 'w') as outfile:
        for fname in list_file:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    return type+"/full_training.utf8"  



# this function is used to create a dataset it takes the input
# name of the folder for example it can be list of just a file 
# for example folder_path = [file1,file2] or file
# type is the type of the file cv or train
def createDataSet(folder_path,type):
    
    if len(folder_path) > 1: 
        #if we have more than one file it means we need to combine 
        #we call the folder combinefile
        folder_path = combinefile(folder_path,type)
    else:
        #otherwise do it simple
        folder_path = "".join(folder_path)
    
    #calling the function in preprocessing for getting the x_train,y_train
    train_data,label_data = preprocessing.preprocessedText(folder_path)
    preprocessing.convertToDataFrame(train_data ,label_data,type)#saving my data in txt file


#this function helps us to read the file
#filename = "input.txt" or "label.txt"
def readFile(fileName):
    data =[]
    with open(fileName,'r',encoding='utf-8') as fp:
        for line in fp:
            data.append(line.strip())

    return data


#this function is loading pretrained embedding from the folder that we have
def loadEmbedding():
    embeddings_index = dict()
    f = open('zh/zh.vec',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

#Comprresed the dimension of pretrained embedding 
#using PCA(principle component analysis)
def compressedData(data,pad_size):
    pca = PCA(n_components=pad_size)
    return pca.fit_transform(data)
    


#Now using the pretrained Embedding I am taking the weight for two things
#first is Unigrams and bigrams 
#embedding index  = {"key":[1,2,3,4,5]} -> its key value pairs
#data it is unigram and bigram 
#pad size ->  size of the unigram and bigram 
#vocabulary -> Acutal vocabulary 

def loadWeightFromEmbedding(embeddings_index,data,pad_size,vocabulary):
    embedding_matrix = np.zeros((len(vocabulary), 300))
    for i, word in data.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
              embedding_matrix[i] = embedding_vector


    x_generic = compressedData(embedding_matrix,pad_size)
    return x_generic



#finding the unigram and bigram of the data 
#unigram = {"a","v","B"}
#bigram  =  {"ab","bc"}
#data  = ["abcbsbsbdsd","sdsdsdvsdv"....] its a x_train file
#vocabulary  = {"a":2,"v":21} 

def preTrainedEmbedding(data,vocabulary):

    unigram =[]
    bigram  =[]

    #first find the unique unigram and unique bigram

    x_data_unigram = set("".join(data))
    x_unigram_index =  {k:v for k,v in enumerate(x_data_unigram)}

    x_data_bigram  =  set(ngrams("".join(data),2))
    x_bigram_index = []
    for word in x_data_bigram:
        x_bigram_index.append("".join(word))


    x_bigram_index = {k:v for k,v in enumerate(x_bigram_index)}
    #lets work on the embedding layers 
    #create the embedding maxtrix  and load the chinese word 2 vec 
    embeddings_index = loadEmbedding()
    unigram_weightmatrix = loadWeightFromEmbedding(embeddings_index,x_unigram_index,64,vocabulary)
    bigram_weightmatrix = loadWeightFromEmbedding(embeddings_index,x_bigram_index,32,vocabulary)


    return unigram_weightmatrix,bigram_weightmatrix

    


#save dictionary using pickle
def dumpDataPickle(dictionary):
    with open("../resources/word_to_index.pickle",'wb') as handle:
        pickle.dump(dictionary,handle)
        

#load the saved dictionary using Pickle
def loadDumpDataPickle(file_path):
    with open(file_path,'rb') as handle:
        dictionary  = pickle.load(handle)
    
    return dictionary
    


#data  = [[ashcbshcbhscbsc],[ascbahscbhasbcas]......]
#this function create the vocabulary and save the vocabulary into pickle
#return word_to_index  = {"a":2,"v":3}

def findUnigram(data):
    data  = "".join(data)
    word_dict =dict()
    word_dict["<PAD>"] = 0
    word_dict["<START>"]= 1
    word_dict["UNK"] =2
    vocabulary = []
    for word in range(len(data)):
        vocabulary.append(data[word:word+2])

    vocabulary_bigram = set(vocabulary)
    vocabulary_unigram = set(data)

    Ngram = vocabulary_unigram.union(vocabulary_bigram)

    # #finding the word to index configuration 

    word_to_index =  {v:k for k,v in enumerate(Ngram)}
    index_to_word =  {k:v for k,v in enumerate(Ngram)}


    word_dict.update({k:v+len(word_dict) for k,v in word_to_index.items()})
    id_to_word=  {v:k for k,v in word_dict.items()}
    dumpDataPickle(word_dict)
    return word_dict,id_to_word


#data  = [["asckascasc","Ascahsbcasv"]....]
#word_to_index = {"as":23,"a":2,..........}
def generateX(data,word_to_index):
  #generate the unigram and generate the bigram 
  #unigram
    x_unigram = [] 
    for sentence in data:
        unigram = ngrams(sentence,1)
        vector  = []
        for word in unigram:
              word = "".join(word)
              if word_to_index.get(word) != None:
                vector.append(word_to_index.get(word))
              else:
                vector.append(word_to_index.get("UNK"))

        x_unigram.append(vector)

    x_unigram = keras.preprocessing.sequence.pad_sequences(x_unigram,truncating='pre',padding='post',maxlen=maximum_length)
  #bigram  
    x_bigram = [] 
    for sentence in data:
        bigram = ngrams(sentence,2)
        vector  = []
        for word in bigram:
              word = "".join(word)
              if word_to_index.get(word) != None:
                
                vector.append(word_to_index.get(word))
              else:
                vector.append(word_to_index.get("UNK"))

        x_bigram.append(vector)

    x_bigram = keras.preprocessing.sequence.pad_sequences(x_bigram,truncating='pre',padding='post',maxlen=maximum_length)

    return x_unigram,x_bigram

  
#we are converting the labels into numeric values 
#data  = [["bbbbies","bbiiiiies"].....]
#return [[1,2,3,2,2]....]
def generateY(data):
    output_label = {"B":0,"I":1,"E":2,"S":3}
    y_data = []
    for sentence in data:
        
        vector = []
        for word in sentence:
              vector.append(output_label[word])
        y_data.append(vector)

    y_data =keras.preprocessing.sequence.pad_sequences(y_data,truncating='pre',padding='post',maxlen=maximum_length)

    return y_data 

#Non stacking classifier Model with forward and backward LSTM

def create_keras_model(vocab_size,hidden_size,unigram_weightmatrix,bigram_weightmatrix):
    #Layer 1 for unigram embedding
    input_data  = keras.layers.Input(shape=(maximum_length,),name='unigram')
    unigram = keras.layers.Embedding(vocab_size,64,weights=[unigram_weightmatrix],mask_zero=True,trainable=True)(input_data)

    #Layer  2 for bigram embedding
    input_data_ = keras.layers.Input(shape=(maximum_length,),name='bigram')
    bigram = keras.layers.Embedding(vocab_size,32,weights=[bigram_weightmatrix],mask_zero=True,trainable=False)(input_data_)

    #concatenating of Layer1 and Layer2
    embedded = keras.layers.concatenate([unigram,bigram])

    #Backward LSTM 
    lstm1_backward   =  keras.layers.LSTM(hidden_size,return_sequences=True,dropout=0.2,recurrent_dropout=0.3,go_backwards=True)(embedded)
    #Forward LSTM
    lstm2_forward = keras.layers.LSTM(hidden_size,return_sequences=True,dropout=0.2,recurrent_dropout=0.3)(embedded)

    #Output of the both the LSTM are merged together
    merged = keras.layers.concatenate([lstm2_forward,lstm1_backward])

    #TimeDistributed gives output for each character we wrap the
    #Dense layer with 4 as output which could be ["B","I","E","S"]
    output = keras.layers.TimeDistributed(keras.layers.Dense(4,"softmax"))(merged)

    #Instantiate the Model 
    model =  keras.models.Model(inputs=[input_data,input_data_],outputs=[output])
    
    #compiling the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    return model


# Main Function From Where Every Other function is being called

def generateXYbigramUnigram():

    #read the training data from the file 
    x_train_ = readFile("write_input.txt")
    y_train_ = readFile("write_labels.txt")

    #read for the cross validation 
    x_cv_    = readFile("write_input_cv.txt")
    y_cv_    = readFile("write_labels_cv.txt")


   
    
   # getting numeric data from the function which will be feeded to the neural network

    word_to_index,index_to_word= findUnigram(x_train_)

    x_train_unigram,x_train_bigram = generateX(x_train_,word_to_index)
    y_train = generateY(y_train_)
    
    
    
    # #Now generating the values for the cross validation
    x_cv_unigram,x_cv_bigram = generateX(x_cv_,word_to_index)
    y_cv = generateY(y_cv_)
    
    
    y_train = keras.utils.to_categorical(y_train)
    y_cv    = keras.utils.to_categorical(y_cv)
    
#     #lets get the pretrained embedding weights 
    unigram_weight_matrix,bigram_weight_matrix  =  preTrainedEmbedding(x_train_,word_to_index)
    
    

    
    #checkpoint path to save the weight 
    checkPointPath  = "resources/cp.h5"
    
    #let's save the weight into the checkpoint 
    batch_size = 32
    epochs = 30



    
    model = None
    cbk = keras.callbacks.TensorBoard("logging/keras_model")
    # print("\nStarting training...")

    es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=200)
    mc = keras.callbacks.ModelCheckpoint("resources/model_weight__.h5", monitor='val_acc', mode='max', save_weights_only=True,verbose=1, save_best_only=True)


    if "model_weight__.h5" in os.listdir("../resources"):

        json_model = open("../resources/model.json")
        load_model  =  json_model.read()
        json_model.close()
       
        model = keras.models.model_from_json(load_model)
        model.load_weights("../resources/model_weight_.h5")
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
        print("Loading the Wait and training .....")

    else:
        model = create_keras_model(len(word_to_index),hidden_unit,unigram_weight_matrix,bigram_weight_matrix)
        # Let's print a summary of the model
        print(model.summary())
        

        model_json =  model.to_json()
        with open("keras_model/model.json",'w') as json_file:
            json_file.write(model_json)

        
    
    
    
    
    model.fit([x_train_unigram,x_train_bigram], y_train,epochs=epochs, batch_size=batch_size,
              shuffle=True, validation_data=([x_cv_unigram,x_cv_bigram],y_cv),callbacks=[cbk,mc,es]) 
    print("Training complete.\n")
    
    return model
   
def plotting(x,y,title,xlabel,ylabel):
  plt.title(title)
  plt.plot(x,y,'r--')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()



#creating the data set for training file
createDataSet(["msr_training.utf8"],"train")
#creating the data set for crossvalidation
createDataSet(["msr_test_gold.utf8"],"cross_validation")

history  = generateXYbigramUnigram();



#plotting Parameter


#accuracy of the data
train_accuracy = history.history.history['acc']
val_accuracy   = history.history.history['val_acc']

#lets pick up the loss
train_loss = history.history.history['loss']
validation_loss = history.history.history['val_loss']

#lets pick up the epochs
epochs = history.history.epoch

#plot the  Accuracy and losses
plotting(epochs,train_accuracy,"Train Accuracy vs Epochs","Epochs","train accuracy")
plotting(epochs,val_accuracy,"Vlidation Accuracy vs Epochs","Epochs","validation accuracy")
plotting(epochs,train_loss,"Train loss vs Epochs","Epochs","train loss")
plotting(epochs,validation_loss,"Validation loss vs Epochs","Epochs","validation loss")


