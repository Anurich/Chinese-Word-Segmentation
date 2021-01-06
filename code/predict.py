from argparse import ArgumentParser
import pickle
import tensorflow
from tensorflow import keras
import numpy as np
from nltk.util import ngrams
import sys
from sklearn.metrics import precision_score
sys.path.append("../HomeWork/icwb2-data/")

import preprocessing

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


#return x_unigram, x_bigram which will use to predict
def unigram(x_data,resources_path,pad_size):

    Ngram = pickle.load(open(resources_path+"/word_to_index.pickle",'rb'))

    x_test_unigram=[]
    for sentence in x_data:
        sentence  = sentence.strip()
        
        vector  =  []
        for word in sentence:
            if Ngram.get(word) != None:
                vector.append(Ngram[word])
            else:
                vector.append(Ngram['UNK'])

        x_test_unigram.append(vector)

    x_test_bigram = []

    for sentence in x_data:
        bigram = ngrams(sentence,2)
        vector = []
        for word in bigram:
            word = "".join(word)
            if Ngram.get(word) != None:
                vector.append(Ngram[word])
            else:
                vector.append(Ngram['UNK'])

        x_test_bigram.append(vector)
  
    x_test_unigram =keras.preprocessing.sequence.pad_sequences(x_test_unigram,truncating='pre',padding='post',maxlen=pad_size)
    x_test_bigram =keras.preprocessing.sequence.pad_sequences(x_test_bigram,truncating='pre',padding='post',maxlen=pad_size)
    

    return x_test_unigram,x_test_bigram



#this function is for saving the label for file if their is spaces in between
def saveOrignalTestFile(data,output_label,output_path):   
    origanl_file =  open(output_path+"/output.txt",'w')
    for real in data:
        origanl_file.write("".join(real))
        origanl_file.write("\n")


#Storage of the actual length of the data
def storeVectorMaximumLength(data,vector):
    for sentence in data:
        vector.append(len(sentence))
    return vector




def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    output_label = {0:"B",1:"I",2:"E",3:"S"}

    data  = open(input_path,'r',encoding='utf8').readlines()
    maximumLength = None
    orignalVector = []
    
    if len(data[0].split()) ==1:
        orignalVector = storeVectorMaximumLength(data,orignalVector)
        maximumLength = max(orignalVector)
        x_test_unigram,x_test_bigram= unigram(data,resources_path,maximumLength)


    else:
        
        x_test,y_test = preprocessing.preprocessedText(input_path)
        orignalVector = storeVectorMaximumLength(x_test,orignalVector)
        maximumLength = max(orignalVector)
        x_test_unigram,x_test_bigram = unigram(x_test,resources_path,maximumLength)
        saveOrignalTestFile(y_test,output_label,resources_path)
       
    # load json and create model
    json_file = open(resources_path+'/model.json', 'rb')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(resources_path+"/model_weight__.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    predicted_value  =  loaded_model.predict([x_test_unigram,x_test_bigram])

    pre_output = []
    for output in predicted_value:
        vector = []
        for line in output:
            vector.append(output_label.get(np.argmax(line)))
        pre_output.append(vector)
    
  

    count =0
    output_file = open(output_path,'w')
    for data in pre_output:
        data  = data[:orignalVector[count]]
        output_file.write("".join(data))
        output_file.write("\n")
        count+=1

    pass
if __name__ == '__main__':
     args = parse_args()
     predict(args.input_path, args.output_path, args.resources_path)
   # predict("../HomeWork/icwb2-data/gold/msr_test_gold.utf8","../resources/","../HomeWork/icwb2-data/")

