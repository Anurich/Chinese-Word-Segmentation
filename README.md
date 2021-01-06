# Chinese Word Segmentation
Chinese word segmentation is a necessary first step in Chinese language processing and there
are many approach to solve this problem of CWS one of the approach is using neural network, 
In this paper I will discuss about implementation, Preprocessing, Network Architecture,
Hyperparameter and result.

<h2>Dataset Information & Preprocessing.</h2>
The dataset is available https://chinesenlp.xyz/docs/word_segmentation.html. <br/> 
Preprocessing Step:
 <ul> 
  <li> Constructing the vocabulary {word_to_index, index_to_word} </li>
  <li> Seprating the data in train and test </li>
  <li> Using the pretrained Embeddings </li>
</ul>

<h2> Model & Loss Information </h2>
I have implemented  forward and backward LSTM, I am using two embedding layer one for unigram and another for bigram,
And concatenating those two layers output and feeding it to both backward and forward LSTM
the output of those is than concatenated and feed through time distributed softmax classifier
with 4 output. <br/> Loss used is  <b>‘categorical_crossentropy’</b>  and metrics used is  <b>‘accuracy’</b>.

<h2> Hyperparameters. </h2>
<ol>
 <li>Padding Size = 50</li> 
  <li>Unigram Embedding Size = 64</li>
  <li>Bigram Embedding Size = 32</li>
  <li>Dropout = 0.1 ,Recurrent Dropout = 0.3, Batch Size = 64 </li>
</ol>
<h2> Installation </h2>
<ul>
 <li>Python == 3.6.6</li>
 <li>Keras</li>
 </ul>
 
<h2> Some Important Information </h2>
 I didn't upload the dataset here as well didn't upload pretrained embedding, you can download it from  here: http://vectors.nlpl.eu/repository/.
 <h2> Result </h2>
<p float="left">
  <img src="resources/Train Accuracy vs Epochs.jpg" width="400" />
  <img src="resources/Train loss vs Epochs.jpg" width="400" /> 
  <img src="resources/Validation loss vs Epochs.jpg" width="400" />
 <img src="resources/Vlidation Accuracy vs Epochs.jpg" width="400" />
</p>
<h2> Note. </h2>
If anyone find any problem in understanding code as well as in running it, feel free to send me mail and ask all your queries and doubt.

