# Chinese Word Segmentation
Chinese word segmentation is a necessary first step in Chinese language processing and there
are many approach to solve this problem of CWS one of the approach is using neural network, 
In this paper I will discuss about implementation, Preprocessing, Network Architecture,
Hyperparameter and result.

<h2>Dataset Information & Preprocessing.</h2>
The dataset is available https://chinesenlp.xyz/docs/word_segmentation.html.  
Preprocessing Step:
 <ul> 
  <li> Constructing the vocabulary {word_to_index, index_to_word} </li>
  <li> Seprating the data in train and test </li>
  <li> Using the pretrained Embeddings </li>
</ul>
