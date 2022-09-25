# Pre-trained Word Embeddings in an Embedding Layer

In text classification tasks, the ability of model to "understand" the semantic similarity of words is extremely important. People may express the same thoughts with different words, synonyms. If your NLP classification model can recognize such similarities, you can expect it to produce more accurate predictions. 

This assigment illustrates how you can take advantage of a vector representation that preserves the contextual similarity of words, showing how you can use pre-trained word embeddings where semantically related words appear closer to each other in the word embedding space.

Let's create a few sentences to play with. In the following sample, note that the first, second, and last sentence differ by one word found in the forth position. Also note that the words in question in the second and the last sentence are synonyms.

The hypothesis is whether the model trained to distinguish between the first and second sentence will be able to "understand" the last sentence - when submitted to the model for classification - should be assigned to the same class as the second sentence.
```python
texts = [
'We had to develop this script.',
'We had to delete this script',
'I need to write a script that can hear a specified port in an endless loop, producing a responce when a request arrives.',
'Do you really know how this programm works?',
'We had to remove this script.'
]
```
Now that we have some sentences to work with, let's convert them to integer sequences as the first step to converting words into embedding vectors.
```python 
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer 
maxWordsCount = 50 
sim_for_del='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer (num_words=maxWordsCount, 
                       filters=sim_for_del, 
                       lower=True, 
                       split=' ', 
                       oov_token='unknown', 
                       char_level=False)
tokenizer.fit_on_texts(texts) 
```
Let's now translate the first four sentences into integer sequences
```python
sequences = tokenizer.texts_to_sequences(texts[0:4])  
```
Then, we need to convert the sequences into a numpy array and define the sequence length.
```python
import numpy as np
sequence_length = 20
x_train = np.array(sequences)
print(x_train[0])
#padding x_train sequences to the same length.
x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=sequence_length, padding='post')
```
We also need to define the Y vector to be used in the training process. Let's assign each sentence to a separate class label.
```python
y_train=[0,1,2,3]
```
Now we are ready to proceed to embedding. Since we want to use pre-trained embeddings, we first need to download it. Here, we have several options. For example, we might use GloVe or FastText embeddings. In this assigment, we'll use the word vectors that come with a spaCy model. In the following code snippet, we download spaCy's en_core_web_lg (the one that comes with the word vectors) and then prepare a corresponding embedding matrix that we can then use in a Keras Embedding layer.
```python
import spacy 
#!python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')
num_tokens = len(tokenizer.word_index) + 1
embedding_dim = 300
# Preparing the embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for w, i in tokenizer.word_index.items():
  embedding_matrix[i] = nlp.vocab[w].vector
```
> **Note**: With libraries like Gensim, you can train word vectors on your own corpus [Training Your Own Model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#training-your-own-model).

As a clissifier, we'll use a widely-used transformer architecture for text classification (can be found in keras documentation at https://keras.io/examples/nlp/text_classification_with_transformer/)
```python
from tensorflow.keras import layers
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tensorflow.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # replacing the original Embedding layer with the one that uses our embedding matrix 
        #self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=tensorflow.keras.initializers.Constant(embedding_matrix), trainable=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen = tensorflow.shape(x)[-1]
        positions = tensorflow.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```
Then, we assembly the end-to-end model, compile it, and fit it.
```python
  embed_dim = 300 #128  # Embedding size for each token
  num_heads = 8  # Number of attention heads
  ff_dim = 128  # Hidden layer size in feed forward network inside transformer
  vocab_size = len(tokenizer.word_index) +1
  inputs = layers.Input(shape=(None,))
  embedding_layer = TokenAndPositionEmbedding(sequence_length, vocab_size, embed_dim)
  x = embedding_layer(inputs)
  transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
  x = transformer_block(x)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dropout(0.1)(x)
  x = layers.Dense(20, activation="relu")(x)
  x = layers.Dropout(0.1)(x)
  outputs = layers.Dense(4, activation="softmax")(x)
  #creating the model
  model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
  #converting y_train to numpy array
  y_train = np.asarray(y_train)
  model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  history = model.fit(x_train, y_train, batch_size=32, epochs=100)
```
Now that we have the model trained, let's check how it can understand synonyms.
```python
new_sequences = tokenizer.texts_to_sequences(texts[4:5])
x_new = np.array(new_sequences)
# padding x_train sequences to the same length.
x_new = tensorflow.keras.preprocessing.sequence.pad_sequences(x_new, maxlen=sequence_length, padding='post')
model.predict(x_new)
```
The output illustrates that the submitted sentence: 'We had to remove this script.' has been classified to the group labeled with 1. If you recall, in training set the same label is assigned to sentence: 'We had to delete this script.' 
```python
array([[0.19706866, 0.79670817, 0.00223614, 0.00398708]], dtype=float32) 
```
The following simple test explains why it works this way:
```python
print(nlp('delete').similarity(nlp('remove')))
0.6060407751427294

print(nlp('develop').similarity(nlp('remove')))
0.3461188112680473
```
The above reveals that 'remove' is much closer to 'delete' in the vector space than it is to 'develop'.
