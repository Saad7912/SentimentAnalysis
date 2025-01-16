# Sentiment Analysis with LSTM

This project involves building a sentiment analysis model using an LSTM (Long Short-Term Memory) neural network to classify text data as positive or negative. The dataset used is the IMDB movie reviews dataset.

## Project Overview

The goal of this project is to preprocess textual data, build a sequential deep learning model, and train it to perform sentiment analysis on movie reviews.

## Steps Involved

1. **Data Preprocessing**:
   - Removal of special characters and punctuation.
   - Conversion of text to lowercase.
   - Tokenization and stemming of words.
   - Removal of stop words.
   - One-hot encoding and padding to prepare data for the LSTM model.

2. **Model Building**:
   - An embedding layer to learn word representations.
   - An LSTM layer to capture sequential dependencies in text.
   - A dense output layer with sigmoid activation for binary classification.

3. **Model Training**:
   - Training the model using the binary cross-entropy loss function.
   - Using validation data to monitor training progress and avoid overfitting.

4. **Evaluation**:
   - Evaluating the model on test data to determine accuracy.
   - Visualizing training and validation accuracy over epochs.

## Key Code Snippets

### Data Preprocessing

```bash
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

onehot_repr = [one_hot(words, voc_size) for words in corpus]
embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=200)
```

### Model Architecture
```bash
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Training the Model
```bash
history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)
```

### Results
- Accuracy: The model achieved an accuracy of approximately 85% on the test set.
- Training and validation accuracy graphs indicate the model’s performance over epochs.

### Future Scope
- Experiment with other architectures like GRU or Bi-LSTM.
- Implement attention mechanisms to better capture context.
- Fine-tune hyperparameters for improved performance.

Feel free to contribute or suggest improvements!


