# all necessary imports
import pickle
import re
import string
from os import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras.layers import Embedding, CuDNNLSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# for pre processing
stop_words = stopwords.words('english')
punctuations = string.punctuation.replace("'", "")
lemmatizer = WordNetLemmatizer()

# dataset loading and analysis
movie_sentiment_df = pd.read_csv("IMDB Dataset.csv")
# print(f"Shape of DF = {movie_sentiment_df.shape}")
reviews = movie_sentiment_df['review']
sentiment = movie_sentiment_df['sentiment']
sentiment = np.array(list(map(lambda x: 1 if x == "positive" else 0, sentiment)))
# analysis plot (count of each class)
ax = sns.countplot(x="sentiment", data=movie_sentiment_df)
plt.title("Data analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x() + 0.35, p.get_height() + 50))
plt.show()

# data pre-processing
def preprocess(text):
    clean_text = text.lower()  # lower casing of text
    clean_text = re.sub(r"\s+[0-9]*\s+", ' ', clean_text)  # removing numbers
    clean_text = re.sub(r"[<]\w+\s[/][>]", ' ', clean_text)  # removing tags
    clean_text = clean_text.translate(str.maketrans('', '', punctuations))  # removing punctuation except '
    clean_text = ' '.join([lemmatizer.lemmatize(i) for i in clean_text.split() if
                           i not in stop_words])  # lemmatization and stop word removal
    return clean_text


# saving pre processed reviews
clean_reviews = []

if path.exists("clean_reviews_pickle"):
    clean_reviews_pickle = open("clean_reviews_pickle", "rb")
    clean_reviews = pickle.load(clean_reviews_pickle)
    clean_reviews_pickle.close()
else:
    for review in reviews:
        clean_reviews.append(preprocess(review))
    clean_reviews_pickle = open("clean_reviews_pickle", "wb")
    pickle.dump(clean_reviews, clean_reviews_pickle)
    clean_reviews_pickle.close()

# split the data in 80:20 train test ratio
X_train, X_test, y_train, y_test = train_test_split(clean_reviews, sentiment, test_size=0.2, random_state=42)

max_words = 5000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

if path.exists("X_train_tts_pickle") and path.exists("X_test_tts_pickle"):
    X_train_tts_pickle = open("X_train_tts_pickle", "rb")
    X_train = pickle.load(X_train_tts_pickle)
    X_train_tts_pickle.close()
    X_test_tts_pickle = open("X_test_tts_pickle", "rb")
    X_test = pickle.load(X_test_tts_pickle)
    X_test_tts_pickle.close()
else:
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train_tts_pickle = open("X_train_tts_pickle", "wb")
    pickle.dump(X_train, X_train_tts_pickle)
    X_train_tts_pickle.close()
    X_test_tts_pickle = open("X_test_tts_pickle", "wb")
    pickle.dump(X_test, X_test_tts_pickle)
    X_test_tts_pickle.close()

vocab_size = len(tokenizer.word_index) + 1
max_len = 100

if path.exists("X_train_pickle") and path.exists("X_test_pickle"):
    X_train_pickle = open("X_train_pickle", "rb")
    X_train = pickle.load(X_train_pickle)
    X_train_pickle.close()
    X_test_pickle = open("X_test_pickle", "rb")
    X_test = pickle.load(X_test_pickle)
    X_test_pickle.close()
else:
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
    X_train_pickle = open("X_train_pickle", "wb")
    pickle.dump(X_train, X_train_pickle)
    X_train_pickle.close()
    X_test_pickle = open("X_test_pickle", "wb")
    pickle.dump(X_test, X_test_pickle)
    X_test_pickle.close()

if path.exists("embedding_dict_pickle"):
    embedding_dict_pickle = open("embedding_dict_pickle", "rb")
    embedding_dict = pickle.load(embedding_dict_pickle)
    embedding_dict_pickle.close()
# embedding for reviews
else:
    glove_file = open("glove.6B.100d.txt", "r", encoding='utf-8')
    embedding_dict = {}
    for each_row in glove_file:
        line = each_row.split()
        word = line[0]
        embedding = np.array([float(val) for val in line[1:]])
        embedding_dict[word] = embedding
    embedding_dict_pickle = open("embedding_dict_pickle", "wb")
    pickle.dump(embedding_dict, embedding_dict_pickle)
    embedding_dict_pickle.close()

if path.exists("embedding_matrix_pickle"):
    embedding_matrix_pickle = open("embedding_matrix_pickle", "rb")
    embedding_matrix = pickle.load(embedding_matrix_pickle)
    embedding_matrix_pickle.close()
else:
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        vector = embedding_dict.get(word)
        if vector is not None:
            embedding_matrix[index] = vector
    embedding_matrix_pickle = open("embedding_matrix_pickle", "wb")
    pickle.dump(embedding_matrix, embedding_matrix_pickle)
    embedding_matrix_pickle.close()

# applying CuDNNLSTM for classification
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(embedding_layer)
model.add(CuDNNLSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)

print(f"Training Set accuracy = {history.history['acc'][-1]*100}%")
print(f"Validation Set accuracy = {history.history['val_acc'][-1]*100}%")
print(f"Test Set Accuracy = {score[1]*100}%")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

# model accuracy plot
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# model loss plot
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


