import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import download
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Download necessary NLTK resources
download('punkt')
download('wordnet')
download('stopwords')
download('omw-1.4')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
critical_terms = {"sad", "cry", "depressed", "hopeless", "am"}  # Critical terms
stop_words = stop_words - critical_terms
additional_stopwords = {'life', 'something', 'anything', 'aand', 'abt', 'ability', 'academic', 'able', 'account', 'advance'}
stop_words.update(additional_stopwords)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'_+', '', text)  # Remove underscores
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated characters
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words).strip()

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim):
    """Load GloVe embeddings into a dictionary."""
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # The word
            coefs = values[1:]  # The vector
            # Validate if the vector length matches the embedding dimension
            if len(coefs) != embedding_dim:
                print(f"Skipping line due to dimension mismatch: {line.strip()}")
                continue
            embeddings_index[word] = np.asarray(coefs, dtype='float32')
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    """Create an embedding matrix for the tokenizer's vocabulary."""
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            print(f"Word '{word}' not found in GloVe embeddings.")
    return embedding_matrix


# Load intents
with open('Chatbot_Final/intents1.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        processed_pattern = preprocess_text(pattern)
        word_list = nltk.word_tokenize(processed_pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Tokenizer and sequences
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts([" ".join(doc[0]) for doc in documents])
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences([" ".join(doc[0]) for doc in documents])
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')

y = np.zeros((len(documents), len(classes)), dtype=int)
for i, doc in enumerate(documents):
    y[i, classes.index(doc[1])] = 1

# Load GloVe and create embedding matrix
glove_path = 'Chatbot_Final\glove\glove.840B.300d.txt'  # Update with your GloVe file path
embedding_dim = 300
embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
embedding_matrix = create_embedding_matrix(word_index, embeddings_index, embedding_dim)

# Save tokenizer and classes
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Build LSTM model with GloVe embeddings
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix],
                    input_length=max_length, trainable=False))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(len(classes), activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model_with_glove.h5')
print("Model training complete!")
