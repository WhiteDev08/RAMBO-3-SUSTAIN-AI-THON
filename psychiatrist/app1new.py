import os
import random
import json
import pickle
import numpy as np
import re
import nltk
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Add domain-specific terms to keep
critical_terms = {"sad", "cry", "depressed", "hopeless", 'am'}
stop_words = stop_words - critical_terms

# Add additional stopwords
additional_stopwords = {'life', 'something', 'anything', 'aand', 'abt', 'ability', 'academic', 'able', 'account', 'advance'}
stop_words.update(additional_stopwords)

# Preprocessing function
def preprocess_text(text):
    """Clean and preprocess text."""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove non-alphabetic characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove underscores
    text = re.sub(r'_+', '', text)
    # Remove excessive repeated characters (e.g., "aaaa" -> "aa")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    # Remove short words (e.g., single characters)
    words = [word for word in words if len(word) > 2]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join cleaned words into a single string
    return ' '.join(words).strip()

# Load resources
base_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(base_dir, 'intents1.json')
model_path = os.path.join(base_dir, 'chatbot_model_with_glove.h5')
tokenizer_path = os.path.join(base_dir, 'tokenizer.pkl')
classes_path = os.path.join(base_dir, 'classes.pkl')

try:
    with open(intents_path, 'r') as f:
        intents = json.load(f)
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model_with_glove.h5')
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

def predict_class(sentence):
    """Predict the intent class of the input sentence."""
    # Preprocess and tokenize the input
    preprocessed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
    sequence_padded = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
    
    # Predict intent probabilities
    res = model.predict(sequence_padded)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Retrieve response based on predicted intent."""
    if not intents_list or float(intents_list[0]['probability']) < 0.25:
        for i in intents_json['intents']:
            if i['tag'] == "fallback":
                return random.choice(i['responses'])
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm here to help, but I didn't understand that."

@app.route('/')
def index():
    """Render the chatbot webpage."""
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    """Handle AJAX requests for chatbot responses."""
    user_message = request.form['message']  # Get user message
    intents_list = predict_class(user_message)  # Predict intents
    response = get_response(intents_list, intents)  # Get response
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
