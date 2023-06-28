from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import re
import string

app = Flask(__name__)

# Load model and tokenizer
model_lstm = tf.keras.models.load_model('lstm_model.h5')
model_cnn = tf.keras.models.load_model('cnn_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sentiment classes
sentiment_classes = ['Negative', 'Neutral', 'Positive']


def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove whitespace
    text = text.strip()
    # Add more preprocessing steps if needed
    text = re.sub(r'www.[^ ]+', '', text)
    text = re.sub(r'http[^ ]+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = re.sub(r'\b\w(1,2)\b', '', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = re.sub(r'([#])|([^a-zA-Z])', ' ', text)
    return text


@app.route('/')
def hello_world():
    return 'Analisis Sentimen ChatGPT'


@app.route('/predict-lstm', methods=['POST'])
def predict_sentiment():
    text = request.json['text']

    # Preprocess text
    text = preprocess_text(text)

    # Tokenize text
    encoded_text = tokenizer.texts_to_sequences([text])
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, padding='post', maxlen=50)

    # Perform prediction
    prediction = model_lstm.predict(encoded_text)[0]
    sentiment_index = tf.argmax(prediction).numpy()
    sentiment = sentiment_classes[sentiment_index]

    return jsonify({'text': text,
                    'sentiment': sentiment,
                    'confidence': float(prediction[sentiment_index])
                    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
