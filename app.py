from flask import Flask, request, jsonify, render_template
import mysql.connector
from mysql.connector import Error

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

def is_data_exists(cursor, text, sentiment, confidence):
    # Check if the data with the given text exists in the table
    query = f"SELECT COUNT(*) FROM predict_result_data WHERE text = %s AND sentiment = %s AND confidence = %s"
    cursor.execute(query, (text,sentiment, confidence))
    result = cursor.fetchone()
    return result[0] > 0


def save_to_database(text, sentiment, confidence, model_name):
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            database='sentimen-chatgpt',
            user='root',
            password='emah1224'
        )
        if conn.is_connected():
            cursor = conn.cursor()

            # Determine the table name based on the model_name
            table_name = f'predict_result_{model_name.lower()}'

            # Create a table if it doesn't exist
            cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                              (id INT AUTO_INCREMENT PRIMARY KEY,
                               text TEXT,
                               sentiment TEXT,
                               confidence FLOAT)''')

             # Check if the data with the same text already exists
            if is_data_exists(cursor, text, sentiment, confidence):
                print(f"Data with text : '{text}' and {sentiment}' and '{confidence}' already exists in the database. Skipping insertion.")
            else:
                # Insert the data into the table
                query2 = f"INSERT IGNORE INTO predict_result_data (text, sentiment, confidence) VALUES (%s, %s, %s)"
                values = (text, sentiment, confidence)
                cursor.execute(query2, values)

                query = f"INSERT IGNORE INTO {table_name} (text, sentiment, confidence) VALUES (%s, %s, %s)"
                values = (text, sentiment, confidence)
                cursor.execute(query, values)

                conn.commit()

            # Commit the changes and close the connection

            cursor.close()
            conn.close()

    except Error as e:
        print('Error connecting to MySQL:', e)



def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Add more preprocessing steps if needed
    text = re.sub(r'www.[^ ]+', '', text)
    text = re.sub(r'http[^ ]+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = re.sub(r'\b\w(1,2)\b', '', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = re.sub(r'([#])|([^a-zA-Z])', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Remove whitespace
    text = text.strip()
    
    return text


@app.route('/')
def index():
    return 'Test Analisis Sentimen ChatGPT'
    
@app.route('/api')
def api():
    return 'Test Analisis Sentimen ChatGPT'


@app.route('/api/predict-lstm', methods=['POST'])
def lstm_model():
    text = request.json['text']

    # Preprocess text
    text = preprocess_text(text)

    # Tokenize text
    encoded_text = tokenizer.texts_to_sequences([text])
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, padding='post', maxlen=50)

    # Perform prediction using the LSTM model
    lstm_prediction = model_lstm.predict(encoded_text)[0]
    lstm_sentiment_index = tf.argmax(lstm_prediction).numpy()
    lstm_sentiment = sentiment_classes[lstm_sentiment_index]

    # Save data to the database with the model_name 'lstm'
    save_to_database(text, lstm_sentiment, float(lstm_prediction[lstm_sentiment_index]), 'lstm')

    return jsonify({'text': text,
                    'sentiment': lstm_sentiment,
                    'confidence': float(lstm_prediction[lstm_sentiment_index])
                    })


@app.route('/api/predict-cnn', methods=['POST'])
def cnn_model():
    data = request.json
    text = data['text']

    # Preprocess text
    text = preprocess_text(text)

    # Tokenize text
    encoded_text = tokenizer.texts_to_sequences([text])
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, padding='post', maxlen=50)

    # Perform prediction using the CNN model
    cnn_prediction = model_cnn.predict(encoded_text)[0]
    cnn_sentiment_index = tf.argmax(cnn_prediction).numpy()
    cnn_sentiment = sentiment_classes[cnn_sentiment_index]

    # Save data to the database with the model_name 'cnn'
    save_to_database(text, cnn_sentiment, float(cnn_prediction[cnn_sentiment_index]), 'cnn')

    return jsonify({'text': text,
                    'sentiment': cnn_sentiment,
                    'confidence': float(cnn_prediction[cnn_sentiment_index])
                    })
