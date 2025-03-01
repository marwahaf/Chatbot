import json
import os
import random
import string
from pathlib import Path

import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download necessary NLTK data
nltk.download("wordnet")
nltk.download("punkt_tab")

# Set file path
data_file = Path(os.path.dirname(os.path.abspath(__file__)), "intents.json")


# Load the dataset
def load_dataset(file_path):
    """Loads and parses the dataset from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


data = load_dataset(data_file)

# Initialize necessary variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
data_X = []
data_y = []


# Preprocess the dataset
def preprocess_data(data):
    """Preprocesses the data into tokenized patterns and labels."""
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            data_X.append(pattern)
            data_y.append(intent["tag"])

        if intent["tag"] not in classes:
            classes.append(intent["tag"])


preprocess_data(data)

# Lemmatize and clean words
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in string.punctuation
]
words = sorted(set(words))
classes = sorted(set(classes))


# Convert sentences to Bag of Words
def create_bag_of_words(data_X, words, data_y, classes):
    """Converts sentences to a Bag of Words model."""
    training_data = []
    output_empty = [0] * len(classes)

    for idx, doc in enumerate(data_X):
        bow = [1 if word in lemmatizer.lemmatize(doc.lower()) else 0 for word in words]
        output_row = list(output_empty)
        output_row[classes.index(data_y[idx])] = 1
        training_data.append([bow, output_row])

    return np.array(training_data, dtype=object)


training_data = create_bag_of_words(data_X, words, data_y, classes)
np.random.shuffle(training_data)

train_X = np.array([item[0] for item in training_data])
train_y = np.array([item[1] for item in training_data])


# Build the model
def build_model(input_shape, output_shape):
    """Builds a neural network model."""
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation="softmax"))

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(
        loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"]
    )

    return model


model = build_model(len(train_X[0]), len(train_y[0]))
model.summary()


# Train the model
def train_model(model, train_X, train_y, epochs=150):
    """Trains the neural network model."""
    model.fit(train_X, train_y, epochs=epochs, verbose=1)


train_model(model, train_X, train_y)


# Preprocessing functions
def clean_sentence(sentence):
    """Cleans the input sentence for prediction."""
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bag_of_words(sentence, words):
    """Converts a sentence into a bag of words vector."""
    sentence_words = clean_sentence(sentence)
    return [1 if word in sentence_words else 0 for word in words]


# Prediction function
def predict_class(sentence, model, words, classes, threshold=0.5):
    """Predicts the class of the input sentence."""
    bow_vector = bag_of_words(sentence, words)
    prediction = model.predict(np.array([bow_vector]))[0]

    predicted_classes = [
        [i, prob] for i, prob in enumerate(prediction) if prob > threshold
    ]
    predicted_classes.sort(key=lambda x: x[1], reverse=True)

    return [classes[i[0]] for i in predicted_classes]


# Get response based on prediction
def get_response(predicted_classes, intents_json):
    """Maps predicted class to a response from the intents."""
    if not predicted_classes:
        return "Sorry, I didn't understand that."

    tag = predicted_classes[0]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't find an appropriate response."


# Main interaction loop
def start_chatbot():
    """Starts the chatbot interaction loop."""
    print("Press '0' to exit the chatbot.")
    while True:
        message = input("You: ")
        if message == "0":
            print("Goodbye!")
            break
        predicted_classes = predict_class(message, model, words, classes)
        response = get_response(predicted_classes, data)
        print(f"Bot: {response}")


start_chatbot()
