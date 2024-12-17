import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------------- Load Pre-trained Model and Tokenizer ----------------
@st.cache_resource
def load_model_and_data():
    model = load_model('chatbot2_120_epochs.h5', compile=False)

    with open("train_qa.txt", "rb") as fp:
        train_data = pickle.load(fp)

    with open("test_qa.txt", "rb") as fp:
        test_data = pickle.load(fp)

    # Create vocab
    vocab = set()
    all_data = train_data + test_data
    for story, question, ans in all_data:
        vocab = vocab.union(set(story))
        vocab = vocab.union(set(question))

    vocab.add('no')
    vocab.add('yes')

    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)

    max_story_len = max([len(data[0]) for data in all_data])
    max_question_len = max([len(data[1]) for data in all_data])

    return model, tokenizer, max_story_len, max_question_len

# Load the model and tokenizer
model, tokenizer, max_story_len, max_question_len = load_model_and_data()

# ---------------- Utility Function for Vectorization ----------------
def vectorize_input(story, question, tokenizer, max_story_len, max_question_len):
    story_seq = [tokenizer.word_index.get(word.lower(), 0) for word in story.split()]
    question_seq = [tokenizer.word_index.get(word.lower(), 0) for word in question.split()]

    story_pad = pad_sequences([story_seq], maxlen=max_story_len)
    question_pad = pad_sequences([question_seq], maxlen=max_question_len)
    return story_pad, question_pad

# ---------------- Streamlit UI ----------------

# Custom CSS Styling
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .subtitle {
            font-size: 24px;
            font-weight: 500;
            color: #2196F3;
        }
        .text-input, .text-area {
            font-size: 18px;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
            border: 2px solid #2196F3;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
        }
        .button:hover {
            background-color: #45a049;
        }
        .output {
            font-size: 20px;
            font-weight: 500;
            color: #f44336;
        }
        .footer {
            font-size: 16px;
            color: #757575;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Text-Based Question Answering Chatbot</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Enter a story and a question to get an answer:</div>', unsafe_allow_html=True)

# User Inputs
story_input = st.text_area("Enter the story:", "John left the kitchen. Sandra dropped the football in the garden.", key="story", height=150, help="Enter a short story.")
question_input = st.text_input("Enter your question:", "Is the football in the garden?", key="question", help="Enter a question related to the story.")

# Custom Button Styling
if st.button("Get Answer", key="predict", help="Click to get the answer to your question", use_container_width=True):
    if story_input and question_input:
        # Vectorize Inputs
        story_pad, question_pad = vectorize_input(story_input, question_input, tokenizer, max_story_len, max_question_len)

        # Make Prediction
        prediction = model.predict([story_pad, question_pad])
        val_max = np.argmax(prediction[0])

        # Find the Predicted Word
        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == val_max:
                predicted_word = word
                break

        # Display Results
        if predicted_word:
            st.markdown(f'<div class="output">Predicted Answer: {predicted_word.capitalize()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output">Confidence Score: {prediction[0][val_max]:.2f}</div>', unsafe_allow_html=True)
        else:
            st.warning("Error in prediction. Please check your input.")

    else:
        st.warning("Please provide both a story and a question!")

# Footer
st.markdown('<div class="footer">**Note:** This chatbot uses a pre-trained neural network model for answering simple story-based questions. Ensure the model file `chatbot2_120_epochs.h5` is in the same directory as this script.</div>', unsafe_allow_html=True)
