import streamlit as st
# st.caching.clear_cache()

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
    for story, question,ans in all_data:
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
st.title("Text-Based Question Answering Chatbot")
st.write("### Enter a story and a question to get an answer:")

# User Inputs
story_input = st.text_area("Enter the story:", "John left the kitchen. Sandra dropped the football in the garden.")
question_input = st.text_input("Enter your question:", "Is the football in the garden?")

if st.button("Get Answer"):
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
        st.subheader("Predicted Answer:")
        st.write(predicted_word.capitalize() if predicted_word else "Error in prediction.")
        st.write(f"**Confidence Score:** {prediction[0][val_max]:.2f}")
    else:
        st.warning("Please provide both a story and a question!")

# Footer
st.markdown("---")
st.markdown("**Note:** This chatbot uses a pre-trained neural network model for answering simple story-based questions. Ensure the model file `chatbot2_120_epochs.h5` is in the same directory as this script.")
