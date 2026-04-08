import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fuzzywuzzy import process 
import streamlit as st

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="TARUMT Chatbot", page_icon="🎓")
st.title("TARUMT Chatbot 🎓")
st.caption("Ask me anything regarding TARUMT!")

# --- SIDEBAR: CHATBOT SELECTION ---
with st.sidebar:
    st.title("Settings")
    
    # Selection for the 3 different models
    bot_selection = st.radio(
        "Choose a Model:",
        ("Course", "Faq", "Financial Aid") # Change names as needed
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared!"}
        ]
        st.rerun()

    st.info(f"Currently active: **{bot_selection}**")

    st.divider()
    st.write("**Model Info:**")
    st.info("Using TensorFlow 2.x + NLTK")
# --------------------------------

# --- CACHE THE AI SO IT DOESN'T RELOAD EVERY TIME YOU TYPE ---
@st.cache_resource
def load_ai_brain(bot_selection):
    lemmatizer = WordNetLemmatizer()

    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # 3-Way Switch Logic to load the correct files based on sidebar selection
    if bot_selection == "Course":
        model = load_model('course/tarumt_chatbot_course.h5')
        words = pickle.load(open('course/words_course.pkl', 'rb'))
        classes = pickle.load(open('course/classes_course.pkl', 'rb'))
        df = pd.read_csv('course/course.csv')
        
    elif bot_selection == "FaQ":
        model = load_model('faq/tarumt_chatbot_faq.h5')
        words = pickle.load(open('faq/words_faq.pkl', 'rb'))
        classes = pickle.load(open('faq/classes_faq.pkl', 'rb'))
        df = pd.read_csv('faq/faq.csv')
        
    else: 
        model = load_model('financial_aid/tarumt_chatbot_financial_aid.h5')
        words = pickle.load(open('financial_aid/words_financial_aid.pkl', 'rb'))
        classes = pickle.load(open('financial_aid/classes_financial_aid.pkl', 'rb'))
        df = pd.read_csv('financial_aid/financial_aid.csv')

    df['User_Message'] = df['User_Message'].astype(str).str.strip()    
    return lemmatizer, model, words, classes, df

# Load variables from cache
lemmatizer, model, words, classes, df = load_ai_brain(bot_selection)

# --- HELPER FUNCTIONS ---
def clean_up(sentence):
    s_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in s_words]

def bow(sentence):
    s_words = clean_up(sentence)
    bag = [0] * len(words)
    for s in s_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array(bag)

def auto_correct(user_input):
    words_in_message = user_input.split()
    corrected_message = []
    for word in words_in_message:
        match, score = process.extractOne(word, words)
        if score > 80: 
            corrected_message.append(match)
        else:
            corrected_message.append(word)
    return " ".join(corrected_message)

# --- STREAMLIT SESSION STATE (CHAT HISTORY) ---
# This checks if this is a new conversation. If yes, it creates an empty list to store messages.
# --- SESSION STATE (Switching Logic) ---
# If the user changes the bot in the sidebar, clear the messages automatically
if "current_bot" not in st.session_state:
    st.session_state.current_bot = bot_selection
    st.session_state.messages = []

if st.session_state.current_bot != bot_selection:
    st.session_state.current_bot = bot_selection
    st.session_state.messages = [] # Reset chat when switching bots

# Default greeting if empty
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": f"Hello! I am the {bot_selection}. How can I help you today?"})

# Display all previous messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- UI DISPLAY ---
st.title(f"{bot_selection}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- MAIN CHAT UI ---
# st.chat_input creates the text box at the bottom of the screen
if user_input := st.chat_input("Type your question here..."):
    
    # 1. Show the user's message on screen & save it to history
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Process the message (Fuzzy Logic & AI Prediction)
    smart_message = auto_correct(user_input)
    p = bow(smart_message) 
    res = model.predict(np.array([p]), verbose=0)[0] 
    max_index = np.argmax(res)
    intent_tag = classes[max_index]
    probability = res[max_index]

    # Optional: Show debug info on the Streamlit screen temporarily
    st.write(f"*DEBUG: Auto-corrected to '{smart_message}'. Intent={intent_tag} | Confidence={probability:.2f}*")

    # 3. Generate the Bot's Response String
    bot_response = ""
    
    if probability > 0.70:
        if bot_selection == "Course" and intent_tag == "general_list":
            response = df[df['Intent'] == 'general_list']['Response'].values[0]
            bot_response += f"{response}\n\n"

            diploma = df[df['Intent'].str.contains('general', na=False)]['Response'].unique()
            degree = df[df['Intent'].str.contains('general', na=False)]['Response'].unique()

            # Using Markdown for nice formatting
            bot_response += "**--- DIPLOMA ---**\n- " + "\n- ".join(diploma) + "\n\n"
            bot_response += "**--- DEGREE ---**\n- " + "\n- ".join(degree) + "\n\n"
            bot_response += "Could you tell me which one are you interested in?"

        else:
            rows = df[df['Intent'].str.strip() == intent_tag.strip()]
            if not rows.empty:
                response_text = rows['Response'].values[0]
                if pd.isna(response_text):
                    bot_response = "[Error] The 'Response' cell for this intent is EMPTY in your CSV!"
                else:
                    bot_response = response_text
            else:
                bot_response = f"I found the intent {intent_tag}, but the response is missing in the dataset."        
    else:
        # try:
        #     options = df[df['Intent'] == 'fallback']['Response'].values
        #     bot_response = options[0]
        # except:
            bot_response = "Sorry, I couldn't understand your request. Could you try to specify more?"

    # 4. Show the AI's response on screen & save it to history
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})