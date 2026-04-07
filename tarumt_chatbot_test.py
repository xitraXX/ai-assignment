import random
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fuzzywuzzy import process 

# --- INITIAL DOWNLOADS & SETUP ---
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Load the AI brain
model = load_model('tarumt_model_programme.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
df = pd.read_csv('tarumt_dataset.csv')
df['User_Message'] = df['User_Message'].astype(str).str.strip()

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

# --- MAIN TERMINAL LOOP ---
print("\n" + "="*50)
print("TARUMT Chatbot is ready! (Type 'quit' to exit)")
print("="*50)

while True:
    # 1. Get User Input
    message = input("\nYou: ").strip()
    
    if message.lower() == "quit":
        print("Goodbye!")
        break

    # 2. Process input (Fuzzy Logic & Prediction)
    smart_message = auto_correct(message)
    p = bow(smart_message) 
    res = model.predict(np.array([p]), verbose=0)[0] 
    max_index = np.argmax(res)
    intent_tag = classes[max_index]
    probability = res[max_index]

    # Optional Debugging
    # print(f"DEBUG: Fixed typos to '{smart_message}'. Intent={intent_tag} ({probability:.2f})")

    # 3. Logic to determine response
    if probability > 0.70:
        if intent_tag == "general_list":
            # Header response from CSV
            response = df[df['Intent'] == 'general_list']['Response'].values[0]
            print(f"TARUMT: {response}")

            # Filter diploma and degree lists
            diploma = df[df['Intent'].str.contains('diploma', na=False)]['Response'].unique()
            degree = df[df['Intent'].str.contains('degree', na=False)]['Response'].unique()

            print("\n--- DIPLOMA ---")
            for d in diploma: print(f"- {d}")
            
            print("\n--- DEGREE ---")
            for d in degree: print(f"- {d}")

            print(f"\nTARUMT: Could you tell me which one are you interested in?")

        else:
            # Standard intent response
            rows = df[df['Intent'].str.strip() == intent_tag.strip()]
            if not rows.empty:
                response_text = rows['Response'].values[0]
                if pd.isna(response_text):
                    print("TARUMT: [Error] The 'Response' cell for this intent is EMPTY in your CSV!")
                else:
                    print(f"TARUMT: {response_text}")
            else:
                print(f"TARUMT: [Error] I can't find '{intent_tag}' in the CSV.")
                
    else:
        # Fallback for low confidence
        try:
            options = df[df['Intent'] == 'fallback']['Response'].values
            print(f"TARUMT: {options[0]}")
        except:
            print("TARUMT: Sorry, I couldn't understand your request. Could you try to specify more?")