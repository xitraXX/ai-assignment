import random
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fuzzywuzzy import process 

# Chatbot Name 
CHATBOT_NAME = "TARUMT"

# Standard Setup
lemmatizer = WordNetLemmatizer()
model = load_model('tarumt_model_programme.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
df = pd.read_csv('tarumt_dataset.csv')

# --- ADDED: Clean the CSV User_Messages just in case there are hidden spaces ---
df['User_Message'] = df['User_Message'].astype(str).str.strip()
# --------------------------------------------

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

# --- ADDED: THE FUZZY AUTO-CORRECT FUNCTION ---
def auto_correct(user_input):
    words_in_message = user_input.split()
    corrected_message = []
    
    for word in words_in_message:
        # Check if the typed word is close to an official word in words.pkl
        match, score = process.extractOne(word, words)
        
        # 80 is the "closeness" score. You can adjust this if it's too strict or too loose!
        if score > 80: 
            corrected_message.append(match)
        else:
            corrected_message.append(word)
            
    return " ".join(corrected_message)
# ----------------------------------------------

print("\nTARUMT Chatbot is ready! (Type 'quit' to exit)")
print("----------------------------------------------")

while True:
    # --- CHANGED: Added .strip() to remove accidental spaces the user might type ---
    message = input("You: ").strip() 
    if message.lower() == "quit": break

    # --- ADDED: Intercept the message and fix typos (e.g., 'acc' -> 'accounting') ---
    smart_message = auto_correct(message)
    
    # Optional: You can uncomment the line below to SEE the typo getting fixed in real-time!
    print(f"DEBUG (Fuzzy Tracker): '{message}' was auto-corrected to '{smart_message}'")
    # -----------------------------------------------------------------------------------

    # 1. Predict Intent
    # --- CHANGED: Pass the 'smart_message' into the Neural Network instead of the raw message ---
    p = bow(smart_message) 
    res = model.predict(np.array([p]), verbose=0)[0] # verbose=0 stops TF from printing messy logs
    max_index = np.argmax(res)
    intent_tag = classes[max_index]
    probability = res[max_index]

    # --- PROBABILITY RESULT DEBUGGER (Temporary) ---
    print(f"DEBUG: Intent={intent_tag} | Confidence={probability:.2f}")

    # Check if AI is confident
    if probability > 0.70:
        if intent_tag == "general_list":
            # 1. Fetch the actual response text from the CSV for this intent
            response = df[df['Intent'] == 'general_list']['Response'].values[0]
            
            print(f"{CHATBOT_NAME}: {response}")

            # 2. Now run your logic to list the actual courses
            diploma = df[df['Intent'].str.contains('diploma', na=False)]['Response'].unique()
            degree = df[df['Intent'].str.contains('degree', na=False)]['Response'].unique()

            print("\n--- DIPLOMA ---")
            print("- " + "\n- ".join(diploma)) 
            
            print("\n--- DEGREE ---")
            print("- " + "\n- ".join(degree))

            print(f"{CHATBOT_NAME}: Could you tell me which one are you interested in?")

        # 2. Otherwise, just give the standard response from the CSV
        else:
            rows = df[df['Intent'].str.strip() == intent_tag.strip()]
            
            if not rows.empty:
                response_text = rows['Response'].values[0]
                if pd.isna(response_text):
                    print(f"TARUMT: [Error] The 'Response' cell for '{intent_tag}' is EMPTY in your CSV!")
                else:
                    print(f"TARUMT: {response_text}")
            else:
                print(f"TARUMT: [Error] I can't find '{intent_tag}' in the CSV. Did you rename it in Excel?")
                
    # --- IF PROBABILITY IS BELOW SET (UNKNOWN) ---
    else:
        try:
            # Grabs the text from the CSV so you can edit it without opening Python!
            options = df[df['Intent'] == 'fallback']['Response'].values
            print(f"{CHATBOT_NAME}: {options[0]}")
        except:
            # A safety net just in case you forget to add 'fallback' to Excel
            print(f"{CHATBOT_NAME}: Sorry, I couldn't understand your request. Could you try to specify more?")