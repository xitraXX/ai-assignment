import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
import pickle

# Ensure data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# 1. Load CSV
df = pd.read_csv('tarumt_dataset.csv')

# --- ignore any row where the Intent is missing ---
df = df.dropna(subset=['Intent']) 

# --- ignore any row with the ignore Intent ---
df = df[~df['Intent'].str.contains('ignore', na=False)]

# --- ADD THIS LINE to remove spaces from the Excel cells ---
df['User_Message'] = df['User_Message'].astype(str).str.strip()
# --------------------------

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', '(', ')']

for index, row in df.iterrows():
    # Convert message to string and tokenize
    token_list = nltk.word_tokenize(str(row['User_Message']))
    words.extend(token_list)
    documents.append((token_list, row['Intent']))
    if row['Intent'] not in classes:
        classes.append(row['Intent'])

# 2. Preprocessing
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 3. Create Training Data
train_x = []
train_y = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    train_x.append(bag)
    train_y.append(output_row)

# Convert to proper Numpy arrays for TensorFlow
train_x = np.array(train_x)
train_y = np.array(train_y)

# 4. Build Neural Network
model = Sequential([
    # Input layer matches the size of your vocabulary
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5), # Prevents the AI from just "memorizing"
    Dense(64, activation='relu'),
    Dropout(0.3),
    # Output layer matches the number of intents (classes)
    Dense(len(train_y[0]), activation='softmax') 
])

# Use a slightly lower learning rate if accuracy doesn't hit 1.0
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=300, batch_size=8, verbose=1)

model.save('tarumt_model_programme.h5')
print("\n--- Model Trained Successfully! ---")
print(f"Vocabulary size: {len(words)}")
print(f"Number of Intents: {len(classes)}")


# ==========================================
# 5. AUTOMATED ACCURACY TEST SECTION
# ==========================================

# Step A: Define the Helper Functions FIRST
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Step B: Define the Testing Function
def test_all_rows(df):
    correct = 0
    total = len(df)
    
    print("\n--- Running Automated Accuracy Test ---")
    for index, row in df.iterrows():
        user_input = row['User_Message']
        expected_intent = row['Intent']
        
        results = predict_class(user_input, model)
        
        # Make sure the bot actually guessed something
        if len(results) > 0:
            predicted_intent = results[0]['intent']
            if predicted_intent == expected_intent:
                correct += 1
                
    accuracy = (correct / total) * 100
    print(f"Total Rows Tested: {total}")
    print(f"Accuracy on Training Data: {accuracy:.2f}%\n")

# Step C: Run the test!
test_all_rows(df)