#########################################################
# Monkey Chatbot
# AI Coursework - Tugra Karakus
#########################################################

# Initialise libraries
import json
import requests
import aiml
from nltk.corpus import stopwords
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from PIL import Image
import numpy as np
import os
import keras
import tkinter as tk
from tkinter import filedialog
from fuzzy import *
import simpful as sf
import re

# Initialise aiml agent
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-answers.xml")

# Initialise NLTK interface and knowledgebase
read_expr = Expression.fromstring
kb = []
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Check knowledgebase integrity
integrity_check = ResolutionProver().prove(None, kb, verbose=False)
if integrity_check:
    print("Error, contradiction found in kb")
    quit()

# Initialise Speech Recogniser
r = sr.Recognizer()

# Initialise similarity dataframe
df = pd.read_csv("QA.csv")

# Welcome user
print("Hi! Feel free to ask any questions about Monkeys! Type 'VOICE' to ask a question with your microphone!")

def get_random_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    joke = response.json()
    print(joke["setup"])
    print(joke["punchline"])

def get_activity():
    url ="https://www.boredapi.com/api/activity"
    response = requests.get(url)
    activity = response.json()  
    print(activity["activity"])

# Image Classifier
def classifier():
    root = tk.Tk()
    root.geometry("1x1+0+0")  
    root.attributes("-topmost", True)  # Make the window appear on top of other windows
    root.lift()  # Bring the window to the front
    file_path = filedialog.askopenfilename(
            parent=root,
            title='Select a file',
            initialdir='/',
            filetypes=(
                ('JPG files', '*.jpg'),
                ('JPEG files', '*.jpeg'),
                ('PNG files', '*.png'),
                ('BMP files', '*.bmp'),
                ('GIF files', '*.gif')
            )
    )
    root.withdraw()
    
    print(file_path)
    model = keras.models.load_model('monkey_model.h5')
    img = Image.open(file_path)
    img = img.convert('RGB') 
    imgArr = np.array(img) 
    imgArr.resize((256,256,3))
    imgArr = np.expand_dims(img.resize((256,256)), axis=0)
    prediction = model.predict(imgArr)
    predicted_class = np.argmax(prediction, axis=1)[0]
    if predicted_class == 0:
        print("Monkey")
    else:
        print("Not Monkey")
        

# Calculate TF_DIF score
def tf_idf(query):
    query = [query]
    v = TfidfVectorizer(stop_words=stopwords.words('english'))
    similarity_index_list = cosine_similarity(v.fit_transform(df['Question']), v.transform(query)).flatten()

    if not all(i == 0 for i in similarity_index_list):
        answer = df.loc[similarity_index_list.argmax(), "Answer"]
        print(answer)
    else:
        print("Sorry, I did not understand your question!")


# Get voice input
def voice_input():
    try:
        with sr.Microphone() as source:
            print("Say Something!")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            voiceInp = r.recognize_google(audio)
            return voiceInp.lower()

    except sr.RequestError as e:
        print("Sorry, I couldn't process your voice query!")
    except sr.UnknownValueError:
        print("Sorry, I was unable to recognise your speech!")


# Main loop
def main():
    while True:
        # Get user input
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

        if userInput == "VOICE":
            userInput = voice_input()
            if userInput is None:
                break
            else:
                print(">", userInput)

        # Preprocess the input and determine response agent
        responseAgent = "aiml"

        # Activate selected response agent
        if responseAgent == "aiml":
            answer = kern.respond(userInput)

            # Postprocess answer for commands
            if answer[0] == '#':
                params = answer[1:].split('$')
                cmd = int(params[0])

                # Goodbye Message
                if cmd == 0:
                    print(params[1])
                    break
        
                # Image Classifier
                elif cmd == 1:
                    classifier()


                # Random joke
                elif cmd == 2:
                    get_random_joke()


                # Random activity   
                elif cmd == 3:
                    get_activity()


                # Rate monkey
                elif cmd == 4:
                    fuzzy_logic()

                                         
                # Add to kb
                elif cmd == 97:
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    kb.append(expr)

                    integrity_check = ResolutionProver().prove(None, kb, verbose=False)

                    if not integrity_check:
                        print("Okay! I'll remember that", object, "is", subject)
                    else:
                        print("Sorry! This contradicts with what I know!")
                        kb.pop()


                # Check kb
                elif cmd == 98:
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')

                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Correct.')
                    else:
                        kb.append(expr)
                        integrity_check = ResolutionProver().prove(None, kb, verbose=False)

                        if integrity_check:
                            print("Incorrect")
                        else:
                            print("Sorry, I don't know")
                        kb.pop()


                # Unresolved Input
                elif cmd == 99:
                    tf_idf(userInput)
            else:
                print(answer)


main()