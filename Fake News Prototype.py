import tkinter as tk
import traceback
from tkinter import ttk
import os
from tkinter import filedialog
import pandas as pd
import spacy
import re
import pickle
from sklearn import ensemble



nlppath = os.getcwd()+"\\nlp.pkl"
nlp = pickle.load(open(nlppath, 'rb'))

def reduce_newlines(text):
    return re.sub(r"\n+", " ", text.replace("\n", " "))

VC = re.compile('[aeiou]+[^aeiou]+')
def count_syllables(word):
    return len(VC.findall(word))

def compute_FKG(num_sents,num_words,num_syllables):
  score = (0.39* (float(num_words)/float(num_sents)))+(11.8*(float(num_syllables)/float(num_words)))-15.59
  return score

def compute_FRE(num_sents,num_words,num_syllables):
  score = 206.835 - 1.015 * (float(num_words) / float(num_sents)) - 84.6 * (float(num_syllables) / float(num_words))
  return score

def compute_CLI(num_sents,num_words,num_syllables,num_chars):
  score = (0.0588*float(float(num_chars/num_words)*100))-(0.296*float(float(num_sents/num_words)*100))-15.8
  return score

def compute_ARI(num_sents,num_words,num_syllables,num_chars):
  score = (4.7* (float(num_chars)/float(num_words)))+(0.5*(float(num_words)/float(num_sents)))-21.43
  return score

import math
def compute_SMOG(num_sents,num_words,num_syllables,num_chars,diff_words):
  score = 1.043*math.sqrt(diff_words*(30/num_sents))+3.1291
  return score

def compute_GFI(num_sents,num_words,num_syllables,num_chars,diff_words):
  score = (0.4*(num_words/num_sents)) + (0.0496*(diff_words/num_words))
  return score

def paragraphs(document):
    start = 0
    for token in document:
        if token.is_space and token.text.count("\n") > 1:
            yield document[start:token.i]
            start = token.i
    yield document[start:]



root = tk.Tk()
root.geometry("800x600")
root.title("Fake News Detector")

# Function to handle text file selection
def select_text_file():
    global text_file_path
    text_file_path = filedialog.askopenfilename(
        title="Select Text File",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
    )
    text_file_label.config(text="Selected Text File: " + text_file_path)

# Function to handle CSV file selection
def select_csv_file():
    global csv_file_path
    csv_file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    csv_file_label.config(text="Selected CSV File: " + csv_file_path)


# Function to read and display files
def read_files():
    global result_display
    global table_display
    try:
      # Read contents of text file and display
      with open(text_file_path, 'r', encoding = "utf8") as text_file:
          contents = text_file.read()

          # text_label = tk.Label(root, text="Text File Contents:")
          # text_label.pack()

          # text_display = tk.Text(root)
          # text_display.insert(tk.END, text_content)
          # text_display.pack()

      # # Read contents of CSV file and display
      # csv_label = tk.Label(root, text="CSV File Contents:")
      # csv_label.pack()

      corpus = pd.read_csv(csv_file_path)

      corpus["Text"] = contents

      to_select = ['WPS', 'BigWords', 'Dic', 'Linguistic', 'pronoun', 'ppron', 'you',
          'shehe', 'number', 'emotion', 'space', 'Conversation', 'netspeak',
          'QMark', 'Exclam', 'OtherP', 'FKG', 'FRE', 'CLI', 'ARI', 'GFI', 'SMOG']

      tokenize = corpus.copy()

      tokenize["Text"] = tokenize["Text"].apply(reduce_newlines)

      #Get the Other Components

      # rerun for ucorpus and ocorpus
      FKG = []
      FRE = []
      CLI = []
      ARI = []
      GFI = []
      SMOG = []
      Characters = []
      Sentences = []

      for content in tokenize["Text"]:
          text_nlp = nlp(content)
          sents = list(text_nlp.sents)
          words = list(w for w in text_nlp)
          num_chars = sum(int(len(words.text)) for words in text_nlp)
          num_sents = len(sents)
          num_words = len(words)
          num_syllables = sum(count_syllables(w.text) for w in text_nlp)
          diff_words = len([w.text for w in text_nlp if count_syllables(w.text)>3])
          Characters.append(num_chars)
          Sentences.append(num_sents)
          FKG.append(compute_FKG(num_sents,num_words,num_syllables))
          FRE.append(compute_FRE(num_sents,num_words,num_syllables))
          ARI.append(compute_ARI(num_sents,num_words,num_syllables,num_chars))
          CLI.append(compute_CLI(num_sents,num_words,num_syllables,num_chars))
          GFI.append(compute_GFI(num_sents,num_words,num_syllables,num_chars,diff_words))
          SMOG.append(compute_SMOG(num_sents,num_words,num_syllables,num_chars,diff_words))

      tokenize["FKG"] = FKG
      tokenize["FRE"] = FRE
      tokenize["CLI"] = CLI
      tokenize["ARI"] = ARI
      tokenize["GFI"] = GFI
      tokenize["SMOG"] = SMOG
      tokenize["Characters"] = Characters
      tokenize["Sentences"] = Sentences

      to_pred = tokenize[to_select]

      filename = os.getcwd()+"\\finalmodel.pkl"
      loaded_model = pickle.load(open(filename, 'rb'))
      result = loaded_model.predict(to_pred)
      result_prob = loaded_model.predict_proba(to_pred)
      result_display.config(text="Result: "+("Genuine" if result[0]==1 else "Fake"))
      result_prob_display.config(text="Confidence Genuine:" + str(result_prob[0][1]) +" Fake:" + str(result_prob[0][0]))
    
      data = [['WPS', '21.43', '23.14',str(tokenize['WPS'][0])],
        ['BigWords', '24.47', '29.84',str(tokenize['BigWords'][0])],
        ['Dic', '65.52', '65.45',str(tokenize['Dic'][0])],
        ['Linguistic', '56.04', '54.13',str(tokenize['Linguistic'][0])],
        ['Pronoun', '7.79', '5.47',str(tokenize['pronoun'][0])],
        ['ppron', '4.44', '2.68',str(tokenize['ppron'][0])],
        ['you', '0', '0',str(tokenize['you'][0])],
        ['shehe', '1.56', '0.65',str(tokenize['shehe'][0])],
        ['number', '1.41', '2.94',str(tokenize['number'][0])],
        ['emotion', '0.83', '0.33',str(tokenize['emotion'][0])],
        ['space', '4.35', '5.73',str(tokenize['space'][0])],
        ['Conversation', '0.11', '0',str(tokenize['Conversation'][0])],
        ['Netspeak', '0', '0',str(tokenize['netspeak'][0])],
        ['QMark', '0', '0',str(tokenize['QMark'][0])],
        ['Exclam', '0', '0',str(tokenize['Exclam'][0])],
        ['OtherP', '3.28', '5.66',str(tokenize['OtherP'][0])],
        ['FKG', '6.143284', '10.273869',str(tokenize['FKG'][0])],
        ['FRE', '88.80451', '74.256354',str(tokenize['FRE'][0])],
        ['CLI', '8.065278', '9.656067',str(tokenize['CLI'][0])],
        ['ARI', '9.400193', '14.619515',str(tokenize['ARI'][0])],
        ['GFI', '8.45088', '11.843351',str(tokenize['GFI'][0])],
        ['SMOG', '7.610547', '9.3871',str(tokenize['SMOG'][0])]]
      
      for i, row in enumerate(data):
        table_display.insert("", i, text=str(i), values=(row[0], row[1], row[2],row[3]))
    
    except Exception as e:
      stack_trace = traceback.format_exc()
      print(stack_trace)
      result_display.config(text="Error")


# Add text file selection button
text_file_button = tk.Button(root, text="Select Text File", command=select_text_file)
text_file_button.pack()

# Add CSV file selection button
csv_file_button = tk.Button(root, text="Select CSV File", command=select_csv_file)
csv_file_button.pack()

# Add button to read and display files
read_button = tk.Button(root, text="Predict", command=read_files)
read_button.pack()

# Add labels to display selected files
text_file_label = tk.Label(root, text="")
text_file_label.pack()

csv_file_label = tk.Label(root, text="")
csv_file_label.pack()

result_display = tk.Label(root,text="")
result_display.pack()

result_prob_display = tk.Label(root, text="")
result_prob_display.pack()

table_display = ttk.Treeview(root, columns=["Features", "Fake", "Genuine","Selected File"], show="headings")
table_display.pack(fill="both",expand=True)
table_display.heading("Features",text="Features")
table_display.heading("Fake",text="Fake")
table_display.heading("Genuine",text="Genuine")
table_display.heading("Selected File",text="Selected File")

root.mainloop()