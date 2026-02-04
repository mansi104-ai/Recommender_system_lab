from collections import Counter
import re
import matplotlib.pyplot as plt

def get_word_frequencies(filename):
  with open(filename, 'r', encoding='utf-8') as file:
    #Read file and convert to lowercase
    text = file.read().lower()
    #Use regex to remove punctuations and split into words
    words = re.findall(r'\b\w+\b',text)

    #Create a Map of the word frequencies
    word_counts = Counter(words)

  return word_counts.most_common(20)

filename = "assignment_1_b_lstm.txt"

word_counts = get_word_frequencies(filename)

#seperate the words and frequencies
words = [word for word, freq in word_counts]
freq = [freq for word,freq in word_counts]

#Plot
plt.figure()
plt.bar(words, freq)
plt.xlabel("Words")
plt.ylabel("Frequencies")
plt.title("Top 20 Words with frequency graph")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()