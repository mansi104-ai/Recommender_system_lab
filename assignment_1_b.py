import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

unique_file_path = "final_written.txt"

def read_file_by_word():
    with open('assignment_1_b_lstm.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        return content.split()

def regex_conversion(word):
    # Remove HTML tags 
    word = re.sub(r'<[^>]*>', '', word)
    # Remove punctuation and special text
    word = re.sub(r'[^\w\s]', '', word)
    # Convert to lowercase for normalization 
    word = word.lower().strip()
    return word

# initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# read words
words = read_file_by_word()

processed_words = []

# process each word
for word in words:
    cleaned_word = regex_conversion(word)

    # remove empty tokens and stopwords
    if cleaned_word == "" or cleaned_word in stop_words:
        continue

    stemmed_word = ps.stem(cleaned_word)
    processed_words.append(stemmed_word)

# write final preprocessed collection
with open(unique_file_path, 'w', encoding='utf-8') as file:
    file.write(" ".join(processed_words))