import re
# with open('assignment_1.txt','r',encoding='utf-8') as file:
#   content = file.read()

# print(content)
def read_file_by_word():
    with open('assignment_1.txt', 'r', encoding='utf-8') as file:
        content = file.read()

        # this creates a list and we cannot lower a string , so either we can 
        # lowe while spiltting or assign eac word to itself by using lower function
        return content.split()

def regex_conversion(word):
    # Remove HTML tags 
    word = re.sub(r'<[^>]*>', '', word)
    # Remove punctuation and special text
    word = re.sub(r'[^\w\s]', '', word)
    # Replace multiple spaces with a single space
    word = re.sub(r'\s+', '', word).strip()
    # Convert to lowercase for normalization 
    word = word.lower()
    return word

# def read_occurences(term):
#     andCnt = 0
#     toCnt = 0
#     arjCount = 0
#
#     if term == "and":
#         andCnt += 1
#     if term == "to":
#         toCnt += 1
#     if term == "arjun":
#         arjCount += 1
#
#     return andCnt, toCnt, arjCount

# initialize counters
andCnt = 0
toCnt = 0
arjCount = 0

# read words from file
words = read_file_by_word()

# process each word
for word in words:
    cleaned_word = regex_conversion(word)

    if cleaned_word == "and":
        andCnt += 1
    elif cleaned_word == "to":
        toCnt += 1
    elif cleaned_word == "arjun":
        arjCount += 1

    print(cleaned_word)

# final counts
print(f'andCnt = {andCnt}', f'toCnt = {toCnt}',f'arjCnt = {arjCount}')