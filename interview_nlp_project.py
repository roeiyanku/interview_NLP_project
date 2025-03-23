import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Example interview responses (answers from an interview with designers and developers)
interview_responses = [
    "I think the interface could be much more intuitive, especially for people who don't understand technology.",
    "If there was a help button on the side, it would be much easier to understand how to use the product.",
    "We could add an auto-correction function for the common errors, so users wouldn't have to contact technical support.",
    "I would like the interface to be simpler, maybe with more guidance for users.",
    "An auto-correction function would help reduce the need for technical support."
]


def clean_text(text):
    #turn words to tokens
    words = word_tokenize(text)

    #delete irrelevent words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return words


# cleaning the text
cleaned_responses = []
for response in interview_responses:
    cleaned_responses.extend(clean_text(response))

freq_dist = FreqDist(cleaned_responses)

#Display
print("Patterns found:")
for word,freq in freq_dist.most_common(5):
    print(f"{word}: {freq}")