import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# load dataset
df = pd.read_csv('design_interviews.csv')


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    words = word_tokenize(text)  # Tokenize words
    words = [w for w in words if w not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)


# cleaning the text
df['cleaned_response'] = df['Response'].astype(str).apply(clean_text)


vectorizer = CountVectorizer(max_features=1000)  # Limit to top 1000 words
X = vectorizer.fit_transform(df['cleaned_response'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics
lda.fit(X)

words = vectorizer.get_feature_names_out() 
for i, topic in enumerate(lda.components_):
    print(f"\nPattern {i + 1}:")
    print([words[j] for j in topic.argsort()[-3:]])  # Top 3 words in each pattern
