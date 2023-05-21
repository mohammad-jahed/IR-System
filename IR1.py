import pandas as pd

# pd.read_csv('dataset/queries.csv')
import nltk as nk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def read_df(path):
    df = pd.read_csv(path)
    return df


def tokenize_df(data_frame):
    return data_frame['text'].apply(lambda x: word_tokenize(x))


def clean_data(tokenized_data):
    """
    Remove stop words and use ProtalStemmer then WordNetLemmatizer
    :param tokenized_data: the dataframe after tokenizing
    :return: cleared data
    """
    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    wn = WordNetLemmatizer()

    clear_tokens = tokenized_data.apply(
        lambda x: [wn.lemmatize(ps.stem(word)) for word in x if word.lower() not in stop_words])
    return clear_tokens


def tfidf_vector(processed_data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_data)
    feature_names = vectorizer.get_feature_names_out()
    return X, feature_names


def create_inverse_index(preprocessed_docs):
    inverse_index = {}
    for doc_idx, doc in enumerate(preprocessed_docs):
        terms = doc.split()
        for term in terms:
            if term not in inverse_index:
                inverse_index[term] = [doc_idx]
            else:
                inverse_index[term].append(doc_idx)
# 

# Removing stop words from tokenized title

# In[16]:


# stop_words = set(stopwords.words('english'))
# tokenizedTextWithoutStopWords = tokenizedText.apply(lambda x: [word for word in x if word.lower() not in stop_words])
# print(tokenizedTextWithoutStopWords)


# Removing prefixes and sufexes from tokenized worlds 

# In[18]:


# ps = PorterStemmer()
# tokenizedTextWithoutStopWordsAndStemmed = tokenizedTextWithoutStopWords.apply(lambda x: [ps.stem(word) for word in x])
# print(tokenizedTextWithoutStopWordsAndStemmed)


# Lemmatized the tokenized text
# 

# In[22]:


# from nltk.stem import WordNetLemmatizer
# tokenizedTextWithoutStopWordsAndStemmedAndLemmatized = tokenizedTextWithoutStopWordsAndStemmed.apply(lambda x: [wn.lemmatize(word) for word in x])
# print(tokenizedTextWithoutStopWordsAndStemmedAndLemmatized)


# Removing duplicates from tokenized text 
# > Indented block
# 
# 

# In[23]:

# TODO: check later
# tokenizedTextWithoutStopWordsAndStemmedAndLemmatizedWithoutDuplicates = tokenizedTextWithoutStopWordsAndStemmedAndLemmatized.apply(lambda x: list(set(x)))
# print(tokenizedTextWithoutStopWordsAndStemmedAndLemmatizedWithoutDuplicates)

