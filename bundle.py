import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as  np
def preprocess_text(text):

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    processed_text = ' '.join(tokens)

    return processed_text


def represent_data(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    return X

# الفهرسة Indexing
def build_index(data):
    inverse_index = {}
    for i, doc in enumerate(data):
        terms = doc.split()
        for term in terms:
            if term not in inverse_index:
                inverse_index[term] = [i]
            else:
                inverse_index[term].append(i)
    return inverse_index

# Read data
df = pd.read_csv('path_to_dataset.csv')
# Data preprocess
df['processed_text'] = df['text'].apply(preprocess_text)
# Data representation
X = represent_data(df['processed_text'])

# Build inverse index
inverse_index = build_index(df['processed_text'])


def match_query(query_vector, document_vectors, inverse_index):

    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # ترتيب النتائج بناءً على قيم التشابه
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_documents = inverse_index.iloc[ranked_indices]

    return ranked_documents


query = "information retrieval"
query_vector = represent_data([preprocess_text(query)])[0]

results = match_query(query_vector, X, inverse_index)
print(results)
# # استعلام المستخدم
# def search(query, inverse_index, data):
#     query = preprocess_text(query)
#     query_terms = query.split()
#
#     # استرجاع المستندات ذات الصلة
#     relevant_docs = set()
#     for term in query_terms:
#         if term in inverse_index:
#             relevant_docs.update(inverse_index[term])
#
#     # إرجاع المستندات المطابقة
#     return data.iloc[list(relevant_docs)]
#
# # استخدام النظام
# query = "information retrieval"
# results = search(query, inverse_index, df)
# print(results)
