# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# vectorizer = TfidfVectorizer()


# def preprocess_text(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token.lower() not in stop_words]

#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]

#     processed_text = ' '.join(tokens)

#     return processed_text


# def represent_data(data):
#     X = vectorizer.fit_transform(data)
#     return X


# # الفهرسة Indexing
# def build_index(data):
#     inverse_index = {}
#     for i, doc in enumerate(data):
#         terms = doc.split()
#         for term in terms:
#             if term not in inverse_index:
#                 inverse_index[term] = [i]
#             else:
#                 inverse_index[term].append(i)
#     return inverse_index


# # # Read data
# # df = pd.read_csv('path_to_dataset.csv')
# # # Data preprocess
# # df['processed_text'] = df['text'].apply(preprocess_text)
# # # Data representation
# # X = represent_data(df['processed_text'])
# #
# # # Build inverse index
# # inverse_index = build_index(df['processed_text'])


# def match_query(query_vector, document_vectors, candidate_docs, data, top_k=10):
#     query_vector = vectorizer.transform([query_vector])
#     similarities = cosine_similarity(query_vector, document_vectors).flatten()

#     # Rank the documents
#     sorted_indices = np.argsort(similarities)[::-1]
#     # print(sorted_indices)
#     # print(data)
#     ranked_documents = [data.iloc[i] for i in sorted_indices if i in candidate_docs]

#     # Return the top-ranked documents
#     search_results = ranked_documents[:top_k]
#     return search_results


# # query = "information retrieval"
# # query_vector = represent_data([preprocess_text(query)])[0]
# #
# # results = match_query(query_vector, X, inverse_index)
# # print(results)


# def get_candidate_docs(query, inverse_index, data):
#     # query = preprocess_text(query)
#     query_terms = query.split()
#     relevant_docs = set()
#     for term in query_terms:
#         if term in inverse_index:
#             relevant_docs.update(inverse_index[term])
#     return relevant_docs


# def process_queries(queries, document_vectors, inverse_index, df):
#     match_result = {}
#     for query in queries.iterrows():
#         processed_query = preprocess_text(query['text'])
#         candidate_docs = get_candidate_docs(processed_query, inverse_index, df)
#         query_results = match_query(processed_query, document_vectors, candidate_docs, df, top_k=5)
#         match_result[query['query_id']] = [doc['doc_id'] for doc in query_results]
#         if len(query_results) > 0:
#             print(f'query: {query}\nresult: {query_results[1]}')
#         else:
#             print(f'query: {query}\n no documents found')

#         print('===============================================')
#     return match_result
#
# # استخدام النظام
# query = "information retrieval"
# results = search(query, inverse_index, df)
# print(results)



import string
import re
import contractions
import pandas as pd
import nltk as nk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
#nk.download('stopwords')
#nk.download('word_tokenize')
from nltk.stem import WordNetLemmatizer
#nk.download('wordnet')
#snk.download('punkt')
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#########Data Preprocessing

def remove_punctuation(text):
    # Remove punctuation characters from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def handle_numbers(text):
    # Replace numbers with a special token
    text = re.sub(r'\d+', '<NUM>', text)
    return text

def handle_contractions(text):
    # Expand contractions in the text
    text = contractions.fix(text)
    return text

def lowercase(text):
    # Convert the text to lowercase
    text = text.lower()
    return text
def preprocess_text(text):
    text_str = str(text)
    
    text_str = remove_punctuation(text_str)
    text_str = handle_numbers(text_str)
    text_str = handle_contractions(text_str)
    text_str = lowercase(text_str)
    
    tokens = text_str.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    tokens = [ps.stem(token) for token in tokens]  
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  
    processed_text = ' '.join(tokens)
    return processed_text

######### Data Representation
def represent_text(text, vectorizer):
    x = vectorizer.fit_transform(text)
    return x

#########Indexing
def build_index(data):
    inverse_index = {}
    for i, doc in enumerate(data):
        terms = doc.split()
        for term in terms:
            if term not in inverse_index:
                inverse_index[term] = [i]
            else:
                # Only add unique document indices
                if i not in inverse_index[term]:
                    inverse_index[term].append(i)
    
    # Sort document indices and add term frequency information
    for term, doc_indices in inverse_index.items():
        doc_indices.sort()
        term_freq = {}
        for i in doc_indices:
            if i not in term_freq:
                term_freq[i] = 1
            else:
                term_freq[i] += 1
        inverse_index[term] = (doc_indices, term_freq)
    
    return inverse_index


#########Query Matching
def match_query(query_vector, document_vectors, candidate_docs, data, top_k=10):
    query_vector = vectorizer.transform([query_vector])
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Rank the documents
    sorted_indices = np.argsort(similarities)[::-1]
    # print(sorted_indices)
    # print(data)
    ranked_documents = [data.iloc[i] for i in sorted_indices if i in candidate_docs]

    # Return the top-ranked documents
    search_results = ranked_documents[:top_k]
    return search_results


def get_candidate_docs(query, inverse_index, data):
    query_terms = query.split()
    relevant_docs = set()
    for term in query_terms:
        if term in inverse_index:
            relevant_docs.update(inverse_index[term])
    return relevant_docs


####### Matching & ranking queries
def process_queries(queries, document_vectors, inverse_index, df):
    match_result = {}
    for index, query in queries.iterrows():
        processed_query = preprocess_text(query['text'])
        candidate_docs = get_candidate_docs(processed_query, inverse_index, df)
        query_results = match_query(processed_query, document_vectors, candidate_docs, df, top_k=5)
        match_result[query['query_id']] = [doc['doc_id'] for doc in query_results]
        if len(query_results) > 0:
            print(f'query: {query}\nresult: {query_results[1]}')
        else:
            print(f'query: {query}\n no documents found')

        print('===============================================')
        return match_result




