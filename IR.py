# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from dateutil.parser import parse
# from datetime import datetime
# from flask import Flask, request, jsonify
# from dateutil.parser import parse
# from datetime import datetime
# import string
# import re
# import contractions
# import pandas as pd
# import nltk as nk
# import numpy as np
# import re
# import country_converter as coco
# import pickle
#
# app = Flask(__name__)
#
# ps = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
# vectorizer = TfidfVectorizer()
# vec = CountVectorizer()
# cc = coco.CountryConverter()
#
#
# def handle_numbers(text):
#     # Replace numbers with a special token
#     text = re.sub(r'\d+', '', text)
#     return text
#
#
# def handle_countries(tokens):
#     text = cc.pandas_convert(series=pd.Series(tokens, name='country'), to='ISO3', not_found=None)
#     text = (' ').join(text)
#     text = text.replace("united states", "USA")
#     text = text.replace("united kingdom", "UK")
#     return word_tokenize(text)
#
#
# def handle_contractions(text):
#     # Expand contractions in the text
#     text = contractions.fix(text)
#     return text
#
#
# def remove_punctuation(text):
#     # Define a translation table that excludes punctuation from dates
#     date_punctuation = string.punctuation.replace('-', '')
#     table = str.maketrans('', '', date_punctuation)
#
#     # Split the text into words
#     words = text.split()
#
#     # Remove punctuation from each word, except for words that look like dates
#     cleaned_words = []
#     for word in words:
#         if '/' in word and len(word.split('-')) == 3:
#             # This word looks like a date, so don't remove punctuation
#             cleaned_words.append(word)
#         else:
#             # Remove punctuation from the word
#             cleaned_word = word.translate(table)
#             cleaned_words.append(cleaned_word)
#
#     # Join the cleaned words back into a string
#     cleaned_text = ' '.join(cleaned_words)
#     return cleaned_text
#
#
# # def remove_punctuation(text):
# #     # Remove punctuation characters from the text
# #     text = text.translate(str.maketrans('', '', string.punctuation))
# #     return text
#
# def lowercase(text):
#     # Convert the text to lowercase
#     text = text.lower()
#     return text
#
#
# def handle_dates(text):
#     # Define regular expression to match dates in text
#     regex = r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b'
#
#     # Search for all dates in text
#     dates = re.findall(regex, text)
#
#     # Loop through all dates found and convert to the desired format
#     for date in dates:
#         # Try to parse the date - skip if it is an invalid date
#         try:
#             datetime_obj = datetime.strptime(date, '%d/%m/%Y')
#         except ValueError:
#             try:
#                 datetime_obj = datetime.strptime(date, '%Y/%m/%d')
#             except ValueError:
#                 try:
#                     datetime_obj = datetime.strptime(date, '%m/%d/%Y')
#                 except ValueError:
#                     continue
#
#         # Convert date to desired format
#         formatted_date = datetime_obj.strftime('%Y-%m-%d')
#
#         # Replace the original date in the text with the formatted date
#         text = text.replace(date, "")
#
#     # Return the modified text with handled dates
#     return text
#
#
# # Data Representation
# def lemmatize_with_pos(tokens):
#     pos_tags = nk.pos_tag(tokens)
#     pos_lemmas = []
#     for word, pos in pos_tags:
#         if pos.startswith('J'):
#             # If the word is an adjective, use 'a' (adjective) as the POS tag
#             pos_lemmas.append(lemmatizer.lemmatize(word, pos='a'))
#         elif pos.startswith('V'):
#             # If the word is a verb, use 'v' (verb) as the POS tag
#             pos_lemmas.append(lemmatizer.lemmatize(word, pos='v'))
#         elif pos.startswith('N'):
#             # If the word is a noun, use 'n' (noun) as the POS tag
#             pos_lemmas.append(lemmatizer.lemmatize(word, pos='n'))
#         else:
#             # For all other cases, use the default POS tag 'n' (noun)
#             pos_lemmas.append(lemmatizer.lemmatize(word))
#     return ' '.join(pos_lemmas)
#
#
# def preprocess_text(text):
#     text_str = str(text)
#     # Normalization
#     text_str = remove_punctuation(text_str)
#     text_str = handle_dates(text_str)
#     # text_str = handle_numbers(text_str)
#     text_str = handle_contractions(text_str)
#     text_str = lowercase(text_str)
#
#     # Tokenization
#     tokens = word_tokenize(text_str)
#     stop_words = set(stopwords.words('english'))
#     # Removing stop worlds
#     tokens = [token for token in tokens if token not in stop_words]
#
#     # tokens = handle_countries(tokens)
#
#     # Stemming
#     # tokens = [ps.stem(token) for token in tokens]
#
#     # Lemmatization
#     processed_text = lemmatize_with_pos(tokens)
#     return processed_text
#
#
# # Data Representation
# def represent_text(text):
#     x = vectorizer.fit_transform(text)
#     return x
#
#
# # Indexing
# def build_index(data):
#     inverse_index = {}
#     for i, doc in enumerate(data):
#         terms = doc.split(' ')
#         for term in terms:
#             if term not in inverse_index:
#                 inverse_index[term] = [i]
#             else:
#                 inverse_index[term].append(i)
#     return inverse_index
#
#
# def match_query(query_vector, document_vectors, candidate_docs, data, top_k):
#     query_vector = vectorizer.transform([query_vector])
#     similarities = cosine_similarity(query_vector, document_vectors).flatten()
#
#     # Rank the documents
#     sorted_indices = np.argsort(similarities)[::-1]
#     # print(sorted_indices)
#     # print(data)
#     ranked_documents = [data.iloc[i] for i in sorted_indices if i in candidate_docs]
#
#     # Return the top-ranked documents
#     search_results = ranked_documents[:top_k]
#     return search_results
#
#
# # def get_candidate_docs(query, inverse_index):
# #     if not query:
# #         return set()
# #
# #     query_terms = query.split(' ')
# #     if not query_terms:
# #         return set()
# #
# #     relevant_docs = set()
# #     for term in query_terms:
# #         if term in inverse_index:
# #             relevant_docs.update(inverse_index[term])
# #     return relevant_docs
#
# def get_candidate_docs(query_text, inverse_index):
#     # Split the query text into terms
#     query_terms = query_text.split()
#
#     # Retrieve the candidate document IDs for the query
#     candidate_doc_ids = set(inverse_index.get(query_terms[0], []))
#     for term in query_terms[1:]:
#         candidate_doc_ids.intersection_update(set(inverse_index.get(term, [])))
#
#     # Return the candidate document IDs
#     return candidate_doc_ids
#
#
# def precision(tp, fp):
#     """
#     Calculate precision given the number of true positives (tp) and false positives (fp).
#     """
#     return tp / (tp + fp + 1e-10)
#
#
# def recall(tp, fn):
#     """
#     Calculate recall given the number of true positives (tp) and false negatives (fn).
#     """
#     return tp / (tp + fn + 1e-10)
#
#
# def f1_score(precision, recall):
#     """
#     Calculate F1-score given precision and recall.
#     """
#     return 2 * precision * recall / (precision + recall + 1e-10)
#
#
# def evaluation(queries, result_matches, qrels, evaluation_results):
#     f1_scores = []
#     for (index, query_id, text) in queries.values:
#         if query_id in evaluation_results:
#             # If this query has already been evaluated, skip it
#             continue
#
#         # Retrieve the ground truth relevance dataset for this query
#         relevant_docs = list(qrels.loc[qrels['query_id'] == query_id, 'doc_id'])
#
#         # Retrieve the relevant documents retrieved by the system for this query
#         result_match = result_matches[query_id]
#         # Compute tp (true positives)
#         tp = len(np.intersect1d(relevant_docs, result_match))
#         # Compute fp (false positives)
#         fp = len(result_match) - tp
#
#         # Compute fn (false negatives)
#         fn = len(list(set(relevant_docs) - set(result_match)))
#
#         # Compute precision, recall, and f1_score
#         prec = precision(tp, fp)
#         rec = recall(tp, fn)
#         f1 = f1_score(prec, rec)
#
#         # Store the f1 score in the list
#         f1_scores.append(f1)
#
#         # Store the evaluation results for this query in the dictionary
#         evaluation_results[query_id] = (prec, rec, f1)
#
#     # Compute the average F1 score and return it
#     avg_f1 = sum(f1_scores) / len(f1_scores)
#     return avg_f1
#######################################################################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
from datetime import datetime
from flask import Flask, request, jsonify
from dateutil.parser import parse
from datetime import datetime
import string
import re
import contractions
import pandas as pd
import nltk as nk
import numpy as np
import re
# import country_converter as coco
import pickle

app = Flask(__name__)

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
vec = CountVectorizer()
# cc = coco.CountryConverter()


def handle_numbers(text):
    # Replace numbers with a special token
    text = re.sub(r'\d+', '', text)
    return text


# def handle_countries(tokens):
#     text = cc.pandas_convert(series=pd.Series(tokens, name='country'), to='ISO3', not_found=None)
#     text = (' ').join(text)
#     text = text.replace("united states", "USA")
#     text = text.replace("united kingdom", "UK")
#     return word_tokenize(text)


def handle_contractions(text):
    # Expand contractions in the text
    text = contractions.fix(text)
    return text


def remove_punctuation(text):
    # Define a translation table that excludes punctuation from dates
    date_punctuation = string.punctuation.replace('-', '')
    table = str.maketrans('', '', date_punctuation)

    # Split the text into words
    words = text.split()

    # Remove punctuation from each word, except for words that look like dates
    cleaned_words = []
    for word in words:
        if '/' in word and len(word.split('-')) == 3:
            # This word looks like a date, so don't remove punctuation
            cleaned_words.append(word)
        else:
            # Remove punctuation from the word
            cleaned_word = word.translate(table)
            cleaned_words.append(cleaned_word)

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


# def remove_punctuation(text):
#     # Remove punctuation characters from the text
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text

def lowercase(text):
    # Convert the text to lowercase
    text = text.lower()
    return text


def handle_dates(text):
    # Define regular expression to match dates in text
    regex = r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b'

    # Search for all dates in text
    dates = re.findall(regex, text)

    # Loop through all dates found and convert to the desired format
    for date in dates:
        # Try to parse the date - skip if it is an invalid date
        try:
            datetime_obj = datetime.strptime(date, '%d/%m/%Y')
        except ValueError:
            try:
                datetime_obj = datetime.strptime(date, '%Y/%m/%d')
            except ValueError:
                try:
                    datetime_obj = datetime.strptime(date, '%m/%d/%Y')
                except ValueError:
                    continue

        # Convert date to desired format
        formatted_date = datetime_obj.strftime('%Y-%m-%d')

        # Replace the original date in the text with the formatted date
        text = text.replace(date, "")

    # Return the modified text with handled dates
    return text


# Data Representation
def lemmatize_with_pos(tokens):
    pos_tags = nk.pos_tag(tokens)
    pos_lemmas = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            # If the word is an adjective, use 'a' (adjective) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='a'))
        elif pos.startswith('V'):
            # If the word is a verb, use 'v' (verb) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='v'))
        elif pos.startswith('N'):
            # If the word is a noun, use 'n' (noun) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='n'))
        else:
            # For all other cases, use the default POS tag 'n' (noun)
            pos_lemmas.append(lemmatizer.lemmatize(word))
    return ' '.join(pos_lemmas)


def preprocess_text(text):
    text_str = str(text)
    # Normalization
    text_str = remove_punctuation(text_str)
    text_str = handle_dates(text_str)
    text_str = handle_numbers(text_str)
    text_str = handle_contractions(text_str)
    text_str = lowercase(text_str)

    # Tokenization
    tokens = word_tokenize(text_str)
    stop_words = set(stopwords.words('english'))
    # Removing stop worlds
    tokens = [token for token in tokens if token not in stop_words]

    # tokens = handle_countries(tokens)

    # Stemming
    tokens = [ps.stem(token) for token in tokens]

    # Lemmatization
    processed_text = lemmatize_with_pos(tokens)
    return processed_text


# Data Representation
def represent_text(text):
    x = vectorizer.fit_transform(text)
    return x


# Indexing
def build_index(data):
    inverse_index = {}
    for i, doc in enumerate(data):
        terms = doc.split(' ')
        for term in terms:
            if term not in inverse_index:
                inverse_index[term] = [i]
            else:
                inverse_index[term].append(i)
    return inverse_index


def match_query(query_vector, document_vectors, candidate_docs, data, top_k):
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


# def get_candidate_docs(query, inverse_index):
#     if not query:
#         return set()
#
#     query_terms = query.split(' ')
#     if not query_terms:
#         return set()
#
#     relevant_docs = set()
#     for term in query_terms:
#         if term in inverse_index:
#             relevant_docs.update(inverse_index[term])
#     return relevant_docs

def get_candidate_docs(query_text, inverse_index):
    if not query_text.strip():
        return set()
    # Split the query text into terms
    query_terms = query_text.split()

    # Retrieve the candidate document IDs for the query
    candidate_doc_ids = set(inverse_index.get(query_terms[0], []))
    for term in query_terms[1:]:
        candidate_doc_ids.intersection_update(set(inverse_index.get(term, [])))

    # Return the candidate document IDs
    return candidate_doc_ids


def precision(tp, fp):
    """
    Calculate precision given the number of true positives (tp) and false positives (fp).
    """
    return tp / (tp + fp + 1e-10)


def recall(tp, fn):
    """
    Calculate recall given the number of true positives (tp) and false negatives (fn).
    """
    return tp / (tp + fn + 1e-10)


def f1_score(precision, recall):
    """
    Calculate F1-score given precision and recall.
    """
    return 2 * precision * recall / (precision + recall + 1e-10)

def calculate_precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_docs_at_k = retrieved_docs[:k]
    relevant_and_retrieved = set(relevant_docs).intersection(set(retrieved_docs_at_k))
    precision = len(relevant_and_retrieved) / k
    return precision

def calculate_mrr(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0

def evaluation(queries, result_matches, qrels, evaluation_results):
    f1_scores = []
    precesion_at_k = []
    average_precision = []
    mrr_scores = []
    for (index, query_id, text) in queries.values:
        if query_id in evaluation_results:
            # If this query has already been evaluated, skip it
            continue

        # Retrieve the ground truth relevance dataset for this query
        relevant_docs = list(qrels.loc[qrels['query_id'] == query_id, 'doc_id'])

        # Retrieve the relevant documents retrieved by the system for this query
        result_match = result_matches[query_id]
        # Compute tp (true positives)
        tp = len(np.intersect1d(relevant_docs, result_match))
        # Compute fp (false positives)
        fp = len(result_match) - tp

        # Compute fn (false negatives)
        fn = len(list(set(relevant_docs) - set(result_match)))

        # Compute precision, recall, and f1_score
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        f1 = f1_score(prec, rec)
        pre_k = calculate_precision_at_k(relevant_docs, result_match, 10)
        mrr = calculate_mrr(relevant_docs, result_match)
        # Store the f1 score in the list
        f1_scores.append(f1)
        average_precision.append(prec)
        precesion_at_k.append(pre_k)
        mrr_scores.append(mrr)
        # Store the evaluation results for this query in the dictionary
        evaluation_results[query_id] = (prec, rec, f1)

    # Compute the average and return it
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_pre = sum(average_precision) / len(average_precision)
    avg_pre_at_k = sum(precesion_at_k) / len(precesion_at_k)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    return avg_f1, avg_pre, avg_pre_at_k, avg_mrr