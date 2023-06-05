from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
from datetime import datetime
import country_converter as coco

import string
import re
import contractions
import pandas as pd
import nltk as nk
import numpy as np
import re

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
vec = CountVectorizer()
cc = coco.CountryConverter()


def handle_numbers(text):
    # Replace numbers with a special token
    text = re.sub(r'\d+', '', text)
    return text


def handle_countries(tokens):
    text = cc.pandas_convert(series=pd.Series(tokens, name='country'), to='ISO3', not_found=None)
    text = (' ').join(text)
    text = text.replace("united states", "USA")
    text = text.replace("united kingdom", "UK")
    return word_tokenize(text)


def handle_contractions(text):
    # Expand contractions in the text
    text = contractions.fix(text)
    return text


# def remove_punctuation(text):
#     # Define a translation table that excludes punctuation from dates
#     date_punctuation = string.punctuation.replace('-', '')
#     table = str.maketrans('', '', date_punctuation)

#     # Split the text into words
#     words = text.split()

#     # Remove punctuation from each word, except for words that look like dates
#     cleaned_words = []
#     for word in words:
#         if '/' in word and len(word.split('-')) == 3 :
#             # This word looks like a date, so don't remove punctuation
#             cleaned_words.append(word)
#         else:
#             # Remove punctuation from the word
#             cleaned_word = word.translate(table)
#             cleaned_words.append(cleaned_word)

#     # Join the cleaned words back into a string
#     cleaned_text = ' '.join(cleaned_words)
#     return cleaned_text
def remove_punctuation(text):
    # Remove punctuation characters from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


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
        text = text.replace(date, formatted_date)

    # Return the modified text with handled dates
    return text


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
    # tokens = [ps.stem(token) for token in tokens]

    # Lemmatization
    processed_text = lemmatize_with_pos(tokens)
    return processed_text


# Data Representation
def represent_text(text):
    x = vectorizer.fit_transform(text)
    return x


#Indexing
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


# Query Matching
def match_query(query_vector, document_vectors, candidate_docs, data, top_k=30):
    # Transform the query vector
    query_vector = vectorizer.transform([query_vector])

    # Compute the cosine similarities between the query vector and the candidate documents
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Rank and filter the documents based on their similarities to the query
    candidate_scores = {i: score for i, score in enumerate(similarities) if i in candidate_docs}
    sorted_indices = sorted(candidate_scores.keys(), key=lambda i: candidate_scores[i], reverse=True)[:top_k]
    ranked_documents = [data.iloc[i] for i in sorted_indices]

    # Return the top-ranked documents
    return ranked_documents


def get_candidate_docs(query, inverse_index):
    # query = preprocess_text(query)
    query_terms = query.split()
    relevant_docs = set()
    for term in query_terms:
        if term in inverse_index:
            relevant_docs.update(inverse_index[term])
    return relevant_docs


# Matching & ranking queries
def process_queries(queries, document_vectors, inverse_index, df):
    match_result = {}
    for index, query in queries.iterrows():
        processed_query = preprocess_text(query['text'])
        candidate_docs = get_candidate_docs(processed_query, inverse_index)
        query_results = match_query(processed_query, document_vectors, candidate_docs, df)
        match_result[query['query_id']] = [doc['doc_id'] for doc in query_results]
    return match_result


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


def evaluation(queries, result_matches, qrels, evaluation_results):
    for i, data in queries.iterrows():
        query_id = queries['query_id'][i]
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

        # Store the evaluation results for this query in the dictionary
        evaluation_results[query_id] = (prec, rec, f1)

    # Return the dictionary with evaluation results for all queries
    return evaluation_results
