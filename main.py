# This is a sample Python script.
import nltk
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import IR1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = IR1.read_df(path='dataset/docs.csv')
    tokenized_df = IR1.tokenize_df(df.head(10))
    print(f'tokenized_df: {tokenized_df}')
    clear_data = IR1.clean_data(tokenized_df)
    print(f'clear_data: {clear_data}')
    merged_text = [' '.join(row) for row in clear_data]
    tfidf_vectors, features = IR1.tfidf_vector(merged_text)
    print(f'tfidf_vectors {tfidf_vectors}\nfeatures: {features}')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
