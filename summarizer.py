from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np

def get_summary(docs_list, k):
    """
    Returns summary of input documents.
    Summary is computed by choosing the top k sentences based on aggregated tf-idf values. 

    doc_list: List of documents to summarize
    k: The number of sentences to have in the summary. 
    """
    # Convert list of docs to list of sentences
    sent_list = []
    for doc in docs_list:
        curr_sents = sent_tokenize(doc)
        sent_list += curr_sents

    # Generate tf-idf values
    stop = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop)
    X = vectorizer.fit_transform(sent_list)
    X_array = X.toarray()

    # Aggregate tf-idf values for each sentence, and get the indicies of the top k values
    agg_X = np.sum(X_array, axis=1)
    sent_lens = np.sum(X_array != 0, axis=1)
    top_indicies = np.sort(np.argpartition(agg_X / sent_lens, -k)[-k:])

    # Return string containing the top k sentences in order (summary)
    summary = ""
    for index in top_indicies:
        summary += sent_list[index] + " "
    summary = summary.strip()
    return summary


if __name__ == "__main__":
    test_docs = ["This is the first test sentence. Here there's a second test sent.", "Second item in thet list. Last sentence for testing."]
    print(get_summary(test_docs, 2))