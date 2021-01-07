""" D&D Class Classification
ML Controller

"""
import re
import spacy
from typing import List
from nltk.stem import PorterStemmer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from Fortuna import TruffleShuffle
from classifier.data_model import DataModel, dict_to_string

__all__ = ('Classifier',)


class CustomTokenizer:
    """ English NLP Tokenizer
    Method stem(str) -> str
    Callable(str) -> List[str] """
    __slots__ = ()
    nlp = spacy.load("en_core_web_sm")
    ps = PorterStemmer()

    def stem(self, word: str) -> str:
        """ Stemming for NLP """
        return self.ps.stem(re.sub('[^a-z]', '', word))

    def __call__(self, document: str) -> List[str]:
        """ Text Normalizer for NLP
        @param document: string to be tokenized.
        @return: List of tokenized strings. """
        document = self.nlp(document)
        return list({
            self.stem(token.lemma_.strip().lower())
            for token in document if not token.is_punct and not token.is_stop
        } - {''})


class Classifier:
    """ NLP Classifier
    Callable Object Factory
    Init takes a csv filename then prepares the data for NLP
    Calling the instance will produce a nearest match to the input string. """
    __slots__ = ('random_class', 'tfidf', 'knn', 'data')

    def __init__(self, filename, n_nearest=1):
        self.data = DataModel(filename)
        self.random_class = TruffleShuffle(self.data.class_details)
        self.tfidf = TfidfVectorizer(
            tokenizer=CustomTokenizer(),
            ngram_range=(1, 4),
            max_features=10000,
        )
        self.knn = NearestNeighbors(
            n_neighbors=n_nearest,
            n_jobs=-1,
        ).fit(self.tfidf.fit_transform(self.data.training_data).todense())

    def __call__(self, user_input: str) -> List[dict]:
        """ Natural Language Search
        @param user_input: natural language input
        @return: Returns the closest matches as a list of dictionaries """
        vec = self.tfidf.transform([user_input]).todense()
        ids, *_ = self.knn.kneighbors(vec, return_distance=False)
        return [self.data.class_details[i] for i in ids]


if __name__ == '__main__':
    classifier = Classifier('classifier/class_data.csv', n_nearest=3)
    tokenizer = CustomTokenizer()
    input_string = "I make potions with herbs grown in my garden"
    print(f"\nUser Input: {input_string}")
    print(f"Search based on: {tokenizer(input_string)}:\n")
    print('\n\n'.join(map(dict_to_string, classifier(input_string))), '\n')
