from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag #Bad results
from nltk.tag.stanford import StanfordPOSTagger as StTagger
from gensim.models import KeyedVectors
# Frecuency print 
# from nltk.probability import FreqDist
# For documents classifications 
# from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

from string import punctuation
import numpy as np
import pprint
from collections import defaultdict

MODEL_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/models/spanish.tagger'
JAR_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/stanford-postagger.jar'
TRAINED_MODEL_V2VEC = 'GoogleNews-vectors-negative300.bin'
V2VEC_MODEL = KeyedVectors.load_word2vec_format(TRAINED_MODEL_V2VEC,
     binary=True)
pp = pprint.PrettyPrinter(indent=4)


def _pos_tag(file, tagger='stanford'):
    if tagger == "stanford":
        tagger = StTagger(MODEL_STANFORD_TAGGER, JAR_SFD_TAGGER)
    raw_text = open(file).read()
    raw_sentences = sent_tokenize(raw_text)
    tagged_sentences = []
    for id, sentence in enumerate(raw_sentences):
        sentence = word_tokenize(sentence)
        if tagger:
            tagged_words = tagger.tag(sentence)
        else:
            tagged_words = pos_tag(sentence, lang='es')
        tagged_sentences.append(tagged_words)
    return tagged_sentences

def featurize(tagged_sentences):
    featurized = {}
    stopw = stopwords.words('spanish') + list(punctuation)
    for tagged_sentence in tagged_sentences:
        print(tagged_sentence)
        tagged_sentence = _clean_sentence(tagged_sentence)
        print(tagged_sentence)
        for idx, (word, POS, real_word) in enumerate(tagged_sentence):
            w2vec = _word_to_vec(word)
            if not w2vec:
                continue
            if word in featurized.keys():
                features = featurized[word]
            else:
                features = defaultdict(int)
                features['isdigit:'] = word.isdigit()
                features['istittle:'] = word.istitle()
                features['w2vec'] = w2vec
                # Muy sesgadas
                #features['word'] = word
                #features['isupper'] = word.isupper()
                #features['mayusinit'] = word[0].isupper()
                #features['lower:'] = word.lower()
                #features['tripla'] = ()
            features[POS] += 1
            if idx == 0:
                features['START'] += 1
            else:
                features[tagged_sentence[idx - 1][0] + "-"] += 1
            if idx == len(tagged_sentence) - 1:
                features['END'] += 1
            else:
                features[tagged_sentence[idx + 1][0] + "+"] += 1
            featurized[word] = features
    return featurized

def vectorize(featurized_words):
    words_index = []
    features_index = []
    for word in featurized_words.keys():
        words_index.append(word)
        features_index.append(featurized_words[word])
    vectorizer = DictVectorizer(sparse=False)
    vectors = vectorizer.fit_transform(features_index)
    return words_index, vectors

def cluster(vectorized_words, word_index):
    kmeans = KMeans(n_clusters=4, random_state=0).fit(vectorized_words)
    return kmeans


def _only_filtered_words(file):
    """ Tokenize text and stem words removing punctuation """
    raw_text = open(file).read()
    tokenizer = RegexpTokenizer(r'\w+')
    stop = stopwords.words('spanish') + list(punctuation)
    # Tokenize to lower
    tokens = tokenizer.tokenize(raw_text.lower())
    # Remove punctuation and stop words
    tokens = [i for i in tokens if i not in stop]
    return tokens

def _clean_sentence(sentence):
    def need_change(word):
        if word.isdigit():
            return "NUMBER"
        return word
    # Sentences as a list
    stop = stopwords.words('spanish') + list(punctuation + "y")
    translator = str.maketrans('', '', punctuation + "¡¿")
    sentence = [(word.lower(), tag, word) for word, tag in sentence]
    sentence = [(word.translate(translator), tag, rword) for word, tag, rword in sentence if word not in stop]
    sentence = [(need_change(word), tag, rword) for word, tag, rword in sentence if len(word) > 0]
    return sentence

def preety_print_cluster(kmeans, refs, only_id=None):
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    items = defaultdict(list)
    for i, label in enumerate(labels):
        items[label] += refs
    pp.pprint(items)

def _word_to_vec(word):
    model = V2VEC_MODEL
    dog = model['dog']
    if word in model:
        return model[word].shape[:100]
    else:
        return None

if __name__ == "__main__":
    file = "one_note_lavoz.txt"
    # file = "lavoz_minidump.txt"
    '''
    pp.pprint(featurize(_pos_tag(file)))
    '''
    words, vectors = vectorize(featurize(_pos_tag(file, None)))
    kmeans = cluster(vectors, words)
    preety_print_cluster(kmeans, words)
    #print(result)

