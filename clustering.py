from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag #Bad results for spanish
from nltk.tag.stanford import StanfordPOSTagger as StTagger
from gensim.models import KeyedVectors
# Frecuency print 
# from nltk.probability import FreqDist
# For documents classifications 
# from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

from string import punctuation
import numpy as np
import pprint
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.spatial.distance import cdist
from datetime import datetime

MODEL_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/models/spanish.tagger'
JAR_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/stanford-postagger.jar'
V2VEC_MODEL = None
#TRAINED_MODEL_V2VEC = 'GoogleNews-vectors-negative300.bin'
TRAINED_MODEL_V2VEC = 'SBW-vectors-300-min5.bin'
pp = pprint.PrettyPrinter(indent=4)


def _tagger(file, tagger_name='stanford'):
    if not tagger_name:
        tagger_name = "NLTK"
        tagger=None
    print("--Tagging (tagger=%s)..." %tagger_name)
    if tagger_name == "stanford":
        tagger = StTagger(MODEL_SFD_TAGGER, JAR_SFD_TAGGER)
    if tagger_name  == "spacy":
        import spacy
        tagger = spacy.load('es_core_web_md')
    raw_text = open(file).read()
    raw_sentences = sent_tokenize(raw_text)
    tagged_sentences = []
    extra_data_sentences = []
    for id, sentence in enumerate(raw_sentences):
        sentence = word_tokenize(sentence)
        if tagger_name == 'stanford':
            tagged_words = tagger.tag(sentence)
            extra_data = None
        elif tagger_name =='spacy':
            tagged_words = []
            extra_data = []
            doc = tagger(" ".join(_clean_sentence(sentence, tuples=False)))
            for token in doc:
                tagged_words.append((str(token).lower(), token.pos_, str(token)))
                extra_data.append((token.dep_, token.head.orth_))
        else:
            tagged_words = pos_tag(sentence, lang='es')
            extra_data = None
        tagged_sentences.append(tagged_words)
        extra_data_sentences.append(extra_data)
    return tagged_sentences, extra_data_sentences

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

def _clean_sentence(sentence, tuples=True):
    def need_change(word):
        if word.isdigit():
            return "NUMBER"
        return word
    stop = stopwords.words('spanish') + list(punctuation + "y")
    translator = str.maketrans('', '', punctuation + "¡¿")
    # Sentences as a list of tuples
    if tuples:
        sentence = [(word.lower(), tag, word) for word, tag in sentence]
        sentence = [(word.translate(translator), tag, rword) for word, tag, rword in sentence if word not in stop]
        sentence = [(need_change(word), tag, rword) for word, tag, rword in sentence if len(word) > 0]
    # Sentences as a list
    else:
        sentence = [word.lower() for word in sentence]
        sentence = [word.translate(translator) for word in sentence]
        sentence = [need_change(word) for word in sentence if len(word) > 1]
    return sentence

def _word_to_vec(word):
    model = V2VEC_MODEL
    if word in model:
        return model[word]
    else:
        return None

def featurize(tagged_sentences, with_w2vec=False, extra_data=None):
    print("--Featurizing (word_to_vec=%s)..." %str(with_w2vec))
    featurized = {}
    stopw = stopwords.words('spanish') + list(punctuation)
    for idy, tagged_sentence in enumerate(tagged_sentences):
        if extra_data is None:
            tagged_sentence = _clean_sentence(tagged_sentence)
        for idx, (word, POS, real_word) in enumerate(tagged_sentence):
            if word in featurized.keys():
                features = featurized[word]
            else:
                features = defaultdict(int)
                features['istittle:'] = word.istitle()
                if with_w2vec:
                    w2vec = _word_to_vec(word)
                    if w2vec is None:
                        continue
                    for w2vid, feature in enumerate(w2vec):
                        features['w2vec' + str(w2vid)] = feature
                # Muy sesgadas
                #features['word'] = word
                #features['isupper'] = word.isupper()
                #features['mayusinit'] = word[0].isupper()
                #features['lower:'] = word.lower()
                #features['tripla'] = ()
            if extra_data:
                try:
                    features[extra_data[idy][idx][0]] += 1
                    features[extra_data[idy][idx][1]] += 1
                except Exception:
                    print(extra_data[idy])

            features[POS] += 1
            features['mentions'] += 1
            #preword
            if idx == 0:
                features['START'] += 1
            else:
                features[tagged_sentence[idx - 1][0] + "-"] += 1
                features[tagged_sentence[idx - 1][1] + "-"] += 1
            #prepreword
            if idx <= 1:
                features['START-'] += 1
            else:
                features[tagged_sentence[idx - 2][0] + "--"] += 1
                features[tagged_sentence[idx - 2][1] + "--"] += 1 
            #posword
            if idx == len(tagged_sentence) - 1:
                features['END'] += 1
            else:
                features[tagged_sentence[idx + 1][0] + "+"] += 1
                features[tagged_sentence[idx + 1][1] + "+"] += 1
            #posposword
            if idx >= len(tagged_sentence) - 2:
                features['END+'] += 1
            else:
                features[tagged_sentence[idx + 2][0] + "++"] += 1
                features[tagged_sentence[idx + 2][1] + "++"] += 1
            featurized[word] = features
    return featurized

def _normalize(matrix):
    print("--Normalizing..")
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]

def _feature_selection(matrix, method="PCA"):
    print("--Selecting features with {} ".format(method))
    if method == "PCA":
        from sklearn.decomposition import PCA
        nf = 300
        pca = PCA(n_components=nf)
        pca.fit(matrix)
        reduced_matrix = pca.transform(matrix)
    return reduced_matrix

def vectorize(featurized_words, normalize=True, feature_selection=False):
    print("--Vectorizing...")
    words_index = []
    features_index = []
    mention_index = []
    for word in featurized_words.keys():
        if featurized_words[word]['mentions'] < 15 or len(word) < 4 or word=="NUMBER" or featurized_words[word]['mentions'] > 80 :
            continue
        mention_index.append(featurized_words[word].pop('mentions'))
        words_index.append(word)
        features_index.append(featurized_words[word])
    vectorizer = DictVectorizer(sparse=False)
    vectors = vectorizer.fit_transform(features_index)
    if normalize:
        vectors = _normalize(vectors)
    if feature_selection:
        print(vectors.shape)
        vectors = _feature_selection(vectors)
        print(vectors.shape)
    return words_index, vectors, mention_index

def _k_distortion(vectorized_words):
    distortions = []
    K = range(1,40)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(vectorized_words)
        kmeanModel.fit(vectorized_words)
        distortions.append(sum(np.min(cdist(vectorized_words, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vectorized_words.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def cluster(vectorized_words, word_index):
    print("--Clustering...")
    kmeans = KMeans(n_clusters=80).fit(vectorized_words)
    return kmeans

def preety_print_cluster(kmeans, refs, mentions):
    print("--Making graph...")
    labels = kmeans.labels_
    labeled = defaultdict(list)
    for id, label in enumerate(labels):
        labeled[label].append(refs[id])
    for label in labeled.keys():
        print(label)
        print(labeled[label])
    centroids = kmeans.cluster_centers_
    size = [len(word) for word in refs]
    data = np.array(list(zip(labels, refs, size, mentions)))
    plt.scatter(data[:,0], data[:,2],
             marker='o',
             c=data[:,0],
             s=list(map((lambda x: int(x)*10), data[:,3])),
             facecolors="white",
             edgecolors="blue")
    for idx, point in enumerate(refs):
        if mentions[idx] > 30:
            plt.annotate(point, xy=(data[:,0][idx], data[:,2][idx]))
    plt.ylabel('longitud')
    plt.xlabel('cluster')
    print("Finalizado (%s)" %str(datetime.now()))
    plt.show()

if __name__ == "__main__":
    with_w2vec = False
    with_spacy = True
    tagger = 'spacy'
    distortion = False
    file = "lavoz2000notas.txt"
    #file = "lavoz_minidump.txt"
    #file = "lavoztextodump.txt"
    print("Iniciando con {} ({})".format(file , str(datetime.now())))
    if with_w2vec:
        print("--Loading w2v model...")
        V2VEC_MODEL = KeyedVectors.load_word2vec_format(TRAINED_MODEL_V2VEC,
                                                        binary=True)
    tagged_sentences, extra_data = _tagger(file, tagger)
    words, vectors, mentions = vectorize(featurize(tagged_sentences,
                                                   with_w2vec=with_w2vec,
                                                   extra_data=extra_data))
    if distortion:
        _k_distortion(vectors)
    kmeans = cluster(vectors, words)
    preety_print_cluster(kmeans, words, mentions)
