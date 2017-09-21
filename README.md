# Clustering con embeddings

## Introducción
En este proyecto se intenta aplicar la técnica de aprendizaje no supervisado denominada clustering sobre un corpus de noticias
de La Voz del Interior, medio gráfico Cordobés.
Se utilizarán tambien técnicas de normalizacion, manipulación de features y vectorización de estas palabras para luego aplicar
clustering.
Como resultado no se espera tener algo significativo debido a que un corpus de noticias no es algo especifico de un nicho de
palabras. Es decir al tener tanta variedad en las palabras, es posible que tengamos que conformarnos con algunos clusters bien
identificados, quizás las palabras que mas se mencionen dentro del texto.
Además utilizaremos variaciones en los features seleccionados para vectorizar, y hasta se incorporarán word-embeddings tales
word2vec o triplas de dependencia, para obtener quizás mejores resultados.
El trabajo se desarrollará integramente en python.

## Las librerias que utilizaremos..
### requirements.txt
* ipython
* nltk
* sklearn
* numpy
* scipy
* gensim
* matplotlib

```python
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
'''
Para separar las oraciones del texto y posteriormente separar las palabras.
RegexpTokenazer puede ser utilizado para evitar algunas palabras inutiles para el procesamiento.
No lo utilizamos porque creemos que eran importantes para un contexto, recien fueron retiradas 
en el proceso de vectorizacion.
'''
from nltk.corpus import stopwords
from nltk import pos_tag #Bad results
from nltk.tag.stanford import StanfordPOSTagger as StTagger
'''
Sobre NLTK utilizamos las stopwords 'spanish' para retirarlas del analisis,
también probamos su POS tagger nativo pero no obtuvimos buenos resultados sobre
el español. Por eso decidimos incorporar StanfordPOSTAgger que corre sobre un backend
de JAVA externo pero esta libreria nos facilita este vinculo.
'''
from gensim.models import KeyedVectors
'''
Gensim fue necesario para realizar word2vec. Word2Vec esta deprecada, su update es
KeyedVectors. Con el importamos un modelo ya entrenado del español.
'''
# from nltk.probability import FreqDist
'''
Para retirar las palabras poco frecuentes
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
'''
SKLEARN fue utilizado tanto para la tarea propia de clustering con KMeans como 
asi tambien para vectorizar las palabras con sus caracteristicas. CountVectorizer no
fue utilizado porque se prefiriò hacer algo mas "casero" para profundizar los conceptos.
'''

from string import punctuation
import numpy as np
import pprint
from collections import defaultdict
'''
Librerias utiles para tratar problemas cotidianos de estructuras de datos.
'''
```
## Herramientas a utilizar/desarrollar
* *Cleaner* o normalizador de texto: Eliminará las palabras poco frecuentes y 'basura' dentro del texto.
* *Featurizer*: Se encargará de disponibilizar las caracteristicas de las palabras al vectorizer
* *Vectorizer*: Hará de cada palabra un vector numerico que describa la palabra representada mediante sus caracteristicas.
* *Clusterizer*: Separará las palabras de acuerdo a la similitud de vectores.

## Uso de binarios externos.
Antes de empezar definiremos algunos modelos públicos utilizados en el desarrollo del trabajo.
Para el caso de POS tag en palabras decidimos inicialmente usar NLTK, pero no descartamos otras alternativas ya que en español
no es tan recomendable.
Como alternativa entonces decidimos probar tambien el [POS tagger de Stanford](link) y comparar resultados. En este caso python
solo ofrece una libreria para integrar un jar y un modelo de Stanford. Definiremos entonces estas constantes:
```python
MODEL_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/models/spanish.tagger'
JAR_SFD_TAGGER = 'stanford-postagger-full-2017-06-09/stanford-postagger.jar'
```
Para el caso de word2vec también integramos un [modelo preentrenado](link a modelo) de Cristian Cardellino, aunque en las pruebas
preeliminares utilizamos uno de GoogleNews aunque con muy malos resultados.
```python
#TRAINED_MODEL_V2VEC = 'GoogleNews-vectors-negative300.bin'
TRAINED_MODEL_V2VEC = 'SBW-vectors-300-min5.bin'
```

## Estructura del código
### Funciones auxiliares.
Para el etiquetado de POS
```python
def _pos_tag(file, tagger='stanford'):
```
Para normalizar cada oración
```python
def _clean_sentence(sentence):
```
Para obtener el vector de W2V para cada palabra
```python
def _word_to_vec(word):
```

### Pipeline
Primero obtenemos de cada palabra sus features
```python
def featurize(tagged_sentences, with_w2vec=True):
```
Luego una vez que caracterizamos cada palabra con sus caracteristicas, es decir
un diccionario donde cada caracteristica esta representada por una clave y tiene
asignado un valor ya sea numerico, booleano o de texto. Vectorizaremos la palabra
automaticamente gracias a DictVectorizer.
```python
def vectorize(featurized_words):
```
Por ultimo con la matriz de vectores, solo nos queda agrupar mediante un proceso
de clustering donde mediante la configuracion de la cantidad de centroides el
algoritmo iterará hasta converger en una asignación significativa de las palabras
vectorizadas. 
```python
def cluster(vectorized_words, word_index):
```
Una vez etiquetados dichos vectores con el numero de cluster al cual pertenecen,
será solo tarea de integrar esa informacion a los datos de las palabras y mostrar
una visualizacion humanizada.
```python
def preety_print_cluster(kmeans, refs, only_id=None):
```
## Preguntas frecuentes
* ¿Cuántos clusters es lo optimo para mi corpus?
En cuanto a esta pregunta la respondimos mediante el uso del cálculo de la 
distorsion entre los clusters. Obtuvimos algunas diferencias interesantes entre
la distorsion generada con el tagger de NLTK y el de Stanford. Esto representa
que el POS tag aporta mucha información, es un feature muy relevante en el 
proceso.
Con NLTK tagger:
[IMAGEN NLTK]()

Con Stanford tagger:
[IMAGEN STANFORD]()

* ¿Cómo realizar un gráfico de los clusters?
A la hora de graficar el cluster no pudimos encontrar una forma interesante de
ver los datos. Optamos por crear 2 dimensiones que nos parecieran relevantes,
tales como la longitud de la palabra y el cluster donde se ubican para graficarlas
en el plano. La dimension del punto será la referencia a la cantidad de menciones
en el texto y a aquellas palabras que tengan gran volumen de menciones colocarles
su nombre para saber de cuales se trata. Cabe aclarar que las caracteristicas
que se grafiquen en el plano puede que no sean relevantes pero en el caso de
palabras aisladas es dificil obtener una característica de evaluación importante.

* ¿Hacer la normalización del texto antes o despues de efectuar la featurización?
En este trabajo la normalización se hizo en varias etapas. La tokenizacion retiró
las stopwords , puntuación y simbolos extraños. Por su parte la vectorizacion,
descartó las palabras poco frecuentes y también las palabras de muy corta longitud
como asi tambien los numeros, ya que suelen ser irrelevantes y posiblemente 
queden aisladas por la repetición muy baja o muy alta.


## Observaciones(revisar)
* NLTK mucho mas rapido, pero mucho peor. Stanford bastante correcto para el 
castellano
* Remover las palabras cortas mejora muchisimo la clusterizacion ya que por mas 
que no sean stopwords pueden ocurrir muchas veces.
* Como ya habiamos mencionado en la introducción, al no ser un texto especifico
de algún área, la clusterizacion puede caer muchas veces en agrupamientos tipo
singletones, donde una palabra puede ser un cluster debido a que tiene 
características muy diferentes al resto. Esto no nos beneficia ya que perdemos 
un cluster solo para dicha palabra.
* No notamos grandes mejoras en cuanto al uso de word-embeddings aunque 
parecen útiles para otras tareas.

## Resultados
### Stanford:
Aqui vemos que se obtuvieron algunos clusters interesantes sobre el corpus de 
300 notas como el caso del n8 donde se agruparon algunas palabras como "inflacion",
"conflicto" ,"momento", "sociedad"  y demás, aunque obviamente también aparecen
palabras poco relacionadas como "hombre" o "mujer" que pueden estar relacionadas
entre si aunque no con la "inflación" por ejemplo.

Por otro lado vemos clusters como el n14 donde se asociaron solo 2 palabras que
suelen tener muchos contextos iguales. "nacional" y "provincial" son dos palabras
correctamente asociadas ya que son muy similares en su uso. De igual forma el
n22 asociando solo numeros escritos con palabras

```python
Iniciando con lavoz300notas.txt (2017-09-21 10:17:11.073778)
--Loading w2v model...
--POS Tagging (tagger=stanford)...
--Featurizing (word_to_vec=True)...
--Vectorizing...
--Clustering...
--Making graph...
0
['pesos']
1
['pasó', 'septiembre', 'sido', 'aseguró', 'bajo', 'hacia', 'señaló', 'haber', 'sino', 'informó', 'podría', 'miércoles', 'cómo', 'octubre', 'crisis']
2
['años']
3
['interior', 'comercio', 'casa']
4
['córdoba']
5
['ciento']
6
['estudiantes', 'horas', 'semáforos', 'trabajadores', 'autoridades', 'víctimas', 'jóvenes', 'veces', 'mujeres', 'alumnos', 'empresas', 'votos', 'elecciones', 'chicos']
7
['ayer']
8
['roca', 'inflación', 'apoyo', 'situación', 'mundo', 'conflicto', 'posibilidad', 'fiscal', 'actividad', 'grupo', 'reunión', 'momento', 'diálogo', 'acto', 'nivel', 'carne', 'semana', 'relación', 'sociedad', 'decisión', 'medida', 'noche', 'lado', 'ruta', 'derecho', 'colegio', 'toma', 'información', 'hombre', 'mujer', 'cuenta', 'embargo', 'tipo', 'jefe', 'central', 'historia', 'zona', 'norte', 'hecho', 'vida']
9
['políticos', 'político', 'importante', 'provinciales', 'largo', 'gran', 'nuevo', 'grandes', 'nueva', 'internacional', 'mejor', 'junto']
10
['país', 'presidente', 'parte', 'partido', 'gobernador']
11
['cámara', 'belgrano', 'justicia', 'buenos', 'manuel', 'cruz', 'fiat', 'villa', 'kirchner', 'santa', 'alta', 'josé', 'brasil', 'aires', 'ecuador', 'maría', 'luis', 'lula', 'ministerio', 'justo', 'daniel', 'correa']
12
['juan', 'policía', 'capital', 'educación', 'argentina', 'provincia']
13
['bien', 'luego', 'frente', 'atrás', 'ahora', 'dentro', 'casi', 'además', 'allí', 'siempre', 'después', 'menos']
14
['nacional', 'provincial']
15
['tras', 'según']
16
['dijo']
17
['pueden', 'debe', 'dice', 'quiere', 'puede']
18
['manera', 'mañana', 'forma', 'tiempo', 'proyecto', 'lugar', 'empresa', 'poder', 'trabajo', 'caso', 'política', 'ciudad']
19
['sólo']
20
['millones']
21
['gobierno']
22
['tres', 'cuatro', 'ocho', 'cinco']
23
['primer', 'primera', 'últimos']
24
['hacer', 'llegar', 'decir', 'tener']
25
['general', 'mayor', 'pasado']
26
['sectores', 'meses', 'personas', 'escuelas', 'días', 'obras']
27
['aunque', 'mientras']
28
['toda', 'cada', 'mismo', 'varios']
29
['hace']
Finalizado (2017-09-21 10:50:04.996516)
```
```python
Iniciando con lavoz300notas.txt (2017-09-21 11:11:29.749428)
--POS Tagging (tagger=NLTK)...
--Featurizing (word_to_vec=False)...
--Vectorizing...
--Clustering...
--Making graph...
0
['primer']
1
['años']
2
['últimos', 'semáforos', 'nueva', 'central', 'ocho', 'internacional', 'fiscal', 'carne', 'pueden', 'nuevo']
3
['educación']
4
['córdoba']
5
['millones']
6
['kirchner', 'schiaretti', 'daniel', 'luis', 'correa', 'belgrano', 'maría', 'justicia', 'santa', 'manuel', 'cámara', 'villa']
7
['ahora', 'tras']
8
['hace']
9
['actividad', 'casi', 'medida', 'toma', 'posibilidad', 'vida', 'pasó', 'historia', 'mejor', 'inflación', 'momento']
10
['tres', 'sectores', 'meses']
11
['luego', 'trabajo', 'política', 'forma', 'cada']
12
['nacional']
13
['autoridades']
14
['dice', 'roca', 'siempre', 'frente', 'cómo', 'gran', 'colegio']
15
['juan']
16
['gobierno']
17
['ayer']
18
['provincial']
19
['además']
20
['sólo', 'dijo']
21
['policía', 'capital']
22
['según']
23
['parte', 'mayor', 'primera', 'caso', 'proyecto', 'puede', 'mismo']
24
['gobernador', 'días', 'país']
25
['empresa']
26
['presidente']
27
['votos', 'situación', 'sociedad', 'varios', 'norte', 'septiembre', 'noche', 'derecho', 'víctimas', 'ruta', 'zona', 'haber', 'reunión', 'octubre']
28
['ciento']
29
['alta', 'ecuador', 'lula', 'ministerio', 'josé', 'cruz', 'néstor', 'brasil', 'interior', 'fiat', 'justo']
30
['provincia']
31
['pasado']
32
['bien', 'crisis', 'sido', 'cuenta', 'lugar', 'conflicto', 'cinco', 'sino', 'poder', 'chicos', 'grupo', 'tiempo']
33
['argentina']
34
['aunque', 'general', 'alumnos']
35
['semana']
36
['menos']
37
['trabajadores', 'miércoles', 'estudiantes']
38
['personas']
39
['importante', 'hombre', 'señaló', 'lado', 'hecho', 'decir', 'atrás', 'acto', 'hacia', 'político', 'tipo', 'largo', 'jefe', 'tener', 'bajo', 'diálogo', 'hacer']
40
['manera', 'mañana', 'mientras', 'obras', 'escuelas']
41
['grandes', 'veces', 'jóvenes', 'elecciones', 'provinciales', 'mujeres', 'horas']
42
['buenos']
43
['después', 'cuatro', 'comercio', 'embargo', 'allí']
44
['relación', 'apoyo', 'nivel', 'políticos', 'información', 'dentro', 'mundo', 'aseguró', 'empresas', 'informó', 'debe', 'podría', 'toda', 'decisión', 'llegar', 'mujer', 'junto']
45
['ciudad']
46
['pesos']
47
['casa', 'partido']
48
['quiere']
49
['aires']
Finalizado (2017-09-21 11:17:24.158506)
```