import os
import string
import matplotlib
import matplotlib_inline
import pandas as pd
import gensim.models
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


matplotlib_inline
nltk.download('stopwords')
nltk.download('wordnet')
# ...................................................................................................................................
# ........... read documents.........................
c1= open("PETCLINIC.txt", "r")
c2 = open("autoparts.txt", "r")



# ...............................   pre processing data .....................................................................

#    stop words
stop_words = set(stopwords.words('english'))

# punctuation
punctuation = set(string.punctuation)

# lemmatization

lemmatization = WordNetLemmatizer()


# function cleaan
def clean(documents):
    # split documents and remove stop words

    split_doc = " ".join([i for i in documents.lower().split() if i not in stop_words])

    # remove punctuation
    punc_doc = ''.join([j for j in split_doc if j not in punctuation])

    # normalize the text
    normalized = " ".join([lemmatization.lemmatize(word) for word in punc_doc.split()])

    return normalized


# .............................................................................................................................................................
# clean documents in the file text

corpus = c2
clean_documents = [clean(doc) for doc in corpus]

# ............... Tables names ...................
Table_name = []

for doc in clean_documents:
    d = 0
    for word in doc.split():
        if d==0 :
           Table_name.append(word)
           d = d+1

# .............converting the list of documents into TF IDF vectors.................................................

tfidf_vectorizer = TfidfVectorizer(lowercase=False)

tf_idf = tfidf_vectorizer.fit_transform(clean_documents)

# ......... view ressult..........
print("\n TF IDF  :")
print(tf_idf)

feature_names = tfidf_vectorizer.get_feature_names()
dense = tf_idf.todense()
denselist = dense.tolist()

all_key_words = []

for d in denselist:
    i = 0
    kwords = []

    for word in d:
        if word > 0:
            kwords.append(feature_names[i])
        i = i + 1
    all_key_words.append(kwords)

print("\n key words \n")
i = 0
for key_w in all_key_words:
    print(i, key_w)
    i = i + 1

# .........................elbow method.................................................................................................

wcss = []  # WCSS Scores = within-cluster-sum of squared

clusters = range(1, 7)
for i in clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, algorithm="elkan")
    kmeans.fit(tf_idf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ..........................................................................................................................
# ...........             k means ............................................................................................
clusters = 5  # modified for Petclinc data will be 3 or 2

model = KMeans(n_clusters=clusters, init="k-means++", max_iter=200, n_init=10, algorithm="elkan")

model.fit(tf_idf)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = tfidf_vectorizer.get_feature_names()

labels = model.labels_

#print("labels :: ", labels)
# .....................Print Words belong to each clusters...............................................................................................................

# for i in range(n_clusters):
# print(f" \n Cluster {i} \n")
# for ind in order_centroids[i, :10]:
# print(terms[ind])

# ...............Obtained Clusters..........................................................................................
for c in range(clusters):
    print(" Cluster", c)
    print("__________________________________________")
    count = 0
    for i in labels:

        if i == c:
            print('Table: {} ---> Document {}'.format(Table_name[count], count))
        count = count + 1
    print("__________________________________________")
#

# ...................... visulization ....................................................................................
# kmeans_indices = model.fit_predict(tf_idf)

# print("indices ", kmeans_indices)

pca = PCA(n_components=2)

scatter_plot_points = pca.fit_transform(tf_idf.toarray())

colors = ["r", "b", "c", "y", "m"]

X_axe = [i[0] for i in scatter_plot_points]
Y_axe = [i[1] for i in scatter_plot_points]

plt.scatter(X_axe, Y_axe, c=[colors[d] for d in labels])
plt.show()
