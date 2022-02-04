# STAGE NLP : PARAGRAPH TAGGING

# STEP 1: Document processing and word embedding
from typing import final
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import sys
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import utill
from wordcloud import WordCloud
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import math
import json
import threading
nltk.download("punkt")

data = pd.read_json("./storage/doc.json")
detech_clusters : final = [73, 40, 72] 
# Tokenize all the paragraph of the FAA documentation
tokenized_sent = []
for s in data.processed:
    tokenized_sent.append(word_tokenize(s.lower()))

# Using the Doc2vec embedding method in order to have a vector representation of each paragrah
# tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
#
# model = Doc2Vec(vector_size=300, alpha=0.025, min_count=1)
# model.build_vocab(tagged_data)
# print("Line Number 34", model)
# for epoch in range(2):
#     model.train(tagged_data, total_examples=model.corpus_count,
#                 epochs=model.epochs)


# b- Embedding with sentenceBERT
# model = SentenceTransformer('paraphrase-mpnet-base-v2')
# sentence_embeddings = model.encode(list(data['processed']))
# embeddings = sentence_embeddings
#
# # c- Document clustering
#
# # Agglomerative clustering
#
# for i in range(len(sentence_embeddings)):
#     # Normalised all my vectors obtained by my embedding method
#     sentence_embeddings[i] = sentence_embeddings[i] / np.linalg.norm(sentence_embeddings[i])
#     # data["embeddings"][i] = sentence_embeddings[i]
#
# clusterer = AgglomerativeClustering(
#     n_clusters=80, affinity="euclidean", linkage="ward")
# clusters = clusterer.fit_predict(sentence_embeddings)
# cluster_labels = clusterer.labels_
# data["clusters"] = clusters
# print(clusters)
# print(cluster_labels)
#
# data.to_json('data_clust2.json', orient='records')
# df = pd.read_json("data_clust2.json")
# clusters = df['clusters']

def getCluster(df):
    '''
     return a list which elements represent ours clusters and each element contain all the paragraph
     that belong to the corresponding cluster.
    '''
    cluster_list = []
    for cluster in range(80):
        pgrh = ''
        for index in range(len(df)):
            if df[df.index == index]['clusters'].iloc[0] == cluster:
                pgrh = pgrh + " " + df[df.index == index]['text'].iloc[0]
        cluster_list.append(pgrh)

    return cluster_list

def run():
    print("Starting Thread")
    global sim_matrix
    global df
    global clusters
    global aggregated_df
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    sentence_embeddings = model.encode(list(data['processed']))
    embeddings = sentence_embeddings

    # c- Document clustering

    # Agglomerative clustering

    for i in range(len(sentence_embeddings)):
        # Normalised all my vectors obtained by my embedding method
        sentence_embeddings[i] = sentence_embeddings[i] / np.linalg.norm(sentence_embeddings[i])
        # data["embeddings"][i] = sentence_embeddings[i]

    clusterer = AgglomerativeClustering(
        n_clusters=80, affinity="euclidean", linkage="ward")
    clusters = clusterer.fit_predict(sentence_embeddings)
    cluster_labels = clusterer.labels_
    data["clusters"] = clusters
    print(clusters)
    print(cluster_labels)

    data.to_json('data_clust2.json', orient='records')
    df = pd.read_json("data_clust2.json")
    cluster_list = getCluster(df)

    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    cluster_embeddings = model.encode(cluster_list)

    sim_matrix = cosine_similarity(cluster_embeddings)
    aggregated_df = df
    aggregated_df = aggregated_df.drop_duplicates(subset ="clusters")
    print("Thread completed")
thread = threading.Thread(target=run)
thread.start()

def tagClusterbyId(df, index, tag):
    '''
    Input:
     df = my dataframe
     index = index of one paragraph in the cluster we want to tag
     tag = tag we want to associate to the cluster
   '''
    # get the cluster to which the paragraph belong
    cluster = index
    print(cluster)
    #     for i in range(len(df)):
    #         if df[df.index == i]['clusters'].iloc[0] == cluster:
    #             df.loc[df.index == i,"label"] = tag
    #             df.loc[df.index == i,"match_probability"] = 1

    df.loc[df.clusters == cluster, "label"] = tag
    df.loc[df.clusters == cluster, "match_probability"] = 1


def tagNotTagCluster(df, cluster, tag, dist_min):
    '''
      method used in Aour algorithm to tag nottagged cluster
    '''
    df.loc[df.clusters == cluster, "label"] = tag
    df.loc[df.clusters == cluster, "match_probability"] = dist_min


def TagAllClusters(df, sim_matrix, submitted_cluster_ID):
    '''
     Main algorithm used to tag all not tagged clusters
    '''
    tag = df[df.clusters == submitted_cluster_ID]['label'].iloc[0]
    for k in range(80):
        if df[df.clusters == k]['match_probability'].iloc[0] == 1.0:
            continue
        distance = sim_matrix[k,:]
        index_tab = np.argsort(distance).tolist()
        for cluster in reversed(index_tab):
            if df[df.clusters == cluster]['match_probability'].iloc[0] == 0.0:
                continue
            if distance[cluster] >= 0.9:
                continue
            dist_min = distance[cluster]

            print(tag)
            tagNotTagCluster(df,k,tag,dist_min)
            # Tuple (current cluster, nearest cluster to the current cluster)
            print((k,cluster))
            break


def triggerTagFunctions(ID, tag):
    tagClusterbyId(aggregated_df, ID, tag)
    print("TagClusterByID completed")
    distance_dict = sim_matrix[ID, :]
    TagAllClusters(aggregated_df, sim_matrix, ID)


def getTaggedClusterInfo():
    cluster_information = []
    total_match_probability = 0
    for cl in range(0, 80):
        if cl not in detech_clusters:
            temp_df_agg = aggregated_df.loc[aggregated_df['clusters'] == cl]
            temp_df = df.loc[df['clusters'] == cl]
            para = temp_df['processed'].tolist()
            match_probability = round(temp_df_agg['match_probability'].tolist()[0] * 100)
            tags = temp_df_agg['label'].tolist()[0]
            if match_probability < 100:
                manually_tagged = 'No'
            else:
                manually_tagged = 'Yes'
            cluster_information.append({'id': cl, 'tags': tags, 'number_of_paragraphs': len(para),
                                        'match_probability': match_probability,
                                        'manually_tagged': manually_tagged,
                                        'paragraphs': para})
    agg_df = aggregated_df
    # agg_df = agg_df.drop_duplicates(subset ="clusters")
    current_MP = round((agg_df['match_probability'].sum() / 80) * 100)
    print("Current=",current_MP)
    avg_match_probabilities = []
    avg_match_prob_file_path = "./storage/avg_match_probabilities.json"
    # Reading from file
    f = open(avg_match_prob_file_path, "r")
    avg_match_probabilities = json.loads(f.read())
    f.close()
    avg_match_probabilities.append(current_MP)
    # Serializing json
    json_object = json.dumps(avg_match_probabilities)
    utill.write_file(json_object, avg_match_prob_file_path)

    print(cluster_information)

    return cluster_information


def getInitialClusterInformation():
    global aggregated_df
    aggregated_df = df
    aggregated_df = aggregated_df.drop_duplicates(subset ="clusters")

    data_cluster_ag = pd.DataFrame({"document": list(data['processed']), "cluster": clusters})
    cluster_info = []
    for cl in range(0, 80):
        if cl not in detech_clusters: 
            temp_df = data_cluster_ag.loc[data_cluster_ag['cluster'] == cl]
            para = temp_df['document'].tolist()
            cluster_info.append({'id': cl, 'tags': '', 'number_of_paragraphs': len(para), 'match_probability': '',
                                 'manually_tagged': 'No',
                                 'paragraphs': para})
    return cluster_info


def getWordcloud(k):
    result = {'cluster': clusters, 'document': list(data['processed'])}
    result = pd.DataFrame(result)
    s = result[result.cluster == k]
    text = s['document'].str.cat(sep=' ')
    text = text.lower()
    text = ' '.join([word for word in text.split()])
    wordcloud = WordCloud(max_font_size=100, max_words=100,
                          background_color="white").generate(text)

    print('Cluster: {}'.format(k))
    image_path = './storage/output/wordcloud.png'
    wordcloud.to_file(image_path)
    return utill.get_base64_encoded_image(image_path)
    # print('Titles')
    # titles = data_cluster_ag[data_cluster_ag.cluster == k]['document']
    # print(titles.to_string(index=False))
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
