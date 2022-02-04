from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np
from PIL import Image

def most_similar(doc_id,similarity_matrix,distance,data):
    print (f'Document: {data.iloc[doc_id]["processed"]}')
    print (f'Document: {data.iloc[doc_id]["raw"]}')
    print ('\n')
    print ('Similar Documents:')
    if distance=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif distance=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print('\n')
        print (f'Document: {data.iloc[ix]["processed"]}')
        print(ix)
        print (f'Document: {data.iloc[ix]["raw"]}')
        print (f'{distance} : {similarity_matrix[doc_id][ix]}')


def get_base64_encoded_image(image_path):
    # # Resizing: Opens a image in RGB mode
    # im = Image.open(image_path,'r')
    # newsize = (200,211)
    # im = im.resize(newsize)
    # im.save(image_path)
    # im.close()
    # Encoding:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def write_file(data,path):
    # Writing to path
    with open(path, "w") as outfile:
        outfile.write(data)
        outfile.close()
    return True