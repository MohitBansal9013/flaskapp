import json
import utill
from flask import Flask, jsonify, request, render_template
import Stage_NLP
import plotly.express as px
from flask_cors import CORS
import pandas as pd
from DCException import DCException

# Flask constructor takes the name of
# current module (__name__) as argument.
application = Flask(__name__,static_folder='storage',template_folder="templates")
cors = CORS(application)
application.config['CORS_HEADERS'] = 'Content-Type'

class DataStore():
    temp_tags = []


dataStore = DataStore()

# The route() function of the Flask class is a decorator,
# which tells the applicationlication which URL should call
# the associated function.
@application.route('/')
def hello_world():
	return 'Hello World from Document Classification App'

@application.route('/tags', methods=['GET'])
def fetch_tags():
# JSON file
    f = open ('./storage/tags.json', "r")
 
    # Reading from file
    tags = json.loads(f.read())
    return jsonify({'data':sorted(tags), 'status':'success','statuscode':200})

@application.route('/wordCloudByCluster', methods=['GET'])
def fetch_wordCloud_by_cluster_id():
    cluster_id = request.args.get("clusterID")
    worldCloud = Stage_NLP.getWordcloud(int(cluster_id))
    
    return jsonify({'data':worldCloud, 'status':'success','statuscode':200})


@application.route('/initialCluster', methods=['GET'])
def fetch_cluster_information():
    cluster_info = Stage_NLP.getInitialClusterInformation()
    json_object = json.dumps([])
    avg_match_prob_file_path = "./storage/avg_match_probabilities.json"
    utill.write_file(json_object,avg_match_prob_file_path)
    return jsonify({'cluster_info': cluster_info, 'status': 'success', 'statuscode': 200})


@application.route('/taggedClusters', methods=['POST'])
def getTaggedClusters():
    cluster_id = request.form.get('clusterID')
    tag = request.form.get('tag')
    
    trigger = Stage_NLP.triggerTagFunctions(int(cluster_id), tag)
    cluster_information = Stage_NLP.getTaggedClusterInfo()
    
    df = pd.read_json('./storage/avg_match_probabilities.json')
    _df = df.rename(columns={df.columns[0]:"Average Match Probability"})
    _df['Iterations'] = df.index+1
    graph_path = './storage/output/graph.png'
    fig = px.line(data_frame=_df,y="Average Match Probability",x='Iterations')
    fig.update_traces(mode='markers+lines',line_color='#cc001e')
    fig.write_image(graph_path)
    graph_base64 = utill.get_base64_encoded_image(graph_path)

    return jsonify({'cluster_info': cluster_information, 'graph': graph_base64,
                    'status': 'success', 'statuscode': 200})

@application.route('/add_tag', methods=['POST'])
def add_tag():
    tag = ''
    tags = []
    tag = request.form.get('tag')
    sanitized_tag = tag.strip()
    # JSON file
    tags_file_path = './storage/tags.json'
    f = open(tags_file_path, "r")
    # Reading from file
    tags = json.loads(f.read())
    f.close()
    if len(sanitized_tag) <1:
        raise DCException("Tag Name is required", statuscode=400)

    tags.extend(dataStore.temp_tags)

    if sanitized_tag not in tags:
        dataStore.temp_tags.applicationend(sanitized_tag)
        tags.extend(dataStore.temp_tags)
    else:
        raise DCException("Tag already exists", statuscode=400)

    return jsonify({'data': sorted(tags), 'status': 'success', 'statuscode': 200})

@application.errorhandler(DCException)
def invalid_api_usage(e):
    return jsonify(e.to_dict())

@application.route('/launchFrontend', methods=['GET'])
def runFrontend():
    return render_template('index.html')
# main driver function
#if __name__ == '__main__':

# run() method of Flask class runs the applicationlication
# on the local development server.
#	application.run()
