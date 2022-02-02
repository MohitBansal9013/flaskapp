from flask import Flask

application = Flask(__name__)

@application.route('/')
def hello_world():
    	return 'Document Classification application'

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	application.run()

