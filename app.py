from flask import Flask, jsonify, make_response, render_template, request, make_response
import json
import uuid
import os
import random

app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = False
app._static_folder = os.path.abspath("templates/static/")

@app.route("/", methods=["GET"])
def index():
	#Load the main page of interface
	return render_template("layouts/index.html")

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
	jsdata = request.form['txt_data']
	txtstring = json.loads(jsdata)

	#Note: we can write to a file and process somewhere else
	# with open("code.txt", "w") as file:
	# 	file.write(txtstring)
	# print ("returning js data:", jsdata)



	#call model to compute language string....
	#write result to language.txt
	#example language.txt = ace/mode/python
	#randomly pick to check simple functionality
	r =  random.randint(0, 1)
	if r == 0:
		language_txt = "ace/mode/python"
	else:
		language_txt = "ace/mode/javascript"
	with open("language.txt", "w") as file:
		file.write(language_txt)

	return jsdata

@app.route('/getmethod')
def get_javascript_data():
	#supported list
	#https://cloud9-sdk.readme.io/docs/language-mode
	with open("language.txt", "r") as file:
		langtext = file.read()
	#it is a simple string no special characters so no need to json it
	print ("selecting language:", langtext)
	return langtext

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)