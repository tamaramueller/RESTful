from flask import (
    Flask,
    render_template,
    jsonify,
    abort,
    request
)
import os
import requests
import urllib
import json
import sys
from read_movie_data import sentimentAnalysis

# Create the application instance
app = Flask(__name__, template_folder="templates")

@app.route('/Emoji', methods=['GET', 'POST'])
def emoji():
    return render_template('Emoji.html')

@app.route('/Emoji/hello', methods=['GET', 'POST'])
def hello():
    user_input = request.form['text']
   
    #favourite emoji works so far only with exactly these inputs, this should definitely be changed
    if "My favourite emoji is" in user_input:
        #save favourite emoji
        favourite_emoji = "".join(user_input.split("My favourite emoji is "))
        if os.path.isfile("favourite_emoji.txt"):
            f = open("favourite_emoji.txt", "w")
            f.write(favourite_emoji)
            return render_template('greeting.html', say="OK! I deleted your last favourite emoji and saved this one!")
        else:
            f = open("favourite_emoji.txt", "w")
            f.write(favourite_emoji)
            return render_template('greeting.html', say="OK!")

    elif "What is my favourite emoji" in user_input:
        #return favourite emoji
        if os.path.isfile("favourite_emoji.txt"):
            with open("favourite_emoji.txt", 'r') as myfile:
                favourite_emoji = myfile.read()
            ret_str = "Your favourite emoji is " + favourite_emoji
            return render_template('greeting.html', say=ret_str)
        else:
            return render_template('greeting.html', say="You haven't told me your favourite emoji yet")



    else:
        #normal interaction
        #option one: use existing sentiment analysis service
        r = requests.post(url="http://text-processing.com/api/sentiment/", data = {'text':user_input})
        data2 = r.json()
        #response is stored at label
        label = data2["label"]
        #comment the following lines to use option two
        if label=="pos":
            return render_template('greeting.html', say=":-)")
        elif label =="neg":
            return render_template('greeting.html', say=":-(")
        else:
            return render_template('greeting.html', say=":-|")

        #option two: train model based on film review data in module read_movie_data.py
        #uncomment the following lines to use option two

        #retval = sentimentAnalysis(request.form['text'])

        #if retval==1:
        #    return render_template('greeting.html', say=":-)")
        #else:
        #    return render_template('greeting.html', say=":-(")

if __name__ == '__main__':
    app.run(debug=True)