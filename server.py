import read
from flask import Flask, jsonify, request
from flask_cors import CORS

#Flask is for server
#Jsonify is for returning data as a json
# CORS (Cross-Origin Resource Sharing), without this html will
# cause errors when flask server tries to connect with web
# extension of a different address

#This initializes web server
app = Flask(__name__)

# This sets up CORS, see above
CORS(app)


# Define the API route '/get-data'
# Full route is http://127.0.0.1:5000/get-data by default
# This is what fetchData.js connects to
@app.route('/get-data', methods=['post'])
def get_data():

    data = request.get_json()
    clientUrl = data.get('url')
    if not clientUrl:
        return jsonify({"message": "Error: No URL was provided"}), 400

    #This is where function call will go to detect bias
    prediction_result = read.predict_url(clientUrl)
    prediction_string = prediction_result[0] + ": " + str(round(100 * prediction_result[1], 2)) + "%"

    # Create a Python dictionary to send back as JSON
    response_data = {
        "message": prediction_string,
        #"site_url_used": clientUrl
    }



    # Return the data as a JSON response
    # Flask's jsonify() turns the Python dict into a
    # JSON response that JavaScript can read
    return jsonify(response_data)



# Runs the app
if __name__ == '__main__':
    # 'debug=True' makes the server auto-reload when changes are saved
    app.run(port=5000, debug=True)
