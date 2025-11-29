from flask import Flask, request, jsonify

# create the Flask application
app = Flask(__name__)


# 1) Simple home page
@app.route("/")
def home():
    return "Hello from Flask! 🚀"


# 2) Tiny 'model' endpoint: returns the square of a number
# call: POST /square  with JSON: {"x": 3}
# or:   GET  /square?x=5  in browser
@app.route("/square", methods=["GET", "POST"])
def square_number():
    if request.method == "POST":
        data = request.get_json()          # read JSON from request body
        x = data.get("x", 0)               # get value 'x', default 0
    else:  # GET request
        x = request.args.get("x", 0, type=int)  # get from URL parameter
    result = x * x                     # our 'model': square the number
    return jsonify({"x": x, "square": result})


if __name__ == "__main__":
    # start the development server
    app.run(debug=True)