from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import MyModel
import os
import asyncio


app = Flask(__name__)
CORS(app)

model_instance = MyModel()


@app.route('/')
def home():
    os.system('cls')
    return render_template('index.html')


@app.route("/predict/", methods=['POST'])
async def predict():
    try:
        files = request.files.get('file')
        if not model_instance.get_model():
            loaded_model = await asyncio.shield(asyncio.create_task(model_instance.load_model(request.root_url)))

        prediction = await asyncio.shield(asyncio.create_task(model_instance.predict(files)))

        status = prediction[0]
        predicted_class = prediction[1]
        if status == "success":
            return jsonify({"status": status, "predicted_class": predicted_class})
        else:
            return jsonify({"status": status, "error": prediction[1]})

    except Exception as e:
        print("Error", str(e))
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    os.system("cls")
    app.run(debug=True, port=8080)
