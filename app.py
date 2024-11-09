from gluoncv.model_zoo import get_model
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from PIL import Image
import io
import flask

app = flask.Flask(__name__)

# Cargar el modelo preentrenado fuera de la función
net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST" and flask.request.files.get("img"):
        try:
            # Leer y transformar la imagen
            img = Image.open(io.BytesIO(flask.request.files["img"].read())).convert('RGB')
            transform_fn = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

            # Convertir a un formato que MXNet puede procesar
            img = nd.array(img)
            img = transform_fn(img)

            # Hacer predicción
            pred = net(img.expand_dims(axis=0))
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
            ind = nd.argmax(pred, axis=1).astype('int')
            prediction = {
                "class": class_names[ind.asscalar()],
                "probability": float(nd.softmax(pred)[0][ind].asscalar())
            }

            return flask.jsonify(prediction)

        except Exception as e:
            # Manejo detallado de errores
            return flask.jsonify({"error": f"Processing error: {str(e)}"}), 400

    return flask.jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
