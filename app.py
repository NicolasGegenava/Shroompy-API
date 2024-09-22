from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
import os
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import cv2
from dotenv import load_dotenv
import gdown

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": os.getenv("SECRET_FRONT")}})

SECRET_API_KEY = os.getenv("SECRET_API_KEY")
SECRET_API_MODEL = os.getenv("SECRET_API_MODEL")

output = 'model.keras'
if not os.path.exists(output):
    model_url = SECRET_API_MODEL
    gdown.download(model_url, output, quiet=False)
    
model = tf.keras.models.load_model(output)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

request_times = {}

def preprocess_image(image):
    image = Image.open(io.BytesIO(image)).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    api_key = request.headers.get('x-api-key')
    if api_key != SECRET_API_KEY:
        return jsonify({'error': f'Unauthorized access'}), 403
    
    ip_address = request.remote_addr
    current_time = time.time()

    if ip_address in request_times:
        last_request_time = request_times[ip_address]
        if current_time - last_request_time < 30:
            return jsonify({'error': f'Wait for {int(30 - (current_time - last_request_time))} seconds'}), 429 

    request_times[ip_address] = current_time

    mushrooms = [
'Amanita citrina',
'Amanita muscaria',
'Amanita pantherina',
'Amanita rubescens',
'Apioperdon pyriforme',
'Armillaria borealis',
'Artomyces pyxidatus',
'Bjerkandera adusta',
'Boletus edulis',
'Boletus reticulatus',
'Calocera viscosa',
'Calycina citrina',
'Cantharellus cibarius',
'Cerioporus squamosus',
'Cetraria islandica',
'Chlorociboria aeruginascens',
'Chondrostereum purpureum',
'Cladonia fimbriata',
'Cladonia rangiferina',
'Cladonia stellaris',
'Clitocybe nebularis',
'Coltricia perennis',
'Coprinellus disseminatus',
'Coprinellus micaceus',
'Coprinopsis atramentaria',
'Coprinus comatus',
'Crucibulum laeve',
'Daedaleopsis confragosa',
'Daedaleopsis tricolor',
'Evernia mesomorpha',
'Evernia prunastri',
'Flammulina velutipes',
'Fomes fomentarius',
'Fomitopsis betulina',
'Fomitopsis pinicola',
'Ganoderma applanatum',
'Graphis scripta',
'Gyromitra esculenta',
'Gyromitra gigas',
'Gyromitra infula',
'Hericium coralloides',
'Hygrophoropsis aurantiaca',
'Hypholoma fasciculare',
'Hypholoma lateritium',
'Hypogymnia physodes',
'Imleria badia',
'Inonotus obliquus',
'Kuehneromyces mutabilis',
'Lactarius deliciosus',
'Lactarius torminosus',
'Lactarius turpis',
'Laetiporus sulphureus',
'Leccinum albostipitatum',
'Leccinum aurantiacum',
'Leccinum scabrum',
'Leccinum versipelle',
'Lepista nuda',
'Lobaria pulmonaria',
'Lycoperdon perlatum',
'Macrolepiota procera',
'Merulius tremellosus',
'Mutinus ravenelii',
'Nectria cinnabarina',
'Panellus stipticus',
'Parmelia sulcata',
'Paxillus involutus',
'Peltigera aphthosa',
'Peltigera praetextata',
'Phaeophyscia orbicularis',
'Phallus impudicus',
'Phellinus igniarius',
'Phellinus tremulae',
'Phlebia radiata',
'Pholiota aurivella',
'Pholiota squarrosa',
'Physcia adscendens',
'Platismatia glauca',
'Pleurotus ostreatus',
'Pleurotus pulmonarius',
'Pseudevernia furfuracea',
'Rhytisma acerinum',
'Sarcomyxa serotina',
'Sarcoscypha austriaca',
'Sarcosoma globosum',
'Schizophyllum commune',
'Stereum hirsutum',
'Stropharia aeruginosa',
'Suillus granulatus',
'Suillus grevillei',
'Suillus luteus',
'Trametes hirsuta',
'Trametes ochracea',
'Trametes versicolor',
'Tremella mesenterica',
'Trichaptum biforme',
'Tricholomopsis rutilans',
'Urnula craterium',
'Verpa bohemica',
'Vulpicida pinastri',
'Xanthoria parietina',
    ]
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'File could not be read'}), 500
        final_input = cv2.resize(image, (224, 224)) 
        final_input = np.expand_dims(final_input, axis=0)

        try:
            prediction = model.predict(final_input)
            predicted_class = np.argmax(prediction[0])
            return jsonify({'prediction': mushrooms[predicted_class]}), 200
        except Exception as e:
            return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)