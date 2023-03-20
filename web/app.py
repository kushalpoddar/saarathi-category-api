from flask import Flask
from flask_restful import Api, Resource, abort, reqparse
from flask_cors import CORS
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import cv2
import requests # request img from web
import shutil
import sys
import traceback

app = Flask(__name__)
CORS(app)
api = Api(app)

# class Asaview(Resource):
# 	def post(self):
# 		try:
# 			#Handling the file upload of PDB
# 			parse = reqparse.RequestParser()
# 			parse.add_argument('pdb_file', type=werkzeug.datastructures.FileStorage, location='files')
# 			parse.add_argument('pdb_code')
# 			args = parse.parse_args()
# 			image_file = args['pdb_file']
# 			pdb_code = args['pdb_code']

# 			if (pdb_code is None) and (image_file is None):
# 				return "Both Code & File cannot be empty"

# 			asadata = {}
# 			pdb_filepath = "temp/" + md5(str(time.time()).encode('utf-8')).hexdigest() + ".pdb"
# 			if image_file is not None:
# 				image_file.save(pdb_filepath)
# 			else:
# 				file_res = requests.get('http://files.rcsb.org/download/'+pdb_code+'.pdb')

# 				file_content = file_res.text
# 				with open(pdb_filepath, 'w') as f:
# 					f.write(file_content)

# 			#Loading the mapping for 3-letter to others
# 			asadata = asaData(pdb_filepath)

# 			return asadata
# 		except Exception as e:
# 			print(e)
			# return "Some error occured", 400

#Handler for protein

class ViTForImageClassification(nn.Module):
    def _init_(self, num_labels=3):
        super(ViTForImageClassification, self)._init_()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None


# Define Model
# model = ViTForImageClassification(len(train_ds.classes))    
# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Adam Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Cross Entropy Loss
# loss_func = nn.CrossEntropyLoss()
# Use GPU if available  

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device=torch.device('cpu')

# if torch.cuda.is_available():
#     model.cuda() 

model1 = torch.load("./modele25_d2aug.pt", map_location=torch.device('cpu'))
model1.eval()  

def predict(IMG_LINK,model1):
	print(IMG_LINK)
	# res = requests.get(IMG_LINK, stream = True)
	# print(res)
	# file_name="./img.png"
	# print(file_name)
	# if res.status_code == 200:
	# 	with open(file_name,'wb') as f:
	# 		shutil.copyfileobj(res.raw, f)
	print('Image sucessfully Downloaded: ',file_name)
	# else:
	# print('Image Couldn\'t be retrieved')
	file_name = "./Garbagehope.jpg"
	img=cv2.imread(file_name)
	print("file read")
	img=cv2.resize(img, (128,128))
	img = np.swapaxes(img, 2,0)
	img=np.expand_dims(img, axis=0)
	img = torch.from_numpy(img)
	# img.shape
	inputs = img[0].permute(1, 2, 0)
	# Save original Input
	originalInput = inputs
	for index, array in enumerate(inputs):
		inputs[index] = np.squeeze(array)
	inputs = torch.tensor(np.stack(feature_extractor(inputs)['pixel_values'], axis=0))
	inputs = inputs.to(device)
	t=np.array([3])
	target=torch.from_numpy(t)
	target = target.to(device)
	prediction, loss = model1(inputs,target)
	# print("{'garbage': 0, 'pothole': 1, 'sewage': 2, 'water': 3}")
	# print(prediction)
	predicted_class = np.argmax(prediction.cpu().detach().numpy())
	keys=['garbage', 'pothole', 'sewage', 'water']
	values=[0, 1, 2, 3]
	value_predicted = keys[values.index(predicted_class)]
	return value_predicted

class Protein(Resource):
	def get(self):
		try:
			return { "category" : predict("https://www.thecooldown.com/wp-content/uploads/2022/11/ffb531ab-1.jpeg", model1) }

		except Exception as e:
			app.logger.info("print exception", e)
			print(e, file=sys.stderr)
			print(traceback.format_exc(), file=sys.stderr)
			return e, 400

# api.add_resource(Asaview, '/classify')
api.add_resource(Protein, '/classify')

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=3201)