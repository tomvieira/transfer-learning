from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# loaded_model = load_model("hist_model_inception.keras")
import warnings
warnings.filterwarnings("ignore")
vgg_model = load_model("hist_model_vgg.h5")

from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (400, 400, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('dataset/test/ANORMAL/c01-2001849-1-400-003.jpg')


pred1 = vgg_model.predict(image)
#print(pred1)

pred2 = load("dataset-unitopatho/ANORMAL/54-B2-TAHG.ndpi_ROI__mpp0.44_reg000_crop_sk00000_(73992,7343,1812,1812).png")
pred2 = vgg_model.predict(pred2)
print(pred2)