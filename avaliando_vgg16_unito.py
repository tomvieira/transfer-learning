from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, accuracy_score
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# loaded_model = load_model("hist_model_inception.keras")
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
from skimage import transform
#funcao carregar imagem
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (400, 400, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

vgg_model = load_model("hist_model_vgg.h5")

pred2 = load("dataset-unitopatho/ANORMAL/54-B2-TAHG.ndpi_ROI__mpp0.44_reg000_crop_sk00000_(73992,7343,1812,1812).png")
pred2 = vgg_model.predict(pred2)
print(np.where(pred2 > 0.5, 'Normal', 'Anormal').flatten())


test_path = 'dataset-unitopatho/'
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(400, 400),
                                            batch_size=32,
                                            shuffle=False,
                                            class_mode='binary',classes=['ANORMAL','NORMAL'])



aux = vgg_model.predict(test_set)
# Reset
test_set.reset()
loss, acc = vgg_model.evaluate(test_set)
#aux = np.argmax(aux, axis=1)
aux = np.where(aux > 0.5, 1, 0).flatten()
print("y predito:")
print(aux)
y_test = test_set.classes
print("y real:")
print(y_test)
# Método para calcular o valor F1-Score
print('F1-Score: {}'.format(f1_score(y_test, aux, average='macro')))
# Método para calcular a Precision
print('Precision : {}'.format(precision_score(y_test, aux, average='macro')))
# Método para calcular o Recall
print('Recall: {}'.format(recall_score(y_test, aux, average='macro')))


print('Matriz de Confusão:')
cm = confusion_matrix(y_test, aux)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Anormal','Normal'])
cm_display.plot()
plt.savefig('Matriz-vgg16-UNITOPATHO')
plt.show()

print ('Accuracy score: ', accuracy_score(y_test, aux))
print('Acuracia obtida com o Vgg16 no Conjunto de Teste UNITOPATHO: {:.2f}'.format(
    acc))

