from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, accuracy_score
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")
vgg_model = load_model("hist_model_vgg.h5")


test_path = 'ebhi-split-2categorias/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(400, 400),
                                            batch_size=32,
                                            shuffle=False,
                                            class_mode='binary',classes=['ANORMAL','NORMAL'])


t = time.time()
# Usando o modelo para predição das amostras de teste
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
plt.savefig('Matriz-vgg16')
plt.show()

print ('Accuracy score: ', accuracy_score(y_test, aux))
print('Acuracia obtida com o Vgg16 no Conjunto de Teste EBHI: {:.2f}'.format(
    acc))

