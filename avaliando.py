from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# loaded_model = load_model("hist_model_inception.keras")
import warnings
warnings.filterwarnings("ignore")
resnet_model = load_model("hist_model_inception.keras")


test_path = 'ebhi-split-2categorias/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(400, 400),
                                            batch_size=32,
                                            class_mode='categorical')

# filenames = test_set.filenames
# nb_samples = len(filenames)
print(test_set)

t = time.time()
# Usando o modelo para predição das amostras de teste
aux = resnet_model.predict(test_set)
loss, acc = resnet_model.evaluate(test_set)
print(aux)
aux = np.argmax(aux, axis=1)
y_test = test_set.classes
# Método para calcular o valor F1-Score
print('F1-Score: {}'.format(f1_score(y_test, aux, average='macro')))
# Método para calcular a Precision
print('Precision : {}'.format(precision_score(y_test, aux, average='macro')))
# Método para calcular o Recall
print('Recall: {}'.format(recall_score(y_test, aux, average='macro')))
# Salvando as acurácias nas listas
# resnet_model.score(X_train, y_train)
# resnet_model.score(X_test, y_test)

print('Matriz de Confusão:')
cm = confusion_matrix(y_test, aux)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.show()

# acc_train = resnet_model.score(X_train, y_train)
# print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Treinamento: {:.2f}'.format(acc_train[0]))
print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Teste: {:.2f}'.format(
    acc))
