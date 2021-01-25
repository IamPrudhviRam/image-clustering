import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil

input_dir = 'raw-img'

glob_dir = input_dir + '/*.jpeg'  #trial
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]  #os.listdir(glo)
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255)
# print(images)
model = tf.keras.applications.MobileNetV2(include_top=False,
weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)
print("pred_images",pred_images)


k = 3
kmodel = KMeans(n_clusters = k, n_jobs=-1, random_state=728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('output')

for i in range(k):
    os.makedirs("output\cluster" + str(i))

for i in range(len(paths)):
    shutil.copy2(paths[i], "output\cluster"+str(kpredictions[i]))

sil = []
kl = []
kmax = 5
for k in range(2, kmax + 1):
    kmeans2 = KMeans(n_clusters=k).fit(pred_images)
    labels = kmeans2.labels_
    sil.append(silhouette_score(pred_images, labels, metric='euclidean'))
    kl.append(k)

print(sil)
plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('K')
plt.show()
