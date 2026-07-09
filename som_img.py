import os
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from matplotlib.gridspec import GridSpec

data = pd.read_csv('Joints/joints.csv', dtype={'img': str})
labels = data.iloc[:, -1].values

def load_joint_images(label_list, img_dir='Joints/'):
    images = []
    for label in label_list:
        img_path = os.path.join(img_dir, f"{label}")
        if os.path.exists(img_path):
            img_data = np.load(img_path)
            img_data = img_data.reshape((128, 128))
            #img_data_list = img_data.tolist()  
            img_flattened = img_data.flatten()
            images.append(img_flattened)
        else:
            print(f"Advertencia: Archivo no encontrado - {img_path}")
            images.append(np.zeros((128, 128)))  # Mantener forma 2D para el placeholder
    return np.array(images)

joint_images = load_joint_images(labels)

# Configuración del SOM

############## VERIFICAR SI INPUT DIM 0 - 1

som_x, som_y = 100, 100
som = MiniSom(som_x, som_y,joint_images.shape[1], 
              sigma=3.0, learning_rate=0.5, 
              neighborhood_function='gaussian', 
              random_seed=42)

# Inicialización de pesos
#som.pca_weights_init(X_scaled)
som.random_weights_init(joint_images)
###som._weights = np.random.uniform(low=0, high=1, size=(som_x, som_y, input_dim))

print("Entrenando SOM...")
som.train_batch(joint_images, 10000, verbose=True)
print("Entrenamiento completado.")

with open('nao_img_som.pkl', 'wb') as f:
    pickle.dump(som, f)
"""
plt.figure(figsize=(15, 15))

# Mapa de distancias U-Matrix
plt.subplot(1, 1, 1)
u_matrix = som.distance_map().T
plt.pcolor(u_matrix, cmap='bone_r')
plt.colorbar(label='Distancia entre neuronas (similitud)')

# Asignar etiquetas
for i, (x, label) in enumerate(zip(joint_images, labels)):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, str(label),
             color='darkred', ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.title(f"Mapa auto-organizado de joints NAO\nTamaño: {som_x}x{som_y}")
plt.xlim([0, som_x])
plt.ylim([0, som_y])
plt.gca().invert_yaxis()  
plt.grid(False)
plt.tight_layout()
plt.show()

# Visualización por clusters naturales
from matplotlib.patches import Patch
# Calcular clusters 
cluster_labels = np.zeros(som_x * som_y)
for i, x in enumerate(joint_images):
    w = som.winner(x)
    cluster_labels[w[0] * som_y + w[1]] += 1

plt.figure(figsize=(15, 15))
plt.pcolor(cluster_labels.reshape(som_x, som_y).T, cmap='viridis')
plt.colorbar(label='Frecuencia de activación')
plt.title("Clusters naturales en el SOM")
plt.show() """

weights = som.get_weights()  # Shape: (som_x, som_y, input_dim)

# Dimensiones de la imagen original (ajusta según tus datos)
img_height, img_width = 128, 128
channels = 1  # 1 para escala de grises, 3 para RGB

# Remodelar los pesos a imágenes
som_images = weights.reshape(som_x, som_y, img_height, img_width, channels)

# Visualización
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(som_x, som_y, fig)

for i in range(som_x):
    for j in range(som_y):
        ax = fig.add_subplot(gs[i, j])
        img = som_images[i, j]
        if channels == 1:
            ax.imshow(img[:, :, 0], cmap='viridis')  # Escala de grises
        else:
            ax.imshow(img)  # RGB
        ax.axis('off')
        ax.set_title(f'({i},{j})', fontsize=6)

plt.suptitle("Vectores de pesos del SOM remodelados como imágenes", y=0.92)
plt.tight_layout()
plt.show()