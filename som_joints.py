import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from matplotlib.patches import Patch

data = pd.read_csv('Joints/joints.csv', dtype={'img': str})
X = data.iloc[:, :-1].values  # Features de los joints
labels = data.iloc[:, -1].values  # Etiquetas (números o nombres)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Configuración del SOM
som_x, som_y = 100, 100
som = MiniSom(som_x, som_y, X_scaled.shape[1], 
              sigma=3.0, learning_rate=0.5, 
              neighborhood_function='gaussian', 
              random_seed=42)

# Inicialización de pesos
#som.pca_weights_init(X_scaled)
som.random_weights_init(X_scaled)

print("Entrenando SOM...")
som.train_batch(X_scaled, 100000, verbose=True)
print("Entrenamiento completado.")

with open('nao_joints_som.pkl', 'wb') as f:
    pickle.dump(som, f)

plt.figure(figsize=(12, 12))

# Mapa de distancias U-Matrix
plt.subplot(1, 1, 1)
u_matrix = som.distance_map().T
plt.pcolor(u_matrix, cmap='bone_r')
plt.colorbar(label='Distancia entre neuronas (similitud)')

# Asignar etiquetas
for i, (x, label) in enumerate(zip(X_scaled, labels)):
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
cluster_labels = np.zeros(som_x * som_y)
for i, x in enumerate(X_scaled):
    w = som.winner(x)
    cluster_labels[w[0] * som_y + w[1]] += 1

plt.figure(figsize=(17, 17))
plt.pcolor(cluster_labels.reshape(som_x, som_y).T, cmap='viridis')
plt.colorbar(label='Frecuencia de activación')
plt.title("Clusters naturales en el SOM")
plt.show()
