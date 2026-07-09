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
            img_flattened = img_data.flatten()
            images.append(img_flattened)
        else:
            print(f"Advertencia: Archivo no encontrado - {img_path}")
            images.append(np.zeros(128 * 128))  # Ajustado a vector plano para consistencia
    return np.array(images)

joint_images = load_joint_images(labels)

scaler = MinMaxScaler()
joint_images = scaler.fit_transform(joint_images)

# Configuración del SOM
som_x, som_y = 100, 100
sigma_inicial = 30
lr_inicial = 5
som = MiniSom(som_x, som_y, joint_images.shape[1], 
              sigma=sigma_inicial, learning_rate=lr_inicial, 
              neighborhood_function='gaussian', 
              random_seed=42)

# Inicialización de pesos
som.pca_weights_init(joint_images)

# --- CONFIGURACIÓN DE MÉTRICAS POR ÉPOCAS ---
num_epochs = 10
iterations_per_epoch = 100  # total = 10,000 iteraciones (50 * 200)

quantization_errors = []
topographic_errors = []

print("Entrenando SOM y calculando métricas...")
for epoch in range(num_epochs):
    # 1. Calcular el decaimiento exponencial para la época actual
    # Al final de num_epochs, sigma se aproximará a 1.0 y lr a 0.01
    current_sigma = sigma_inicial * np.exp(-epoch / (num_epochs / np.log(sigma_inicial / 1.0)))
    current_lr = lr_inicial * np.exp(-epoch / (num_epochs / np.log(lr_inicial / 0.01)))
    
    # 2. Inyectar dinámicamente los nuevos parámetros calculados en la red
    som.sigma = current_sigma
    som.learning_rate = current_lr
    
    # 3. Entrenar la porción correspondiente a esta época
    som.train_random(joint_images, iterations_per_epoch, verbose=False)
    
    # 4. Cálculo de métricas para monitorear convergencia
    q_error = som.quantization_error(joint_images)
    t_error = som.topographic_error(joint_images)
    
    quantization_errors.append(q_error)
    topographic_errors.append(t_error)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1:02d}/{num_epochs} | Sigma: {current_sigma:.2f} | LR: {current_lr:.3f} | Q-Error: {q_error:.4f} | T-Error: {t_error:.4f}")

print("Entrenamiento completado.")

with open('nao_img_som.pkl', 'wb') as f:
    pickle.dump(som, f)

# --- PLOT 1: MÉTRICAS DE APRENDIZAJE ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(quantization_errors, label='Error de Cuantificación', color='blue', lw=2)
ax[0].set_title('Evolución del Error de Cuantificación')
ax[0].set_xlabel('Época')
ax[0].set_ylabel('Error')
ax[0].grid(True, alpha=0.3)

ax[1].plot(topographic_errors, label='Error Topológico', color='orange', lw=2)
ax[1].set_title('Evolución del Error Topológico')
ax[1].set_xlabel('Época')
ax[1].set_ylabel('Error')
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- PLOT 2: REJILLA DE ACTIVACIÓN (HIT MAP) ---
frequencies = som.activation_response(joint_images) # Devuelve matriz de tamaño (som_x, som_y)
plt.figure(figsize=(10, 9))
plt.pcolor(frequencies.T, cmap='viridis') 
plt.colorbar(label='Número de veces que la neurona fue ganadora (Hits)')
plt.title(f"Rejilla de Activación (Hit Map) - Tamaño {som_x}x{som_y}")
plt.xlim([0, som_x])
plt.ylim([0, som_y])
plt.gca().invert_yaxis()
plt.grid(True, which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.2)
plt.show()

# --- PLOT 3: VECTORES DE PESOS (Submureado a 20x20 por rendimiento) ---
weights = som.get_weights()  # (100, 100, 16384)
img_height, img_width = 128, 128
channels = 1 

som_images = weights.reshape(som_x, som_y, img_height, img_width, channels)

# NOTA: Para evitar crash de memoria por crear 10,000 subplots en Matplotlib,
# tomamos una sub-muestra del mapa (por ejemplo cada 5 neuronas: matriz de 20x20).
step = 5
sub_x, sub_y = som_x // step, som_y // step

fig = plt.figure(figsize=(18, 18))
gs = GridSpec(sub_x, sub_y, fig)

for i in range(sub_x):
    for j in range(sub_y):
        ax = fig.add_subplot(gs[i, j])
        # Mapeamos los índices correspondientes al salto de paso
        img = som_images[i * step, j * step]
        
        ax.imshow(img[:, :, 0], cmap='viridis')
        ax.axis('off')

plt.suptitle(f"Muestra de pesos del SOM como imágenes (Sub-rejilla {sub_x}x{sub_y})", y=0.92, fontsize=16)
plt.tight_layout()
plt.show()