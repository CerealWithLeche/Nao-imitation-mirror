import os
import gc  
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
import pickle

# 1. CARGAR LAS REPRESENTACIONES LATENTES DEL AUTOENCODER (DIMENSIÓN 64)
if not os.path.exists('joint_images_latent_64.npy'):
    raise FileNotFoundError("Primero debes ejecutar el script del Autoencoder para generar 'joint_images_latent_64.npy'")

X_visual_latent = np.load('joint_images_latent_64.npy')
print(f"Dataset visual latente cargado. Forma: {X_visual_latent.shape}") # Debería ser (10000, 64)

# 2. CONFIGURACIÓN DEL SOM VISUAL CON EXPANSIÓN GLOBAL
som_x, som_y = 100, 100
sigma_inicial = 30.0       # Radio amplio para que el mapa se expanda hacia la periferia
lr_inicial = 0.5           # Ajuste fuerte inicial

# El tercer parámetro ahora es X_visual_latent.shape[1], que equivale a 64 dimensiones
som = MiniSom(som_x, som_y, X_visual_latent.shape[1], 
              sigma=sigma_inicial, learning_rate=lr_inicial, 
              neighborhood_function='gaussian', 
              random_seed=42)

# Inicialización PCA para alinear el mapa con las direcciones de mayor varianza visual
som.pca_weights_init(X_visual_latent)

# --- CONFIGURACIÓN DE MÉTRICAS Y DECAIMIENTO DINÁMICO ---
num_epochs = 50
iterations_per_epoch = 1000  # Total de 50,000 iteraciones

# Parámetros de validación por submuestra para evitar el error 'Killed' por RAM
eval_fraction = 0.20  # Evaluamos usando solo el 20% de los datos al azar
num_eval_samples = int(len(X_visual_latent) * eval_fraction)
np.random.seed(42)

quantization_errors = []
topographic_errors = []
epochs_evaluated = []

print(f"Entrenando SOM Visual. Muestras totales: {len(X_visual_latent)}. Muestras de validación por bloque: {num_eval_samples}")
print("Entrenando con decaimiento dinámico de Sigma y Learning Rate...")

for epoch in range(num_epochs):
    # 1. Calcular el decaimiento exponencial para la época actual
    current_sigma = sigma_inicial * np.exp(-epoch / (num_epochs / np.log(sigma_inicial / 1.0)))
    current_lr = lr_inicial * np.exp(-epoch / (num_epochs / np.log(lr_inicial / 0.01)))
    
    # 2. Inyectar dinámicamente los parámetros calculados en la red
    som.sigma = current_sigma
    som.learning_rate = current_lr
    
    # 3. Entrenar la porción correspondiente a esta época
    som.train_random(X_visual_latent, iterations_per_epoch, verbose=False)
    
    # 4. Calcular métricas de manera espaciada utilizando la submuestra
    if (epoch + 1) % 5 == 0 or epoch == 0:
        indices_eval = np.random.choice(len(X_visual_latent), num_eval_samples, replace=False)
        sub_samples = X_visual_latent[indices_eval]
        
        q_error = som.quantization_error(sub_samples)
        t_error = som.topographic_error(sub_samples)
        
        quantization_errors.append(q_error)
        topographic_errors.append(t_error)
        epochs_evaluated.append(epoch + 1)
        
        print(f"Época {epoch+1:02d}/{num_epochs} | Sigma: {current_sigma:.2f} | LR: {current_lr:.3f} | Q-Error (Sub): {q_error:.4f} | T-Error (Sub): {t_error:.4f}")
        
        # Limpieza explícita de RAM
        del sub_samples
        gc.collect()

print("Entrenamiento completado.")

# 3. GUARDAR EL SOM VISUAL CON ENTRENAMIENTO BIEN DISTRIBUIDO
with open('nao_img_som.pkl', 'wb') as f:
    pickle.dump(som, f)
print("Modelo visual guardado exitosamente como 'nao_img_som.pkl'")

# --- GRÁFICAS DE CONVERGENCIA ---
plt.figure(figsize=(10, 4))
plt.plot(epochs_evaluated, quantization_errors, label='Error de Cuantificación', color='blue', marker='o', lw=2)
plt.axhline(y=min(quantization_errors), color='gray', linestyle='--', alpha=0.7)
plt.title('Evolución del Error de Cuantificación (SOM Visual)')
plt.xlabel('Épocas')
plt.ylabel('Error promedio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(epochs_evaluated, topographic_errors, label='Error Topológico', color='orange', marker='o', lw=2)
plt.title('Evolución del Error Topológico (SOM Visual)')
plt.xlabel('Épocas')
plt.ylabel('Proporción de errores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- REJILLA DE ACTIVACIÓN (HIT MAP O CLUSTERS NATURALES) ---
frequencies = som.activation_response(X_visual_latent)
plt.figure(figsize=(10, 9))
plt.pcolor(frequencies.T, cmap='viridis') 
plt.colorbar(label='Número de veces que la neurona fue ganadora (Hits)')
plt.title(f"Rejilla de Activación (Hit Map Visual) - Tamaño {som_x}x{som_y}")
plt.xlim([0, som_x])
plt.ylim([0, som_y])
plt.gca().invert_yaxis()
plt.grid(True, which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.2)
plt.show()