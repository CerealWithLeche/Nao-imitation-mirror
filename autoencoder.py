import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 1. DEFINICIÓN DEL AUTOENCODER CONVOLUCIONAL
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Comprime de 1x128x128 a un vector plano de 'latent_dim'
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Flatten(),                                          # -> 16384
            nn.Linear(64 * 16 * 16, latent_dim)                    # -> Espacio Latente (64)
        )
        
        # Decoder: Reconstruye el vector latente de vuelta a 1x128x128
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),                         # -> 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 16 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 1 x 128 x 128
            nn.Sigmoid() # Escala la salida final estrictamente entre 0 y 1 (binario/silueta)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# 2. DATASET PERSONALIZADO PARA ARCHIVOS .NPY
class JointImagesDataset(Dataset):
    def __init__(self, csv_file, img_dir='Joints/'):
        data = pd.read_csv(csv_file, dtype={'img': str})
        # La última columna tiene el identificador/nombre de la pose
        self.labels = data.iloc[:, -1].values
        self.img_dir = img_dir
        self.images = self._load_all_images()

    def _load_all_images(self):
        loaded_images = []
        print(f"Cargando {len(self.labels)} imágenes desde '{self.img_dir}' en memoria...")
        
        for filename in self.labels:
            filename_str = str(filename).strip()
            
            # Construcción directa de la ruta usando el nombre completo del CSV
            img_path = os.path.join(self.img_dir, filename_str)
            
            if os.path.exists(img_path):
                img_data = np.load(img_path).reshape(128, 128).astype(np.float32)
                # Normalizar a rango [0, 1] si los píxeles vienen de 0 a 255
                if img_data.max() > 1.0:
                    img_data /= 255.0
            else:
                print(f"Advertencia: Archivo no encontrado - {img_path}")
                img_data = np.zeros((128, 128), dtype=np.float32)
                
            loaded_images.append(img_data)
            
        return np.array(loaded_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (Canales, Alto, Ancho) -> (1, 128, 128) para las capas Conv2D de PyTorch
        img = np.expand_dims(self.images[idx], axis=0)
        return torch.tensor(img, dtype=torch.float32)

# 3. PIPELINE DE ENTRENAMIENTO
if __name__ == '__main__':
    # Configuración de hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    dataset = JointImagesDataset(csv_file='Joints/joints.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Inicializar Red, Pérdida y Optimización
    latent_dim = 64
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy es excelente para siluetas/segmentación binaria
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Bucle de entrenamiento
    num_epochs = 20
    print("Iniciando entrenamiento del Autoencoder...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(device)
            
            # Forward
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_imgs)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_imgs.size(0)
            
        epoch_loss = train_loss / len(dataset)
        print(f"Época [{epoch+1:02d}/{num_epochs}] | Pérdida de Reconstrucción: {epoch_loss:.5f}")

    # Guardar los pesos del modelo entrenado
    torch.save(model.state_dict(), 'conv_autoencoder.pth')
    print("Modelo guardado como 'conv_autoencoder.pth'")

    # 4. EXTRACCIÓN Y EXPORTACIÓN DEL ESPACIO LATENTE PARA EL SOM
    print("Generando representaciones compactas (embeddings de dimensión 64)...")
    model.eval()
    all_embeddings = []
    
    # DataLoader sin barajar para mantener el orden estricto del CSV original
    export_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch_imgs in export_dataloader:
            batch_imgs = batch_imgs.to(device)
            embeddings = model.encoder(batch_imgs)
            all_embeddings.append(embeddings.cpu().numpy())
            
    # Concatenar todos los lotes en una matriz final de forma (N, 64)
    joint_images_latent = np.concatenate(all_embeddings, axis=0)
    
    # Guardar en un archivo binario npy listo para cargar directo en tu script del SOM
    np.save('joint_images_latent_64.npy', joint_images_latent)
    print(f"Dataset reducido exportado exitosamente. Nueva forma: {joint_images_latent.shape}")