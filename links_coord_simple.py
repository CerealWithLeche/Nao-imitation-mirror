import os, inspect
import pybullet_data
import pybullet as p
import numpy as np
import csv
import pickle
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
Hacer links barriendo todo el som de los joints -> Ejecuta cada unidad del som de los joints y vinculalo con el BMU del som de las imagenes(Actualmente se vinculan el BMU de los joints
con el BMU de las img)
"""

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

joint_limits = np.array([[-1.3265, -0.3142, 0.0349066, -1.54462],  
                        [0.3142, 1.3265, 1.54462, -0.0349066]])

scaler = MinMaxScaler()
scaler.fit(joint_limits)

# < ----------- FUNCIONES DE SEGMENTACIÓN ----------------->
def arreglar_segmentation(segmentation):
    segmentation = np.array(segmentation)
    segmentation[segmentation == 0] = 1
    segmentation[segmentation == -1] = 0
    return segmentation

def segmentacion(nao, width, height, view_matrix, projection_matrix):
    _, _, _, _, segmentation = p.getCameraImage(
        width, height,
        view_matrix,
        projection_matrix,
        #renderer=p.ER_BULLET_HARDWARE_OPENGL
        renderer=p.ER_TINY_RENDERER
    )
    return arreglar_segmentation(segmentation)

def mover_joint(nao, joint_index, target_position, width, height, view_matrix, projection_matrix):
    p.resetJointState(nao, joint_index, target_position)
    p.stepSimulation()
    return segmentacion(nao, width, height, view_matrix, projection_matrix)

# < ----------- INICIALIZACIÓN DE PYBULLET ----------------->
# Cambiamos p.GUI por p.DIRECT para eliminar la ventana del simulador
physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# < ----------- LLAMADAO A URDF E INCIALIZACIÓN PREVIA DE JOINTS ----------------->
nao = p.loadURDF("/home/luis-ignacio-zamora/Documents/nao/nao.urdf", [0, 0, 0], useFixedBase=True)
p.resetJointState(nao, 56, targetValue=1.5)
p.resetJointState(nao, 39, targetValue=1.5)
p.resetJointState(nao, 41, targetValue=-2.08) 
p.resetJointState(nao, 58, targetValue= 2.08)   
p.stepSimulation() 
p.changeDynamics(nao, -1, mass=0)

# < ----------- CARGAR SOMS DE JOINTS E IMAGEN ----------------->
with open('nao_joints_som.pkl', 'rb') as f:
    som_joint = pickle.load(f)

with open('nao_img_som.pkl', 'rb') as f:
    som_img = pickle.load(f)

# < ----------- PARÁMETROS DE LA CÁMARA ----------------->
width, height = 128, 128
camera_pos = [0.5, 0, .15]
target_pos = [0, 0, 0.1]
up_vector = [0, 0, .1]
view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.02, 1.0)
np.set_printoptions(precision=3, suppress=True)


# < ----------- INICIO DE BALBUCEO MOTRÍZ ----------------->
links_wc_simple = defaultdict(list)
links_simple = defaultdict(list)

            #57: RShoulderRoll      (-1.32645, 0.314159) 
            #40: LShoulderRoll      (-0.314159, 1.32645)
            #59: RElbowRoll         (0.0349066, 1.54462) 
            #42: LElbowRoll         (-1.54462, -0.0349066)
            
            #39: LShoulderPitch     (-2.08567, 2.08567)
            #41: LElbowYaw          (-2.08567, 2.08567)

for i in range(som_joint.get_weights().shape[0]):      
    for j in range(som_joint.get_weights().shape[1]):  
        pos = [i, j]
        
        print(f"Neurona en posición ({j},{i})")
        
        neuron_weights = som_joint.get_weights()[i, j]
        joint_desnormalized = scaler.inverse_transform(neuron_weights.reshape(1, -1))[0]  
        print(f"Unidad desnormalizada: {joint_desnormalized}")
        
        mover_joint(nao, 57, joint_desnormalized[0], width, height, view_matrix, projection_matrix)
        mover_joint(nao, 40, joint_desnormalized[1], width, height, view_matrix, projection_matrix)
        mover_joint(nao, 59, joint_desnormalized[2], width, height, view_matrix, projection_matrix)
        segmentacion_result = mover_joint(nao, 42, joint_desnormalized[3], width, height, view_matrix, projection_matrix)
    
        img_flattened = segmentacion_result.flatten()
        winner_img = som_img.winner(img_flattened)
        winner_img_coords = (int(winner_img[0]), int(winner_img[1]))
        
        #key = tuple(winner_img_coords)
        key = tuple(pos)
        #key = tuple(float(x) for x in joint_desnormalized)
        
        print(f"Coordenada de BMU img: {winner_img_coords}")
        
        # Extraer valores numericos 
        bmu_img = som_img.get_weights()[winner_img_coords]
        
        
        # Transformar en 1s y 0s la BMU de img y agregarla al diccionario
        binary_repr = (bmu_img > 0.5).astype(int)  
        
        #if key not in links_wc_simple :
        #    links_wc_simple[key] = winner_img_coords
            
        if key not in links_simple :
            links_simple[key] = winner_img_coords
            print(f"Key: {key}, Value: {winner_img_coords}")
        
        #Calcular error de segmentación con distancia Hamming
        diferentes = (img_flattened != binary_repr)
        hamming = np.sum(diferentes.astype(int))
        
        #print(f"Error de segmentación con distancia hamming de img:{hamming/16384}")
        #print(f"Total de pixeles diferentes:{hamming}")
        print(f"Tamaño de diccionario: {len(links_simple)}")
        print("--------------------------\n")
        
    """ if i == 1:
            break
    else: 
        continue
    break """
        
# < ----------- GUARDAR DICIONARIO ----------------->
file_path_links_simple  = "diccionario_simple.pkl"
with open(file_path_links_simple , 'wb') as f: 
    pickle.dump(links_simple , f, protocol=pickle.HIGHEST_PROTOCOL)
    
#file_path_links_wc_simple  = "diccionario_wc_simple.pkl"
#with open(file_path_links_wc_simple , 'wb') as f: 
#    pickle.dump(links_wc_simple , f, protocol=pickle.HIGHEST_PROTOCOL)
#print(len(links_wc_simple))