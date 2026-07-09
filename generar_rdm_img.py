import os, inspect
import pybullet_data
import pybullet as p
import numpy as np
import csv
import matplotlib.pyplot as plt

# Directorio actual del archivo
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def save_joints(joint_list, filename='joints.csv'):
    joints_dir = os.path.join(currentdir, "Joints")
    os.makedirs(joints_dir, exist_ok=True)
    filepath = os.path.join(joints_dir, filename)
    write_header = not os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["40", "42", "57", "59", "img"])
            #writer.writerow(["57", "40", "42", "59", "img"])
            #56: RShoulderPitch     (-2.08567, 2.08567)
            #57: RShoulderRoll      (-1.32645, 0.314159) 
            #58: RElbowYaw          (-2.08567, 2.08567)
            #59: RElbowRoll         (0.0349066, 1.54462) 
            
            #39: LShoulderPitch     (-2.08567, 2.08567)
            #40: LShoulderRoll      (-0.314159, 1.32645) -> (0, 1.32645)
            #41: LElbowYaw          (-2.08567, 2.08567)
            #42: LElbowRoll         (-1.54462, -0.0349066)
            
            #round(np.random.uniform(-1.3265, 0.3142), 6),       #57
            ##round(np.random.uniform(-0.3142, 1.3265), 6),       #40
            ##round(np.random.uniform(0.0349066, 1.54462), 6),    #59
            #round(np.random.uniform(-1.54462, -0.0349066), 6)   #42
            
        writer.writerow(joint_list)

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
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    return arreglar_segmentation(segmentation)

def mover_joint(nao, joint_index, target_position, width, height, view_matrix, projection_matrix):
    p.resetJointState(nao, joint_index, target_position)
    p.stepSimulation()
    return segmentacion(nao, width, height, view_matrix, projection_matrix)

  
def delta(minimo, maximo, paso=0.0174):
    opciones = np.arange(minimo, maximo, paso)
    opciones = np.round(opciones, 3)
    return np.random.choice(opciones)


# Inicialización de PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

nao = p.loadURDF("/home/luis-ignacio-zamora/Documents/nao/nao.urdf", [0, 0, 0], useFixedBase=True)
p.resetJointState(nao, 56, targetValue=1.5)
p.resetJointState(nao, 39, targetValue=1.5)
p.resetJointState(nao, 41, targetValue=-2.08) 
p.resetJointState(nao, 58, targetValue= 2.08)  
p.stepSimulation() 
p.changeDynamics(nao, -1, mass=0)

# Parámetros de cámara
width, height = 128, 128 #16,384
camera_pos = [0.5, 0, .15]
target_pos = [0, 0, 0.1]
up_vector = [0, 0, .1]
view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.02, 1.0)

fotos = 10000  
np.set_printoptions(precision=3, suppress=True)

while fotos != 0:
    #57: RShoulderRoll      (-1.32645, 0.314159)        #40: LShoulderRoll      (-0.314159, 1.32645)
    #59: RElbowRoll         (0.0349066, 1.54462)        #42: LElbowRoll         (-1.54462, -0.0349066)
    """
    40     | LShoulderRoll        | Revolute     | -0.314159    | 1.32645      | 1.78       | 7.19
    42     | LElbowRoll           | Revolute     | -1.54462     | -0.0349066   | 1.53       | 7.19
    57     | RShoulderRoll        | Revolute     | -1.32645     | 0.314159     | 1.78       | 7.19
    59     | RElbowRoll           | Revolute     | 0.0349066    | 1.54462      | 1.53       | 7.19
    """
    """
    rnd_joint = [
        delta(0, 1.3265),               # 40
        delta(-1.5446, -0.0349)         # 42
        delta(-1.3265, 0),              # 57
        delta(0.0349, 1.5446),          # 59
    ]
    """
    rnd_joint = [
        delta(-0.3141, 1.3264),         # 40
        delta(-1.5446, -0.0349),        # 42
        delta(-1.3265, 0.3141),         # 57
        delta(0.0349, 1.5446),          # 59
    ]

    # Mueve cada articulación
    #mover_joint(nao, 40, rnd_joint[0], width, height, view_matrix, projection_matrix)
    #mover_joint(nao, 42, rnd_joint[1], width, height, view_matrix, projection_matrix)
    #mover_joint(nao, 57, rnd_joint[2], width, height, view_matrix, projection_matrix)
    #segmentacion1 = mover_joint(nao, 59, rnd_joint[3], width, height, view_matrix, projection_matrix)

    p.resetJointState(nao, 40, rnd_joint[0])
    p.resetJointState(nao, 42, rnd_joint[1])
    p.resetJointState(nao, 57, rnd_joint[2])
    p.resetJointState(nao, 59, rnd_joint[3])
    
    # Procesamos la física una sola vez
    p.stepSimulation()
    
    # CORREGIDO: Una única captura de cámara por ciclo (mucho más rápido)
    segmentacion1 = segmentacion(nao, width, height, view_matrix, projection_matrix)

    filename = f"Posición{fotos}.npy"
    
    # Agrega el nombre del archivo al final del registro
    save_joints(rnd_joint + [filename])
    
    # Guarda la segmentación
    np.save(os.path.join(currentdir, "Joints", filename), segmentacion1)
    
    fotos -= 1
    print(f"Step: {fotos}")
    
    
    print("Forma de segmentacion1:", segmentacion1.shape)
    print("Tipo de datos:", segmentacion1.dtype)
    
print("Joints guardados")
