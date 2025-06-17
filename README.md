# Kohonen 3D Network - Multi-Dataset Neural Visualization

Una implementación completa de una Red Neuronal Auto-Organizativa (SOM - Self-Organizing Map) de Kohonen en 3D con visualización interactiva en OpenGL. Soporta **MNIST**, **Fashion-MNIST** y **AfroMNIST** con métricas detalladas de evaluación y visualización dual.

## 🎯 Características Principales

### 🧠 **Red Neuronal Kohonen 3D**
- **Red auto-organizativa tridimensional** (6×6×6 = 216 neuronas)
- **Algoritmo de entrenamiento completo** con BMU (Best Matching Unit)
- **Función de vecindad gaussiana** con decaimiento temporal
- **Auto-organización topológica** preservada en 3D

### 📊 **Soporte para 3 Datasets**
| Dataset | Tipo | Descripción | Formato |
|---------|------|-------------|---------|
| **MNIST** | Dígitos | Números escritos a mano (0-9) | `.ubyte` binario |
| **Fashion-MNIST** | Ropa | Prendas de vestir (10 categorías) | `.ubyte` binario |
| **AfroMNIST** | Caracteres | Caracteres etíopes (10 símbolos) | `.npy` NumPy |

### 🎮 **Visualización 3D Interactiva**
- **🎨 Modo Colores**: Esferas coloreadas por clase dominante
- **🖼️ Modo Imágenes**: Visualización de prototipos aprendidos en las esferas
- **🎛️ Controles intuitivos**: Rotación, zoom, alternancia de modos
- **📐 Cubo wireframe**: Delimitador del espacio neuronal 3D

### 📈 **Métricas de Evaluación Completas**
- **Exactitud (Accuracy)** general
- **Matriz de confusión** detallada por dataset
- **Precisión, Recall y F1-Score** por clase
- **Promedios macro** de todas las métricas
- **Reportes automáticos** guardados en archivos

## 🛠️ Instalación

### Dependencias (Arch Linux)

```bash
# Paquetes principales
sudo pacman -S cmake gcc freeglut

# Librerías de desarrollo
sudo pacman -S mesa glu git
```

### Clonar y Preparar el Proyecto

```bash
# Clonar el repositorio
git clone <repository-url>
cd kohonen3d

# Crear estructura de directorios
mkdir -p data build
```

### Estructura del Dataset

```
data/
├── train-images-idx3-ubyte               # MNIST
├── train-labels-idx1-ubyte               
├── t10k-images-idx3-ubyte                
├── t10k-labels-idx1-ubyte                
├── fashion-train-images-idx3-ubyte       # Fashion-MNIST
├── fashion-train-labels-idx1-ubyte       
├── fashion-t10k-images-idx3-ubyte        
├── fashion-t10k-labels-idx1-ubyte        
├── Ethiopic_MNIST_X_train.npy            # AfroMNIST
├── Ethiopic_MNIST_y_train.npy            
├── Ethiopic_MNIST_X_test.npy             
└── Ethiopic_MNIST_y_test.npy             
```

## 🚀 Compilación y Ejecución

```bash
# Compilar
cd build
cmake ..
make -j$(nproc)

# Ejecutar
cd ..
./build/Kohonen3D
```

## 🎮 Uso del Programa

### 1. **Selección de Dataset**
Al ejecutar, aparece el menú de selección:
```
Select dataset for training:
1. MNIST (digits 0-9)
2. Fashion-MNIST (clothing items)  
3. AfroMNIST (Ethiopic characters)
Enter choice (1, 2, or 3):
```

### 2. **Proceso Automático**
El programa ejecuta automáticamente:
- ✅ **Carga** del dataset seleccionado
- ✅ **Inicialización** de la red Kohonen 3D
- ✅ **Entrenamiento** por épocas con progreso
- ✅ **Evaluación** en dataset de prueba
- ✅ **Métricas** completas mostradas
- ✅ **Visualización** 3D interactiva

### 3. **Controles de Visualización**

| Control | Acción |
|---------|--------|
| **🖱️ Mouse drag** | Rotar cámara alrededor del cubo |
| **⌨️ SPACE** | Alternar: colores ↔ imágenes |
| **⬆️⬇️ Flechas** | Mover cámara verticalmente |
| **⬅️➡️ Flechas** | Mover cámara horizontalmente |
| **➕ +** | Zoom in (acercar) |
| **➖ -** | Zoom out (alejar) |
| **🚪 ESC** | Salir del programa |

## 🎨 Modos de Visualización

### 🔴 **Modo Colores**
Cada esfera muestra un color que representa la clase dominante que aprendió esa neurona:

#### 🔢 MNIST (Dígitos)
| Color | Dígito | Color | Dígito |
|-------|--------|-------|--------|
| 🔴 **Rojo** | 0 | 🔵 **Cyan** | 5 |
| 🟢 **Verde** | 1 | 🟠 **Naranja** | 6 |
| 🔵 **Azul** | 2 | 🟣 **Púrpura** | 7 |
| 🟡 **Amarillo** | 3 | 🩷 **Rosa** | 8 |
| 🟣 **Magenta** | 4 | ⚫ **Gris** | 9 |

#### 👕 Fashion-MNIST (Ropa)
| Color | Prenda | Color | Prenda |
|-------|--------|-------|--------|
| 🔴 **Rojo oscuro** | T-shirt/top | 🟡 **Amarillo** | Sandal |
| 🔵 **Azul oscuro** | Trouser | 🟢 **Verde** | Shirt |
| 🟣 **Púrpura** | Pullover | ⚪ **Blanco** | Sneaker |
| 🩷 **Rosa** | Dress | 🤎 **Marrón claro** | Bag |
| 🤎 **Marrón** | Coat | ⚫ **Negro** | Ankle boot |

#### 🇪🇹 AfroMNIST (Caracteres Etíopes)
| Color | Carácter | Pronunciación | Color | Carácter | Pronunciación |
|-------|----------|---------------|-------|----------|---------------|
| 🟡 **Dorado** | ሀ | ha | 🔵 **Azul cielo** | ሰ | sa |
| 🟢 **Verde brillante** | ለ | le | 🟢 **Lima** | ሸ | sha |
| 🟠 **Naranja quemado** | ሐ | hha | 🟠 **Naranja brillante** | ቀ | qe |
| 🔵 **Azul real** | መ | me | 🤎 **Tierra** | በ | be |
| 🟣 **Púrpura profundo** | ሠ | se | | | |

### 🖼️ **Modo Imágenes**
Cada esfera muestra la imagen prototipo que la neurona aprendió:
- **MNIST**: Dígitos 0-9 como píxeles amarillos/naranjas
- **Fashion-MNIST**: Siluetas de prendas de vestir
- **AfroMNIST**: Caracteres etíopes como píxeles coloridos

## 📊 Métricas y Reportes

### Métricas Calculadas
- **Accuracy**: Porcentaje de clasificaciones correctas
- **Confusion Matrix**: Matriz NxN mostrando predicciones vs realidad
- **Per-class Metrics**: 
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **F1-Score**: Media harmónica de precision y recall
- **Macro Averages**: Promedios no ponderados de todas las clases

### Archivos Generados
- `MNIST_classification_report.txt`
- `Fashion-MNIST_classification_report.txt`  
- `AfroMNIST_classification_report.txt`

### Ejemplo de Salida (AfroMNIST)
```
========================================
       CLASSIFICATION METRICS REPORT
========================================
Dataset: AfroMNIST
Overall Accuracy: 84.32%

=== CONFUSION MATRIX ===
True\Pred    ሀ (ha)  ለ (le)  ሐ (hha) መ (me)  ሠ (se)
ሀ (ha)         41      2       1      0      1
ለ (le)          1     47       0      1      0
ሐ (hha)         2      0      44      1      2
...

=== PER-CLASS METRICS ===
Class           Precision    Recall      F1-Score
ሀ (ha)             0.8936    0.9111      0.9023
ለ (le)             0.9588    0.9592      0.9590
ሐ (hha)            0.8980    0.8980      0.8980
...
```

## ⚙️ Configuración Avanzada

### Parámetros de la Red (en `main.cpp`)
```cpp
const int networkWidth = 10;        // Dimensión X de la red (6x6x6)
const int networkHeight = 10;       // Dimensión Y de la red
const int networkDepth = 10;        // Dimensión Z de la red
const int trainingEpochs = 100;     // Número de épocas de entrenamiento
const int maxTrainingSamples = 1000; // Límite de muestras de entrenamiento
const int maxTestSamples = 500;    // Límite de muestras de prueba
```

### Hiperparámetros del Algoritmo (en `KohonenNetwork.cpp`)
```cpp
// Decaimiento exponencial del learning rate
float learningRate = 0.5f * std::exp(-epoch / (epochs / 3.0f));

// Decaimiento exponencial del radio de vecindad
float neighborhoodRadius = std::max(width, std::max(height, depth)) / 2.0f * 
                          std::exp(-epoch / (epochs / 3.0f));
```

## 🔬 Aspectos Técnicos

### Algoritmo de Kohonen Implementado
1. **Inicialización**: Pesos aleatorios uniformes [0,1]
2. **Competencia**: Búsqueda de BMU por distancia euclidiana
3. **Cooperación**: Función de vecindad gaussiana 3D
4. **Adaptación**: Actualización de pesos con decaimiento temporal
5. **Clasificación**: Asignación de clase dominante post-entrenamiento

### Auto-Organización 3D
- **Preservación topológica**: Neuronas vecinas aprenden patrones similares
- **Especialización regional**: Formación de clusters por similitud
- **Mapeo no-lineal**: Proyección de alta dimensión (784D) a 3D
- **Vecindad tridimensional**: Influencia espacial en todas las direcciones

### Cargadores de Datos Personalizados
- **Formato binario (.ubyte)**: Para MNIST y Fashion-MNIST
- **Formato NumPy (.npy)**: Para AfroMNIST con cargador personalizado
- **Normalización automática**: Todos los píxeles a rango [0,1]
- **Validación de dimensiones**: Verificación de tamaño 28x28

## 🏗️ Arquitectura del Código

### Componentes Principales
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MNISTLoader   │────│ KohonenNetwork  │────│    Metrics      │
│ (Carga datos)   │    │ (Red neuronal)  │    │ (Evaluación)    │
│                 │    │                 │    │                 │
│ • .ubyte        │    │ • Algoritmo SOM │    │ • Accuracy      │
│ • .npy          │    │ • BMU search    │    │ • Confusion     │
│ • Normalización │    │ • Neighborhood  │    │ • Precision/Rec │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │    Renderer     │────│      main       │
                       │ (Visualización) │    │ (Coordinador)   │
                       │                 │    │                 │
                       │ • OpenGL/GLUT   │    │ • Flujo control │
                       │ • 2 modos vista │    │ • Selección UI  │
                       │ • Interactividad│    │ • Reportes      │
                       └─────────────────┘    └─────────────────┘
```

### Archivos del Proyecto
```
kohonen3d/
├── src/
│   ├── main.cpp              # Programa principal y coordinación
│   ├── KohonenNetwork.cpp    # Implementación del algoritmo SOM
│   ├── KohonenNetwork.h      # Definiciones de la red neuronal
│   ├── MNISTLoader.cpp       # Cargador multi-formato de datos
│   ├── MNISTLoader.h         # Interface para carga de datasets
│   ├── Renderer.cpp          # Sistema de visualización OpenGL
│   ├── Renderer.h            # Controles y renderizado 3D
│   ├── Metrics.cpp           # Cálculo de métricas de evaluación
│   ├── Metrics.h             # Estructuras de reportes
│   ├── NpyLoader.cpp         # Cargador personalizado .npy
│   └── NpyLoader.h           # Parser de archivos NumPy
├── data/                     # Datasets (descargados por usuario)
├── build/                    # Archivos de compilación
├── CMakeLists.txt           # Configuración de build
└── README.md                # Esta documentación
```


### Tecnologías y Librerías
- **C++17**: Lenguaje principal con STL moderno
- **OpenGL/GLUT**: Renderizado 3D de alto rendimiento  
- **CMake**: Sistema de construcción multiplataforma
- **Cargador .npy personalizado**: Sin dependencias externas

### Capturas
#### Fashion
![fashion](https://github.com/user-attachments/assets/01dc6fc5-d644-4411-81b0-0fc6d31f9463)

#### MINIST
![mnist](https://github.com/user-attachments/assets/c125ebce-8d2a-4ab7-afa9-c90ea9129d1d)

#
