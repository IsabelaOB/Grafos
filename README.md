# Enterprise Graph Machine Learning Pipeline: Financial Fraud Detection

##  Arquitectura del Proyecto (Software Design Patterns)
Siguiendo principios de ingeniería de software y separación de responsabilidades (Clean Architecture), el proyecto está estructurado por capas desacopladas:

```text
graph_enterprise_project/
│
├── data/
│   ├── raw_nodes.csv          # Datos tabulares individuales de las cuentas/usuarios
│   └── raw_edges.csv          # Conexiones o transacciones financieras entre nodos
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Capa de Ingesta y generación automática de datos
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py        # Motor Topológico: PageRank, Clustering y Embeddings Espectrales
│   └── models/
│       ├── __init__.py
│       └── evaluator.py       # Pipeline de ML: Validación cruzada y comparación Baseline vs Graph ML
│
├── main.py                    # Orquestador maestro (Controller / Entrypoint)
├── requirements.txt
└── README.md

## Metodología y Valor Analítico:
Este sistema resuelve el problema de detección de anomalías/fraude financiero combinando variables tabulares tradicionales con propiedades topológicas de teoría de redes:

Ingeniería de Redes: Cálculo de PageRank, Coeficiente de Clustering local, grado de conexiones y Spectral Graph Embeddings (8 dimensiones latentes).

Validación Cruzada Estricta: Uso de StratifiedKFold para evitar el sesgo por desbalance de clases (~7% de anomalías).

Threshold Tuning: Optimización del umbral de decisión basada en el análisis de curvas Precision-Recall para maximizar el F1-Score operativo.

Experimento Comparativo: Contraste cuantitativo entre un modelo tradicional (solo datos tabulares) y un modelo enriquecido con Graph Machine Learning (Tabular + Red), demostrando la mejora en el ROC-AUC.



Instrucciones de Instalación y Ejecución:

1. Clonar el repositorio e instalar dependencias
Bash
git clone [https://github.com/tu-usuario/graph_enterprise_project.git](https://github.com/tu-usuario/graph_enterprise_project.git)
cd graph_enterprise_project
pip install -r requirements.txt

2. Ejecutar el pipeline completo
El sistema cuenta con un gestor automático: si no detecta los archivos en la carpeta data/, generará un escenario financiero simulado de forma transparente al correr el orquestador:

Bash
python main.py