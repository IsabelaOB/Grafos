import matplotlib.pyplot as plt
import networkx as nx
from src.data.loader import DataLoader
from src.features.engineer import GraphFeatureEngineer
from src.models.evaluator import ModelEvaluator

def main():
    print(">>> [1/4] Inicializando DataLoader (Verificando o generando CSVs)...")
    loader = DataLoader(data_dir="data")
    df_nodes, df_edges = loader.generate_or_load_data()
    
    print(">>> [2/4] Ejecutando ingeniería de características topológicas (Graph ML)...")
    engineer = GraphFeatureEngineer(n_components=8)
    df_enriched, G = engineer.build_features(df_nodes, df_edges)
    df_enriched.to_csv("data/dataset_procesado_final.csv", index=False)
    print("    [OK] Dataset enriquecido exportado a 'data/dataset_procesado_final.csv'.")
    
    print(">>> [3/4] Ejecutando evaluación comparativa de modelos (Baseline vs Graph ML)...")
    evaluator = ModelEvaluator(n_splits=5)
    evaluator.evaluate_comparison(df_enriched)
    
    print(">>> [4/4] Renderizando visualización de la red transaccional...")
    plt.figure(figsize=(10, 7))
    labels = nx.get_node_attributes(G, 'target')
    color_map = ['crimson' if labels.get(node, 0) == 1 else 'skyblue' for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k=0.15)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=40, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="gray")
    plt.title("Arquitectura Enterprise - Red Transaccional Multidimensional", fontsize=12)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()