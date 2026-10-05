import networkx as nx
import pandas as pd
import numpy as np
from sklearn.manifold import SpectralEmbedding

class GraphFeatureEngineer:
    """Capa encargada de transformar la topología de la red en variables cuantitativas (Feature Store)."""
    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def build_features(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> pd.DataFrame:
        # 1. Construir grafo con NetworkX
        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            G.add_node(row['node_id'], target=row['target'])
            
        for _, row in df_edges.iterrows():
            if G.has_node(row['source']) and G.has_node(row['target']):
                if G.has_edge(row['source'], row['target']):
                    G[row['source']][row['target']]['monto'] += row['monto']
                else:
                    G.add_edge(row['source'], row['target'], monto=row['monto'])

        # 2. Calcular Métricas Topológicas Clave (Teoría de Redes)
        pagerank = nx.pagerank(G, alpha=0.85)
        clustering = nx.clustering(G)
        degree = dict(G.degree())

        # 3. Calcular Graph Embeddings (Spectral Embedding sobre matriz de adyacencia)
        # Asegurar orden consistente de nodos
        nodes_list = list(G.nodes())
        sub_df = df_nodes.set_index('node_id').loc[nodes_list].reset_index()
        
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes_list)
        # Manejo defensivo por si hay componentes desconectados grandes
        n_comp = min(self.n_components, adj_matrix.shape[0] - 1) if adj_matrix.shape[0] > 1 else 1
        n_comp = max(n_comp, 1)
        
        se = SpectralEmbedding(n_components=n_comp, affinity='precomputed', random_state=42)
        embeddings = se.fit_transform(adj_matrix)

        # 4. Consolidar el Dataset Enriquecido (Tabular + Topológico)
        df_features = pd.DataFrame(embeddings, columns=[f'embed_dim_{i}' for i in range(n_comp)])
        df_features['node_id'] = nodes_list
        df_features['pagerank'] = [pagerank[n] for n in nodes_list]
        df_features['clustering'] = [clustering[n] for n in nodes_list]
        df_features['degree'] = [degree[n] for n in nodes_list]

        # Fusionar con los atributos tabulares originales del CSV
        df_final = pd.merge(sub_df, df_features, on='node_id')
        return df_final, G