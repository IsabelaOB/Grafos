import os
import pandas as pd
import numpy as np

class DataLoader:
    """Capa encargada de gestionar, simular o cargar los archivos CSV tabulares y relacionales."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.nodes_path = os.path.join(self.data_dir, "raw_nodes.csv")
        self.edges_path = os.path.join(self.data_dir, "raw_edges.csv")

    def generate_or_load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(self.nodes_path) or not os.path.exists(self.edges_path):
            print("[INFO] No se encontraron los archivos CSV. Generando dataset simulado realista...")
            np.random.seed(42)
            n_nodes = 1000
            
            # Crear nodos (Cuentas/Usuarios con atributos tabulares)
            node_ids = [f"node_{i}" for i in range(n_nodes)]
            df_nodes = pd.DataFrame({
                'node_id': node_ids,
                'antiguedad_meses': np.random.randint(1, 60, size=n_nodes),
                'ingreso_promedio': np.random.exponential(1500, size=n_nodes) + 500,
                'target': np.random.choice([0, 1], size=n_nodes, p=[0.93, 0.07]) # 7% fraude
            })
            
            edges = []
            for _ in range(3500):
                u = np.random.choice(node_ids)
                v = np.random.choice(node_ids)
                if u != v:
                    monto = np.random.exponential(300)
                    edges.append({'source': u, 'target': v, 'monto': monto})
            
            df_edges = pd.DataFrame(edges)
            
            df_nodes.to_csv(self.nodes_path, index=False)
            df_edges.to_csv(self.edges_path, index=False)
            print("[OK] Archivos 'raw_nodes.csv' y 'raw_edges.csv' generados con éxito en la carpeta /data.")
        
        df_nodes = pd.read_csv(self.nodes_path)
        df_edges = pd.read_csv(self.edges_path)
        return df_nodes, df_edges