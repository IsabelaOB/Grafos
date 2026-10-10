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
            
            # 1. Crear nodos (Cuentas/Usuarios)
            node_ids = np.array([f"node_{i}" for i in range(n_nodes)])
            targets = np.random.choice([0, 1], size=n_nodes, p=[0.93, 0.07]) # 7% fraude
            
            df_nodes = pd.DataFrame({
                'node_id': node_ids,
                'antiguedad_meses': np.random.randint(1, 60, size=n_nodes),
                'ingreso_promedio': np.random.exponential(1500, size=n_nodes) + 500,
                'target': targets
            })
            
            # Separar nodos para crear topología realista
            fraud_nodes = df_nodes[df_nodes['target'] == 1]['node_id'].values
            normal_nodes = df_nodes[df_nodes['target'] == 0]['node_id'].values
            
            edges = []
            
            # 2. Transacciones normales (Ruido de fondo)
            for _ in range(2500):
                u = np.random.choice(node_ids)
                v = np.random.choice(node_ids)
                if u != v:
                    monto = np.random.exponential(300)
                    edges.append({'source': u, 'target': v, 'monto': monto})
                    
            # 3. INYECTAR RED CRIMINAL Y RUIDO CRUZADO (Para un ROC-AUC más realista)
            
            # A. Conexiones exclusivas entre estafadores (Homofilia)
            for _ in range(150):
                u = np.random.choice(fraud_nodes)
                v = np.random.choice(fraud_nodes)
                if u != v:
                    monto = np.random.exponential(5000)
                    edges.append({'source': u, 'target': v, 'monto': monto})
                    
            # B. Conexiones de estafadores a cuentas normales (Para confundir al modelo)
            for _ in range(200):
                u = np.random.choice(fraud_nodes)
                v = np.random.choice(normal_nodes)
                monto = np.random.exponential(1000)
                edges.append({'source': u, 'target': v, 'monto': monto})
            
            df_edges = pd.DataFrame(edges)
            
            # Guardar en CSV
            df_nodes.to_csv(self.nodes_path, index=False)
            df_edges.to_csv(self.edges_path, index=False)
            print("[OK] Archivos generados con topología de red criminal inyectada y ruido cruzado.")
        
        df_nodes = pd.read_csv(self.nodes_path)
        df_edges = pd.read_csv(self.edges_path)
        return df_nodes, df_edges