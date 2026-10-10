import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

class ModelEvaluator:
    """Capa de negocio encargada de la validación cruzada estricta y la comparación de modelo StratifiedKFolds."""
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_comparison(self, df_enriched: pd.DataFrame):
        y = df_enriched['target'].values
        
        # Escenario A: Modelo Tradicional (SOLO datos tabulares de los CSVs)
        cols_tabular = ['antiguedad_meses', 'ingreso_promedio']
        X_tabular = df_enriched[cols_tabular].values
        
        # Escenario B: Modelo Enterprise Graph ML (Datos Tabulares + Métricas Topológicas + Embeddings)
        cols_graph = [c for c in df_enriched.columns if c not in ['node_id', 'target']]
        X_graph = df_enriched[cols_graph].values

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Entrenar Baseline Tabular
        clf_tab = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=self.random_state)
        probs_tab = cross_val_predict(clf_tab, X_tabular, y, cv=cv, method='predict_proba')[:, 1]
        auc_tab = roc_auc_score(y, probs_tab)

        # Entrenar Graph ML Enriched
        clf_graph = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=self.random_state)
        probs_graph = cross_val_predict(clf_graph, X_graph, y, cv=cv, method='predict_proba')[:, 1]
        auc_graph = roc_auc_score(y, probs_graph)

        print("\n" + "="*50)
        print(" RESULTADOS COMPARATIVOS")
        print("="*50)
        print(f" 🔹 ROC-AUC Modelo Tradicional (Solo Tabular): {auc_tab:.4f}")
        print(f"  ROC-AUC Modelo Graph ML (Tabular + Red):   {auc_graph:.4f}")
        print("="*50)

        # Threshold Tuning avanzado para el modelo de Graph ML
        precisions, recalls, thresholds = precision_recall_curve(y, probs_graph)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        print(f"\n[INFO] Umbral óptimo calculado por F1-Score: {best_threshold:.4f}")
        y_pred_tuned = (probs_graph >= best_threshold).astype(int)
        
        print("\n--- REPORTE DE CLASIFICACIÓN (GRAPH ML CON UMBRAL AJUSTADO) ---")
        print(classification_report(y, y_pred_tuned))
        
        return auc_tab, auc_graph