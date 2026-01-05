# ================================================================
# Items — Upload CSV → Détection de doublons → Saisie (Optimisé sans Plotly)
# Auteur : Zineb FAKKAR – Janv 2026
# Optimisations : Cache, Vectorisation, Multithreading
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# -------- Configuration Streamlit --------
st.set_page_config(
    page_title="Items — Doublons Optimisé", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Dataclasses pour meilleure structure --------
@dataclass
class Item:
    """Classe pour représenter un item avec toutes ses propriétés"""
    id: str = ""
    reference: str = ""
    item_name: str = ""
    french_name: str = ""
    uom_name: str = ""
    type_name: str = ""
    sub_category_name: str = ""
    category_name: str = ""
    company_name: str = ""
    last_price: str = ""
    last_use: str = ""
    created_at: str = ""
    search_text: str = ""
    item_name_norm: str = ""
    ref_root: str = ""
    dupe_text: str = ""
    block_hash: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Item':
        """Crée un Item depuis un dictionnaire"""
        return cls(**{k: data.get(k, "") for k in cls.__annotations__.keys() if k in data})

# -------- Classes de cache optimisées --------
class ItemCache:
    """Cache pour les items avec indexation rapide"""
    def __init__(self):
        self._items_by_id = {}
        self._items_by_name_norm = {}
        self._items_by_ref_root = {}
        self._all_items = []
        self._search_texts = []
        
    def build(self, items: List[Item]):
        """Construit les index de cache"""
        self._all_items = items
        self._search_texts = [item.search_text for item in items]
        
        # Indexations
        self._items_by_id = {item.id: item for item in items if item.id}
        self._items_by_name_norm = {}
        for item in items:
            if item.item_name_norm:
                self._items_by_name_norm.setdefault(item.item_name_norm, []).append(item)
        
        self._items_by_ref_root = {}
        for item in items:
            if item.ref_root:
                self._items_by_ref_root.setdefault(item.ref_root, []).append(item)
    
    def get_by_id(self, item_id: str) -> Optional[Item]:
        """Récupère un item par son ID"""
        return self._items_by_id.get(item_id)
    
    def get_by_name_norm(self, name_norm: str) -> List[Item]:
        """Récupère tous les items avec un nom normalisé"""
        return self._items_by_name_norm.get(name_norm, [])
    
    def get_by_ref_root(self, ref_root: str) -> List[Item]:
        """Récupère tous les items avec une référence racine"""
        return self._items_by_ref_root.get(ref_root, [])
    
    @property
    def all_items(self) -> List[Item]:
        return self._all_items
    
    @property
    def search_texts(self) -> List[str]:
        return self._search_texts

# -------- Fonctions utilitaires optimisées avec cache --------
@st.cache_data(ttl=3600, show_spinner=False)
def strip_accents_batch(texts: List[str]) -> List[str]:
    """Version vectorisée pour traiter plusieurs textes à la fois"""
    results = []
    for text in texts:
        if pd.isna(text) or not text:
            results.append("")
            continue
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        results.append(text)
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def clean_text_batch(texts: List[str]) -> List[str]:
    """Version vectorisée de clean_text"""
    results = []
    for text in texts:
        if pd.isna(text) or not text:
            results.append("")
            continue
            
        # Convertir en minuscules et normaliser
        text = str(text).lower()
        
        # Supprimer accents
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Nettoyage regex optimisé
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'[_:/\\\-]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        results.append(text)
    return results

def ref_root_batch(refs: List[str]) -> List[str]:
    """Version vectorisée de ref_root"""
    results = []
    for ref in refs:
        if pd.isna(ref) or not ref:
            results.append("")
            continue
        ref = str(ref).lower().replace(' ', '')
        ref = re.sub(r'[-_/\.]', '', ref)
        results.append(ref)
    return results

# -------- Lecture CSV optimisée --------
EXPECTED_COLS = [
    "id", "reference", "item_name", "french_name", "uom_name",
    "type_name", "sub_category_name", "category_name", "company_name",
    "last_price", "last_use", "created_at"
]

RENAME_MAP = {
    "nom": "item_name", "name": "item_name",
    "libelle": "french_name", "libellé": "french_name",
    "unite": "uom_name", "uom": "uom_name",
    "type": "type_name",
    "sous_categorie": "sub_category_name", "sous-categorie": "sub_category_name",
    "sous catégorie": "sub_category_name",
    "categorie": "category_name", "catégorie": "category_name",
    "societe": "company_name", "société": "company_name",
    "prix": "last_price", "dernier_prix": "last_price",
    "derniere_utilisation": "last_use", "dernière_utilisation": "last_use",
    "cree_le": "created_at", "créé_le": "created_at",
    "unit": "uom_name", "company": "company_name",
    "category": "category_name", "sub_category": "sub_category_name",
    "created": "created_at",
}

@st.cache_data(ttl=3600, show_spinner=False)
def auto_detect_sep(sample_bytes: bytes) -> str:
    """Détecte rapidement le séparateur"""
    head = sample_bytes[:4096].decode("utf-8", errors="ignore")
    return ";" if head.count(";") >= head.count(",") else ","

@st.cache_data(ttl=3600, show_spinner=True)
def read_and_normalize_df(uploaded_file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, ItemCache]:
    """Lit et normalise le DataFrame en une seule passe avec cache"""
    
    # Détecter le séparateur
    sep = auto_detect_sep(uploaded_file_bytes)
    
    # Lire avec optimisation mémoire
    try:
        df = pd.read_csv(
            BytesIO(uploaded_file_bytes), 
            dtype=str, 
            encoding="utf-8", 
            sep=sep,
            on_bad_lines='skip',
            low_memory=True
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            BytesIO(uploaded_file_bytes), 
            dtype=str, 
            encoding="latin-1", 
            sep=sep,
            on_bad_lines='skip',
            low_memory=True
        )
    
    # Renommer colonnes
    rename_dict = {k: v for k, v in RENAME_MAP.items() 
                  if k in df.columns and v not in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # S'assurer que toutes les colonnes attendues existent
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""
    
    # Normaliser les colonnes texte
    text_cols = ["item_name", "french_name", "reference", "uom_name", 
                "type_name", "sub_category_name", "category_name", "company_name"]
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
    
    # Calculer les champs normalisés en une seule passe
    df["_item_name_norm"] = clean_text_batch(df["item_name"].tolist())
    df["_ref_root"] = ref_root_batch(df["reference"].tolist())
    
    # Créer le texte de recherche
    df["search_text"] = df[text_cols].apply(
        lambda row: " ".join([str(x) for x in row if x]), axis=1
    ).str.lower()
    
    # Pré-calculer le texte pour détection de doublons
    dupe_cols = ["item_name", "french_name", "reference", "uom_name", 
                 "type_name", "sub_category_name", "category_name"]
    df["_dupe_text"] = df[dupe_cols].apply(
        lambda row: clean_text_batch([" ".join([str(x) for x in row if x])])[0], 
        axis=1
    )
    
    # Construire le cache
    items = []
    for _, row in df.iterrows():
        item = Item(
            id=row.get("id", ""),
            reference=row.get("reference", ""),
            item_name=row.get("item_name", ""),
            french_name=row.get("french_name", ""),
            uom_name=row.get("uom_name", ""),
            type_name=row.get("type_name", ""),
            sub_category_name=row.get("sub_category_name", ""),
            category_name=row.get("category_name", ""),
            company_name=row.get("company_name", ""),
            last_price=row.get("last_price", ""),
            last_use=row.get("last_use", ""),
            created_at=row.get("created_at", ""),
            search_text=row.get("search_text", ""),
            item_name_norm=row.get("_item_name_norm", ""),
            ref_root=row.get("_ref_root", ""),
            dupe_text=row.get("_dupe_text", "")
        )
        items.append(item)
    
    cache = ItemCache()
    cache.build(items)
    
    return df, cache

# -------- Similarité rapide (Jaccard sur tokens) --------
class FastSimilarity:
    """Classe pour calculs de similarité rapides"""
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def tokenize(text: str) -> Set[str]:
        """Tokenisation avec cache"""
        if not text:
            return set()
        return set(text.split())
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Similarité Jaccard rapide"""
        if not text1 or not text2:
            return 0.0
        
        set1 = FastSimilarity.tokenize(text1)
        set2 = FastSimilarity.tokenize(text2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def prefix_similarity(text1: str, text2: str, prefix_len: int = 3) -> float:
        """Similarité basée sur préfixes communs"""
        if not text1 or not text2:
            return 0.0
        
        words1 = text1.split()[:prefix_len]
        words2 = text2.split()[:prefix_len]
        
        common = sum(1 for w1 in words1 for w2 in words2 if w1 and w2 and w1 == w2)
        total = max(len(words1), len(words2))
        
        return common / total if total > 0 else 0.0

# -------- Détection de doublons optimisée --------
def find_duplicates_fast(cache: ItemCache, new_item: Item, 
                        topn: int = 10, threshold: float = 0.82) -> List[Tuple[Item, float, str]]:
    """
    Trouve les doublons potentiels rapidement avec plusieurs stratégies
    """
    results = []
    
    # 1. Recherche exacte par nom normalisé
    exact_matches = cache.get_by_name_norm(new_item.item_name_norm)
    for item in exact_matches:
        if item.id != new_item.id:  # Éviter de se matcher avec soi-même
            results.append((item, 1.0, "exact_name"))
    
    if len(results) >= topn:
        return sorted(results, key=lambda x: x[1], reverse=True)[:topn]
    
    # 2. Recherche par référence racine
    ref_matches = cache.get_by_ref_root(new_item.ref_root)
    for item in ref_matches:
        if item.id != new_item.id:
            # Calculer une similarité combinée
            name_sim = FastSimilarity.jaccard_similarity(
                new_item.item_name_norm, item.item_name_norm
            )
            if name_sim >= threshold * 0.7:  # Seuil plus bas pour référence commune
                score = 0.5 + (name_sim * 0.5)  # Score mixte
                results.append((item, min(score, 0.95), "same_ref"))
    
    # 3. Recherche par similarité Jaccard (rapide)
    new_dupe_text = new_item.dupe_text or new_item.search_text
    if new_dupe_text and len(cache.search_texts) > 0:
        # Échantillonner pour accélérer (si trop d'items)
        max_samples = min(1000, len(cache.all_items))
        sample_indices = np.random.choice(len(cache.all_items), max_samples, replace=False)
        
        for idx in sample_indices:
            item = cache.all_items[idx]
            if item.id == new_item.id:
                continue
            
            # Filtre rapide par préfixe
            prefix_sim = FastSimilarity.prefix_similarity(
                new_dupe_text, item.dupe_text or item.search_text
            )
            
            if prefix_sim > 0.3:  # Seuil bas pour continuer
                jaccard_sim = FastSimilarity.jaccard_similarity(new_dupe_text, item.dupe_text or item.search_text)
                
                if jaccard_sim >= threshold:
                    # Boost si même catégorie/type
                    bonus = 0.1 if new_item.category_name == item.category_name else 0
                    bonus += 0.05 if new_item.type_name == item.type_name else 0
                    final_score = min(jaccard_sim + bonus, 1.0)
                    
                    results.append((item, final_score, "jaccard"))
    
    # Trier et limiter les résultats
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topn]

# -------- Détection globale de doublons (optimisée) --------
def detect_global_duplicates_optimized(df: pd.DataFrame, cache: ItemCache,
                                      block_cols: List[str], 
                                      threshold: float = 0.82,
                                      max_block_size: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détection globale de doublons avec optimisation mémoire et parallélisation
    """
    start_time = time.time()
    
    if len(df) <= 1:
        return pd.DataFrame(), pd.DataFrame()
    
    # Préparer les données
    work_df = df.copy()
    
    # Ajouter une colonne de hachage pour le blocage
    if block_cols:
        # Créer une clé de blocage
        work_df["_block_key"] = work_df[block_cols].fillna("").astype(str).agg("|".join, axis=1)
    else:
        work_df["_block_key"] = "ALL"
    
    # Diviser en blocs
    groups = []
    all_members = []
    group_id = 1
    
    # Traiter chaque bloc en parallèle
    blocks = list(work_df.groupby("_block_key"))
    
    def process_block(block_key: str, block_data: pd.DataFrame) -> List[Dict]:
        """Traite un bloc de données"""
        block_results = []
        
        if len(block_data) <= 1:
            return block_results
        
        # Réduire la taille si nécessaire
        if len(block_data) > max_block_size:
            # Échantillonner ou diviser
            sample_size = min(max_block_size, len(block_data))
            block_data = block_data.sample(sample_size, random_state=42)
        
        # Convertir en objets Item pour ce bloc
        block_items = []
        for _, row in block_data.iterrows():
            item = Item(
                id=row.get("id", ""),
                item_name=row.get("item_name", ""),
                item_name_norm=row.get("_item_name_norm", ""),
                ref_root=row.get("_ref_root", ""),
                dupe_text=row.get("_dupe_text", ""),
                search_text=row.get("search_text", ""),
                category_name=row.get("category_name", ""),
                type_name=row.get("type_name", ""),
                reference=row.get("reference", "")
            )
            block_items.append(item)
        
        # Matrice de similarité rapide (par lots)
        n = len(block_items)
        processed_pairs = set()
        
        for i in range(n):
            item_i = block_items[i]
            
            for j in range(i + 1, n):
                item_j = block_items[j]
                
                pair_key = tuple(sorted([item_i.id, item_j.id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Calcul rapide de similarité
                sim = FastSimilarity.jaccard_similarity(
                    item_i.dupe_text or item_i.search_text,
                    item_j.dupe_text or item_j.search_text
                )
                
                # Boost si même référence racine
                if item_i.ref_root and item_j.ref_root and item_i.ref_root == item_j.ref_root:
                    sim = max(sim, 0.9)
                
                if sim >= threshold:
                    # Même nom normalisé = score maximal
                    if item_i.item_name_norm and item_j.item_name_norm and item_i.item_name_norm == item_j.item_name_norm:
                        sim = 1.0
                    
                    block_results.append({
                        'i': i,
                        'j': j,
                        'score': sim,
                        'item_i': item_i,
                        'item_j': item_j
                    })
        
        return block_results
    
    # Traitement parallèle des blocs
    all_pairs = []
    with ThreadPoolExecutor(max_workers=min(4, len(blocks))) as executor:
        futures = {executor.submit(process_block, key, data): (key, data) 
                  for key, data in blocks if len(data) > 1}
        
        for future in as_completed(futures):
            block_results = future.result()
            all_pairs.extend(block_results)
    
    # Construire les groupes de doublons
    if all_pairs:
        # Utiliser Union-Find pour regrouper
        parent = {}
        rank = {}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root == y_root:
                return
            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            elif rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[y_root] = x_root
                rank[x_root] += 1
        
        # Initialiser Union-Find
        all_items_set = set()
        for pair in all_pairs:
            all_items_set.add(pair['item_i'].id)
            all_items_set.add(pair['item_j'].id)
        
        for item_id in all_items_set:
            parent[item_id] = item_id
            rank[item_id] = 0
        
        # Union des paires
        for pair in all_pairs:
            union(pair['item_i'].id, pair['item_j'].id)
        
        # Construire les composantes
        components = {}
        for item_id in all_items_set:
            root = find(item_id)
            components.setdefault(root, []).append(item_id)
        
        # Créer les DataFrames de résultat
        groups_data = []
        members_data = []
        
        for root, item_ids in components.items():
            if len(item_ids) <= 1:
                continue
            
            # Récupérer les données originales
            group_df = work_df[work_df['id'].isin(item_ids)].copy()
            
            # Trouver le représentant (celui avec la référence la plus longue)
            group_df['ref_len'] = group_df['reference'].str.len()
            rep_idx = group_df['ref_len'].idxmax()
            representative = group_df.loc[rep_idx]
            
            groups_data.append({
                'group_id': group_id,
                'size': len(group_df),
                'representative_reference': representative.get('reference', ''),
                'representative_name': representative.get('item_name', ''),
                'rule': 'fuzzy_blocked',
                'avg_score': np.mean([p['score'] for p in all_pairs 
                                    if p['item_i'].id in item_ids or p['item_j'].id in item_ids])
            })
            
            # Ajouter les membres
            group_df = group_df.drop(columns=['ref_len', '_block_key'], errors='ignore')
            group_df.insert(0, 'group_id', group_id)
            members_data.append(group_df)
            
            group_id += 1
        
        if groups_data:
            groups_df = pd.DataFrame(groups_data)
            members_df = pd.concat(members_data, ignore_index=True)
            
            # Trier
            groups_df = groups_df.sort_values(['size', 'avg_score'], ascending=[False, False])
            
            execution_time = time.time() - start_time
            st.info(f"⏱️ Analyse terminée en {execution_time:.2f} secondes")
            
            return groups_df, members_df
    
    return pd.DataFrame(), pd.DataFrame()

# -------- Interface Streamlit optimisée --------
def main():
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin: 0.25rem;
        }
        
        .badge-success { background: #10b981; color: white; }
        .badge-warning { background: #f59e0b; color: white; }
        .badge-danger { background: #ef4444; color: white; }
        .badge-info { background: #3b82f6; color: white; }
        
        .duplicate-item {
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            border: 1px solid #e2e8f0;
        }
        
        .progress-container {
            margin: 1rem 0;
            padding: 1rem;
            background: #f1f5f9;
            border-radius: 8px;
        }
        
        .dataframe-container {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }
        
        .stDataFrame {
            border-radius: 10px;
        }
        
        /* Style pour les barres de progression */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Détection de Doublons - Version Optimisée</h1>', unsafe_allow_html=True)
    
    # Sidebar - Upload
    with st.sidebar:
        st.markdown("### 📤 Chargement des données")
        
        uploaded_file = st.file_uploader(
            "Téléversez votre fichier CSV",
            type=['csv'],
            help="Format CSV avec UTF-8 ou Latin-1"
        )
        
        # Options de chargement
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            
            # Vérifier si le fichier a déjà été chargé
            if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
                with st.spinner("Chargement et optimisation en cours..."):
                    df, cache = read_and_normalize_df(file_bytes, uploaded_file.name)
                    
                    st.session_state.df = df
                    st.session_state.cache = cache
                    st.session_state.file_hash = file_hash
                    st.session_state.filename = uploaded_file.name
                    
                    st.success(f"✅ Fichier chargé : {uploaded_file.name}")
                    st.metric("Lignes", len(df))
                    st.metric("Colonnes", len(df.columns))
        
        # Boutons d'actions
        st.markdown("---")
        st.markdown("### ⚡ Actions rapides")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.cache_data.clear()
                for key in list(st.session_state.keys()):
                    if key != 'file_hash':
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📊 Aperçu", use_container_width=True) and 'df' in st.session_state:
                with st.expander("Aperçu des données", expanded=True):
                    st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Statistiques rapides
        if 'df' in st.session_state:
            st.markdown("---")
            st.markdown("### 📈 Statistiques")
            
            df_stats = st.session_state.df
            total_items = len(df_stats)
            
            if total_items > 0:
                # Noms uniques
                unique_names = df_stats['item_name'].nunique()
                duplicate_rate = ((total_items - unique_names) / total_items * 100) if total_items > 0 else 0
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Items uniques", unique_names)
                with col_stat2:
                    st.metric("Taux de doublons", f"{duplicate_rate:.1f}%")
    
    # Main content
    if 'df' not in st.session_state:
        st.info("👈 Veuillez téléverser un fichier CSV dans la barre latérale pour commencer.")
        
        # Aide et informations
        with st.expander("ℹ️ Comment utiliser cette application"):
            st.markdown("""
            ### Étapes d'utilisation :
            
            1. **Téléverser un CSV** contenant vos items
            2. **Configurer les paramètres** de détection
            3. **Analyser les doublons** globalement
            4. **Vérifier les doublons** pour un nouvel item
            5. **Exporter les résultats**
            
            ### Formats supportés :
            - CSV avec séparateur `;` ou `,`
            - Encodage UTF-8 ou Latin-1
            - Colonnes recommandées : `item_name`, `reference`, `category_name`
            
            ### Optimisations :
            - ⚡ **Cache intelligent** pour les requêtes répétées
            - 🚀 **Algorithmes rapides** (Jaccard, préfixes)
            - 📊 **Vectorisation** des opérations
            - 🧵 **Parallélisation** pour gros fichiers
            """)
        return
    
    # Données chargées - affichage principal
    df = st.session_state.df
    cache = st.session_state.cache
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs([
        "🔍 Analyse Globale", 
        "📝 Saisie & Vérification", 
        "📊 Statistiques & Export"
    ])
    
    # Tab 1 - Analyse Globale
    with tab1:
        st.header("🧹 Détection globale des doublons")
        
        # Configuration
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            # Colonnes de blocage disponibles - INCLURE item_name par défaut
            available_cols = [col for col in [
                'item_name', 'company_name', 'type_name', 'category_name', 
                'sub_category_name', 'uom_name'
            ] if col in df.columns]
            
            # Par défaut : item_name + type_name + category_name
            default_blocks = ['item_name', 'type_name', 'category_name']
            # N'utiliser que ceux qui existent dans les données
            default_blocks = [col for col in default_blocks if col in available_cols]
            
            block_cols = st.multiselect(
                "Colonnes de blocage",
                options=available_cols,
                default=default_blocks,
                help="Réduit les comparaisons aux items similaires (item_name recommandé)"
            )
        
        with col_config2:
            threshold = st.slider(
                "Seuil de similarité",
                min_value=0.60,
                max_value=0.95,
                value=0.82,
                step=0.01,
                help="Plus élevé = moins de faux positifs"
            )
            
            max_block = st.number_input(
                "Taille max par bloc",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Optimise la mémoire pour gros fichiers"
            )
        
        with col_config3:
            sampling = st.checkbox(
                "Échantillonnage intelligent",
                value=True,
                help="Analyse un échantillon pour très gros fichiers"
            )
            
            if sampling and len(df) > 5000:
                sample_size = st.slider(
                    "Taille de l'échantillon",
                    min_value=1000,
                    max_value=min(10000, len(df)),
                    value=min(5000, len(df)),
                    step=500
                )
            else:
                sample_size = len(df)
        
        # Bouton d'analyse
        if st.button("🚀 Lancer l'analyse globale", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours... Cette opération est optimisée pour la vitesse"):
                progress_bar = st.progress(0)
                
                # Utiliser un échantillon si nécessaire
                if sample_size < len(df):
                    analysis_df = df.sample(sample_size, random_state=42).copy()
                    st.info(f"🔬 Analyse sur échantillon de {sample_size} items ({sample_size/len(df)*100:.1f}% des données)")
                else:
                    analysis_df = df.copy()
                
                # Lancer l'analyse
                progress_bar.progress(30)
                groups_df, members_df = detect_global_duplicates_optimized(
                    analysis_df, cache, block_cols, threshold, max_block
                )
                progress_bar.progress(100)
                
                # Afficher les résultats
                if len(groups_df) == 0:
                    st.success("🎉 Aucun doublon détecté avec ces paramètres !")
                else:
                    # Résumé
                    st.markdown(f"### 📊 Résultats : {len(groups_df)} groupes de doublons détectés")
                    
                    # Statistiques rapides
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Groupes", len(groups_df))
                    with col_stat2:
                        total_dupes = groups_df['size'].sum() - len(groups_df)
                        st.metric("Doublons totaux", total_dupes)
                    with col_stat3:
                        avg_group_size = groups_df['size'].mean()
                        st.metric("Taille moyenne", f"{avg_group_size:.1f}")
                    
                    # Afficher les groupes
                    with st.expander("📋 Liste des groupes", expanded=True):
                        display_cols = ['group_id', 'size', 'representative_name', 
                                      'representative_reference', 'avg_score']
                        display_cols = [c for c in display_cols if c in groups_df.columns]
                        
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(
                            groups_df[display_cols].head(20),
                            use_container_width=True,
                            column_config={
                                'avg_score': st.column_config.ProgressColumn(
                                    "Similarité",
                                    format="%.2f",
                                    min_value=0,
                                    max_value=1
                                ),
                                'size': st.column_config.NumberColumn(
                                    "Taille",
                                    help="Nombre d'items dans le groupe"
                                )
                            }
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Afficher les membres d'un groupe sélectionné
                    if len(members_df) > 0:
                        st.markdown("### 👥 Détail d'un groupe")
                        selected_group = st.selectbox(
                            "Sélectionner un groupe pour voir les membres",
                            options=groups_df['group_id'].tolist(),
                            key="group_selector"
                        )
                        
                        group_members = members_df[members_df['group_id'] == selected_group]
                        if len(group_members) > 0:
                            with st.expander(f"📋 Membres du groupe {selected_group}", expanded=False):
                                member_cols = ['id', 'item_name', 'reference', 
                                             'category_name', 'type_name', 'company_name']
                                member_cols = [c for c in member_cols if c in group_members.columns]
                                
                                st.markdown(f"**{len(group_members)} items dans ce groupe**")
                                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                                st.dataframe(
                                    group_members[member_cols],
                                    use_container_width=True
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Boutons d'export
                    st.markdown("### 💾 Export des résultats")
                    col_export1, col_export2 = st.columns(2)
                    with col_export1:
                        csv_groups = groups_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Télécharger les groupes",
                            data=csv_groups,
                            file_name="groupes_doublons.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_export2:
                        if len(members_df) > 0:
                            csv_members = members_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Télécharger tous les membres",
                                data=csv_members,
                                file_name="membres_doublons.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
    
    # Tab 2 - Saisie & Vérification
    with tab2:
        st.header("📝 Vérifier un nouvel item")
        
        # Formulaire de saisie
        with st.form("nouvel_item_form"):
            col_left, col_right = st.columns(2)
            
            with col_left:
                item_name = st.text_input("Nom de l'item *", "", 
                                        help="Nom principal de l'item")
                french_name = st.text_input("Nom français", "")
                reference = st.text_input("Référence", "")
                uom_name = st.text_input("Unité de mesure", "")
            
            with col_right:
                type_name = st.text_input("Type", "")
                sub_category_name = st.text_input("Sous-catégorie", "")
                category_name = st.text_input("Catégorie", "")
                company_name = st.text_input("Société", "")
            
            # Options de vérification
            st.markdown("---")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                topn = st.slider("Nombre de résultats", 3, 20, 8)
            with col_opt2:
                threshold_check = st.slider("Seuil de similarité", 0.60, 0.95, 0.82, 0.01)
            
            submitted = st.form_submit_button("🔍 Vérifier les doublons", type="primary")
        
        if submitted and item_name:
            with st.spinner("Recherche de doublons en cours..."):
                # Créer l'item
                new_item = Item(
                    item_name=item_name,
                    french_name=french_name,
                    reference=reference,
                    uom_name=uom_name,
                    type_name=type_name,
                    sub_category_name=sub_category_name,
                    category_name=category_name,
                    company_name=company_name,
                    item_name_norm=clean_text_batch([item_name])[0],
                    ref_root=ref_root_batch([reference])[0],
                    dupe_text=clean_text_batch([
                        f"{item_name} {french_name} {reference} {uom_name} "
                        f"{type_name} {sub_category_name} {category_name}"
                    ])[0]
                )
                
                # Rechercher les doublons
                duplicates = find_duplicates_fast(cache, new_item, topn, threshold_check)
                
                # Afficher les résultats
                if not duplicates:
                    st.success("✅ Aucun doublon potentiel détecté !")
                else:
                    # Séparer par type
                    exact_duplicates = [d for d in duplicates if d[2] == "exact_name"]
                    other_duplicates = [d for d in duplicates if d[2] != "exact_name"]
                    
                    if exact_duplicates:
                        st.error(f"⚠️ {len(exact_duplicates)} doublon(s) exact(s) trouvé(s)")
                        
                        for item, score, rule in exact_duplicates[:3]:  # Limiter à 3
                            with st.container():
                                col_info, col_score = st.columns([4, 1])
                                with col_info:
                                    st.markdown(f"""
                                    <div class="duplicate-item">
                                        <strong>{item.item_name}</strong><br>
                                        <small>Réf: {item.reference} | Cat: {item.category_name} | Type: {item.type_name}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col_score:
                                    st.markdown(f'<span class="metric-badge badge-danger">{score:.0%}</span>', 
                                              unsafe_allow_html=True)
                    
                    if other_duplicates:
                        st.warning(f"🔍 {len(other_duplicates)} item(s) similaire(s)")
                        
                        for item, score, rule in other_duplicates[:5]:  # Limiter à 5
                            with st.container():
                                col_info, col_score = st.columns([4, 1])
                                with col_info:
                                    st.markdown(f"""
                                    <div class="duplicate-item">
                                        <strong>{item.item_name}</strong><br>
                                        <small>Réf: {item.reference} | Cat: {item.category_name} | Type: {item.type_name}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col_score:
                                    badge_class = "badge-warning" if score > 0.9 else "badge-info"
                                    st.markdown(f'<span class="metric-badge {badge_class}">{score:.0%}</span>', 
                                              unsafe_allow_html=True)
                    
                    # Statistiques
                    if duplicates:
                        avg_score = np.mean([score for _, score, _ in duplicates])
                        st.metric("Similarité moyenne", f"{avg_score:.1%}")
                        
                        # Export des candidats
                        st.markdown("### 💾 Export des candidats")
                        candidates_df = pd.DataFrame([
                            {
                                'item_name': item.item_name,
                                'reference': item.reference,
                                'category_name': item.category_name,
                                'type_name': item.type_name,
                                'score': score,
                                'match_type': rule
                            }
                            for item, score, rule in duplicates
                        ])
                        
                        csv_candidates = candidates_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Télécharger la liste des candidats",
                            data=csv_candidates,
                            file_name="candidats_doublons.csv",
                            mime="text/csv"
                        )
    
    # Tab 3 - Statistiques & Export
    with tab3:
        st.header("📊 Statistiques détaillées")
        
        if 'df' in st.session_state:
            # Calcul des statistiques
            total_items = len(df)
            unique_names = df['item_name'].nunique()
            unique_refs = df['reference'].nunique() if 'reference' in df.columns else 0
            
            # Distribution par catégorie
            if 'category_name' in df.columns:
                cat_dist = df['category_name'].value_counts().head(10)
                
                # Métriques principales
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total items", total_items)
                with col_stat2:
                    st.metric("Noms uniques", unique_names)
                with col_stat3:
                    duplicate_rate = ((total_items - unique_names) / total_items * 100) if total_items > 0 else 0
                    st.metric("Taux de doublons", f"{duplicate_rate:.1f}%")
                
                # Visualisations avec Streamlit native (pas de Plotly)
                st.markdown("### 📈 Top catégories")
                if len(cat_dist) > 0:
                    # Utiliser st.bar_chart
                    st.bar_chart(cat_dist)
                    
                    # Afficher le tableau
                    with st.expander("📋 Voir le détail par catégorie"):
                        cat_df = pd.DataFrame({
                            'Catégorie': cat_dist.index,
                            'Nombre d\'items': cat_dist.values,
                            'Pourcentage': (cat_dist.values / total_items * 100).round(1)
                        })
                        st.dataframe(cat_df, use_container_width=True)
                
                # Distribution doublons vs uniques
                st.markdown("### 🏷️ Distribution doublons/uniques")
                if total_items > 0:
                    chart_data = pd.DataFrame({
                        'Type': ['Items uniques', 'Doublons potentiels'],
                        'Nombre': [unique_names, total_items - unique_names]
                    })
                    
                    # Afficher avec st.bar_chart
                    st.bar_chart(chart_data.set_index('Type'))
                    
                    # Métriques détaillées
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    with col_detail1:
                        st.metric("Références uniques", unique_refs)
                    with col_detail2:
                        if 'type_name' in df.columns:
                            unique_types = df['type_name'].nunique()
                            st.metric("Types uniques", unique_types)
                    with col_detail3:
                        if 'company_name' in df.columns:
                            unique_companies = df['company_name'].nunique()
                            st.metric("Sociétés", unique_companies)
            
            # Export des données
            st.markdown("---")
            st.markdown("### 💾 Export des données")
            
            export_format = st.radio(
                "Format d'export",
                ["CSV (Recommandé)", "Excel", "JSON"],
                horizontal=True
            )
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Export complet
                if export_format == "CSV (Recommandé)":
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Télécharger toutes les données (CSV)",
                        data=csv_data,
                        file_name="items_complet.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                elif export_format == "Excel":
                    try:
                        # Pour Excel, on va créer un buffer
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Items')
                        excel_data = excel_buffer.getvalue()
                        
                        st.download_button(
                            "📥 Télécharger toutes les données (Excel)",
                            data=excel_data,
                            file_name="items_complet.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except:
                        st.warning("Export Excel non disponible. Installez 'openpyxl' ou utilisez CSV.")
                else:  # JSON
                    json_data = df.to_json(orient='records', force_ascii=False)
                    st.download_button(
                        "📥 Télécharger toutes les données (JSON)",
                        data=json_data,
                        file_name="items_complet.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col_exp2:
                # Export des items uniques
                unique_df = df.drop_duplicates(subset=['item_name', 'reference'], keep='first')
                csv_unique = unique_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    "📥 Télécharger les items uniques",
                    data=csv_unique,
                    file_name="items_uniques.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Options avancées
            with st.expander("⚙️ Options avancées d'export"):
                # Sélection de colonnes
                selected_cols = st.multiselect(
                    "Colonnes à exporter",
                    options=df.columns.tolist(),
                    default=['id', 'item_name', 'reference', 'category_name', 'type_name', 'company_name']
                )
                
                # Filtres
                filter_category = st.selectbox(
                    "Filtrer par catégorie",
                    options=['Toutes'] + sorted(df['category_name'].unique().tolist())
                )
                
                if st.button("Générer l'export personnalisé", key="custom_export"):
                    filtered_df = df.copy()
                    
                    if filter_category != 'Toutes':
                        filtered_df = filtered_df[filtered_df['category_name'] == filter_category]
                    
                    if selected_cols:
                        filtered_df = filtered_df[selected_cols]
                    
                    if len(filtered_df) > 0:
                        csv_custom = filtered_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            "📥 Télécharger l'export personnalisé",
                            data=csv_custom,
                            file_name="export_personnalise.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Aucune donnée à exporter avec ces filtres.")

if __name__ == "__main__":
    main()