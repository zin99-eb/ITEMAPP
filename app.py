# ================================================================
# Items — Connexion Directe au Data Warehouse Netis Group
# Auteur : Zineb FAKKAR – Fév 2026
# Version corrigée avec persistance de la sélection des groupes
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime
import hashlib
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import warnings
import zipfile
from sqlalchemy import create_engine, text
import urllib.parse
warnings.filterwarnings('ignore')

# ================================================================
# Configuration Streamlit
# ================================================================
st.set_page_config(
    page_title="Items — Netis Group DWH",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# === CONNEXION DATA WAREHOUSE NETIS GROUP
# ================================================================

def init_connection_params():
    """Initialiser les paramètres de connexion pour Netis Group DWH"""
    if 'db_config' not in st.session_state:
        st.session_state.db_config = {
            'db_type': 'postgresql',
            'host': 'dw.netisgroup.net',
            'port': '8822',
            'database': 'Netis-DW',
            'schema': 'SCM',
            'username': 'it_powerbi_user',
            'password': 'Netis@@IT2025#',
            'query': """SELECT * 
FROM "SCM"."API_Items" 
WHERE created_at >= '2024-01-01' 
AND status = 'Qualified'AND type = 0.0"""
        }

def create_connection_string(config):
    """Créer la chaîne de connexion"""
    username = config['username']
    password = config['password']
    host = config['host']
    port = config['port']
    database = config['database']
    
    if password:
        password = urllib.parse.quote_plus(password)
    
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"

@st.cache_resource
def get_engine():
    """Obtenir le moteur SQLAlchemy"""
    if 'db_config' not in st.session_state:
        return None
    
    config = st.session_state.db_config
    
    try:
        conn_str = create_connection_string(config)
        engine = create_engine(
            conn_str,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
            connect_args={
                'connect_timeout': 30,
                'sslmode': 'require'
            }
        )
        return engine
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=True)
def load_data_from_dwh(query: str = None) -> pd.DataFrame:
    """Charger les données depuis le Data Warehouse Netis Group"""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    
    if query is None:
        query = st.session_state.db_config['query']
    
    try:
        with st.spinner("Connexion à Netis Group DWH..."):
            st.info(f"Exécution de la requête sur le schéma SCM...")
            
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            
        if df.empty:
            st.warning("Aucune donnée trouvée avec cette requête.")
            return pd.DataFrame()
        
        st.success(f"✅ {len(df):,} lignes chargées depuis Netis Group DWH")
        
        with st.expander("📋 Colonnes chargées"):
            st.write(list(df.columns))
            
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.info("💡 Vérifiez que vous êtes connecté au VPN Netis Group")
        return pd.DataFrame()

def test_connection():
    """Tester la connexion à Netis Group DWH"""
    engine = get_engine()
    if engine is None:
        return False, "Moteur de connexion non initialisé"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test")).fetchone()
            if result and result[0] == 1:
                return True, "✅ Connexion réussie à Netis Group DWH !"
            else:
                return False, "Test de connexion échoué"
    except Exception as e:
        return False, f"❌ Échec de connexion : {e}"

def refresh_data():
    """Forcer le rafraîchissement des données"""
    st.cache_data.clear()
    if 'df' in st.session_state:
        del st.session_state.df
    if 'cache' in st.session_state:
        del st.session_state.cache
    if 'last_load' in st.session_state:
        del st.session_state.last_load
    if 'selected_group' in st.session_state:
        del st.session_state.selected_group
    st.rerun()

# ================================================================
# === IA SÉMANTIQUE — EMBEDDINGS
# ================================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedding_model = load_embedding_model()

@st.cache_data(ttl=3600, show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    return embedding_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# ================================================================
# Dataclass Item
# ================================================================
@dataclass
class Item:
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
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Item':
        return cls(**{k: data.get(k, "") for k in cls.__annotations__.keys() if k in data})

# ================================================================
# Cache d'items
# ================================================================
class ItemCache:
    def __init__(self):
        self._items_by_id: Dict[str, Item] = {}
        self._items_by_name_norm: Dict[str, List[Item]] = {}
        self._items_by_ref_root: Dict[str, List[Item]] = {}
        self._all_items: List[Item] = []
        self._search_texts: List[str] = []

    def build(self, items: List[Item]):
        self._all_items = items
        self._search_texts = [item.search_text for item in items]
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
        return self._items_by_id.get(item_id)

    def get_by_name_norm(self, name_norm: str) -> List[Item]:
        return self._items_by_name_norm.get(name_norm, [])

    def get_by_ref_root(self, ref_root: str) -> List[Item]:
        return self._items_by_ref_root.get(ref_root, [])

    @property
    def all_items(self) -> List[Item]:
        return self._all_items

    @property
    def search_texts(self) -> List[str]:
        return self._search_texts

# ================================================================
# Fonctions utilitaires (texte)
# ================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def strip_accents_batch(texts: List[str]) -> List[str]:
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
    results = []
    for text in texts:
        if pd.isna(text) or not text:
            results.append("")
            continue
        text = str(text).lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'[_:/\\\-]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        results.append(text)
    return results

def ref_root_batch(refs: List[str]) -> List[str]:
    results = []
    for ref in refs:
        if pd.isna(ref) or not ref:
            results.append("")
            continue
        ref = str(ref).lower().replace(' ', '')
        ref = re.sub(r'[-_/\.]', '', ref)
        results.append(ref)
    return results

def safe_filename(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r'[^A-Za-z0-9_\-\.]+', '', name)
    return name or "BL"

# ================================================================
# Normalisation des données DWH
# ================================================================
EXPECTED_COLS = [
    "id", "reference", "item_name", "french_name", "uom_name",
    "type_name", "sub_category_name", "category_name", "company_name",
    "last_price", "last_use", "created_at",
    "status", "requestor_name", "department_name", "updated_at"
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

def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, ItemCache]:
    """Normalise le DataFrame et construit le cache avec embeddings"""
    
    rename_dict = {k: v for k, v in RENAME_MAP.items() if k in df.columns and v not in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""

    text_cols = ["item_name", "french_name", "reference", "uom_name",
                 "type_name", "sub_category_name", "category_name", "company_name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    df = df.reset_index(drop=True)

    df["_item_name_norm"] = clean_text_batch(df["item_name"].tolist())
    df["_ref_root"] = ref_root_batch(df["reference"].tolist())

    df["search_text"] = df[text_cols].apply(lambda row: " ".join([str(x) for x in row if x]), axis=1).str.lower()

    dupe_cols = ["item_name", "french_name", "reference", "uom_name", "type_name", "sub_category_name", "category_name"]
    df["_dupe_text"] = df[dupe_cols].apply(lambda row: clean_text_batch([" ".join([str(x) for x in row if x])])[0], axis=1)

    df["_semantic_input"] = (
        df["item_name"].fillna("") + " " +
        df["french_name"].fillna("") + " " +
        df["category_name"].fillna("") + " " +
        df["type_name"].fillna("")
    )
    
    if len(df) > 0:
        with st.spinner("Calcul des embeddings sémantiques..."):
            semantic_embeddings = embed_texts(df["_semantic_input"].tolist())
    else:
        semantic_embeddings = np.array([])

    items: List[Item] = []
    for i, row in df.iterrows():
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
            dupe_text=row.get("_dupe_text", ""),
            embedding=semantic_embeddings[i] if len(semantic_embeddings) > 0 else None
        )
        items.append(item)

    cache = ItemCache()
    cache.build(items)

    return df, cache

# ================================================================
# Similarité rapide (Jaccard)
# ================================================================
class FastSimilarity:
    @staticmethod
    @lru_cache(maxsize=10000)
    def tokenize(text: str) -> Set[str]:
        if not text:
            return set()
        return set(text.split())

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
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
        if not text1 or not text2:
            return 0.0
        words1 = text1.split()[:prefix_len]
        words2 = text2.split()[:prefix_len]
        common = sum(1 for w1 in words1 for w2 in words2 if w1 and w2 and w1 == w2)
        total = max(len(words1), len(words2))
        return common / total if total > 0 else 0.0

# ================================================================
# Détection de doublons (saisie)
# ================================================================
def find_duplicates_fast(cache: ItemCache, new_item: Item,
                         topn: int = 10, threshold: float = 0.82) -> List[Tuple[Item, float, str]]:
    """Trouve les doublons potentiels"""
    results: List[Tuple[Item, float, str]] = []

    exact_matches = cache.get_by_name_norm(new_item.item_name_norm)
    for item in exact_matches:
        if item.id != new_item.id:
            results.append((item, 1.0, "exact_name"))

    if len(results) >= topn:
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:topn]
        return results_sorted

    ref_matches = cache.get_by_ref_root(new_item.ref_root)
    for item in ref_matches:
        if item.id != new_item.id:
            name_sim = FastSimilarity.jaccard_similarity(new_item.item_name_norm, item.item_name_norm)
            if name_sim >= threshold * 0.7:
                score = 0.5 + (name_sim * 0.5)
                results.append((item, min(score, 0.95), "same_ref"))

    new_dupe_text = new_item.dupe_text or new_item.search_text
    if new_dupe_text and len(cache.search_texts) > 0:
        max_samples = min(1000, len(cache.all_items))
        sample_indices = np.random.choice(len(cache.all_items), max_samples, replace=False)

        for idx in sample_indices:
            item = cache.all_items[idx]
            if item.id == new_item.id:
                continue

            prefix_sim = FastSimilarity.prefix_similarity(new_dupe_text, item.dupe_text or item.search_text)
            if prefix_sim > 0.3:
                jaccard_sim = FastSimilarity.jaccard_similarity(new_dupe_text, item.dupe_text or item.search_text)
                if jaccard_sim >= threshold:
                    bonus = 0.1 if new_item.category_name == item.category_name else 0
                    bonus += 0.05 if new_item.type_name == item.type_name else 0
                    final_score = min(jaccard_sim + bonus, 1.0)
                    results.append((item, final_score, "jaccard"))

    if new_item.embedding is not None:
        candidates = cache.all_items
        same_cat = [it for it in candidates if it.category_name and it.category_name == new_item.category_name]
        if same_cat:
            candidates = same_cat

        max_semantic = 5000
        if len(candidates) > max_semantic:
            sel = np.random.choice(len(candidates), max_semantic, replace=False)
            candidates = [candidates[i] for i in sel]

        for item in candidates:
            if item.id == new_item.id or item.embedding is None:
                continue
            semantic_sim = cosine_similarity(new_item.embedding, item.embedding)
            if semantic_sim >= threshold:
                bonus = 0.05 if new_item.category_name == item.category_name else 0
                final_score = min(semantic_sim + bonus, 1.0)
                results.append((item, final_score, "semantic_ai"))

    best_by_id: Dict[str, Tuple[Item, float, str]] = {}
    for (it, sc, rl) in results:
        if (it.id not in best_by_id) or (sc > best_by_id[it.id][1]):
            best_by_id[it.id] = (it, sc, rl)
    results = list(best_by_id.values())

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topn]

# ================================================================
# Détection globale
# ================================================================
def detect_global_duplicates_optimized(df: pd.DataFrame, cache: ItemCache,
                                       block_cols: List[str],
                                       threshold: float = 0.82,
                                       max_block_size: int = 2500,
                                       use_semantic: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Détection globale avec blocage + Jaccard + option IA"""
    start_time = time.time()

    if len(df) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    work_df = df.copy()

    if block_cols:
        work_df["_block_key"] = work_df[block_cols].fillna("").astype(str).agg("|".join, axis=1)
    else:
        work_df["_block_key"] = "ALL"

    blocks = list(work_df.groupby("_block_key"))

    def process_block(block_key: str, block_data: pd.DataFrame) -> List[Dict]:
        block_results = []
        if len(block_data) <= 1:
            return block_results

        if len(block_data) > max_block_size:
            sample_size = min(max_block_size, len(block_data))
            block_data = block_data.sample(sample_size, random_state=42)

        block_items: List[Item] = []
        for _, row in block_data.iterrows():
            item_id = row.get("id", "")
            cache_item = cache.get_by_id(item_id) if item_id else None
            item = Item(
                id=item_id,
                item_name=row.get("item_name", ""),
                item_name_norm=row.get("_item_name_norm", ""),
                ref_root=row.get("_ref_root", ""),
                dupe_text=row.get("_dupe_text", ""),
                search_text=row.get("search_text", ""),
                category_name=row.get("category_name", ""),
                type_name=row.get("type_name", ""),
                reference=row.get("reference", ""),
                embedding=(cache_item.embedding if cache_item else None)
            )
            block_items.append(item)

        n = len(block_items)
        processed_pairs = set()

        for i in range(n):
            item_i = block_items[i]
            text_i = item_i.dupe_text or item_i.search_text

            for j in range(i + 1, n):
                item_j = block_items[j]
                text_j = item_j.dupe_text or item_j.search_text

                pair_key = tuple(sorted([item_i.id, item_j.id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                sim = FastSimilarity.jaccard_similarity(text_i, text_j)

                if item_i.ref_root and item_j.ref_root and item_i.ref_root == item_j.ref_root:
                    sim = max(sim, 0.9)

                if (sim < threshold) and use_semantic and (item_i.embedding is not None) and (item_j.embedding is not None):
                    pref = FastSimilarity.prefix_similarity(text_i, text_j)
                    if pref >= 0.2 or (item_i.category_name and item_i.category_name == item_j.category_name):
                        sem_sim = cosine_similarity(item_i.embedding, item_j.embedding)
                        if sem_sim >= threshold:
                            sim = sem_sim

                if sim >= threshold:
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

    all_pairs = []
    with ThreadPoolExecutor(max_workers=min(4, len(blocks))) as executor:
        futures = {executor.submit(process_block, key, data): (key, data) for key, data in blocks if len(data) > 1}
        for future in as_completed(futures):
            block_results = future.result()
            all_pairs.extend(block_results)

    if all_pairs:
        parent: Dict[str, str] = {}
        rank: Dict[str, int] = {}

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

        all_items_set = set()
        for pair in all_pairs:
            all_items_set.add(pair['item_i'].id)
            all_items_set.add(pair['item_j'].id)

        for item_id in all_items_set:
            parent[item_id] = item_id
            rank[item_id] = 0

        for pair in all_pairs:
            union(pair['item_i'].id, pair['item_j'].id)

        components: Dict[str, List[str]] = {}
        for item_id in all_items_set:
            root = find(item_id)
            components.setdefault(root, []).append(item_id)

        groups_data = []
        members_data = []
        group_id = 1

        for root, item_ids in components.items():
            if len(item_ids) <= 1:
                continue

            group_df = work_df[work_df['id'].isin(item_ids)].copy()
            group_df['ref_len'] = group_df['reference'].astype(str).str.len()
            rep_idx = group_df['ref_len'].idxmax()
            representative = group_df.loc[rep_idx]

            groups_data.append({
                'group_id': group_id,
                'size': len(group_df),
                'representative_reference': representative.get('reference', ''),
                'representative_name': representative.get('item_name', ''),
                'rule': 'fuzzy_blocked' + ('+semantic' if use_semantic else ''),
                'avg_score': np.mean([p['score'] for p in all_pairs
                                      if p['item_i'].id in item_ids or p['item_j'].id in item_ids])
            })

            group_df = group_df.drop(columns=['ref_len', '_block_key'], errors='ignore')
            group_df.insert(0, 'group_id', group_id)
            members_data.append(group_df)

            group_id += 1

        if groups_data:
            groups_df = pd.DataFrame(groups_data)
            members_df = pd.concat(members_data, ignore_index=True)
            groups_df = groups_df.sort_values(['size', 'avg_score'], ascending=[False, False])

            execution_time = time.time() - start_time
            st.info(f"⏱️ Analyse terminée en {execution_time:.2f} secondes")

            return groups_df, members_df

    return pd.DataFrame(), pd.DataFrame()

# ================================================================
# Fallback pour group_id
# ================================================================
def build_temp_groups(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp['group_id'] = np.nan

    if "_ref_root" in temp.columns:
        grp = temp.groupby("_ref_root", dropna=False)
        gid = 1
        for key, g in grp:
            if not key or key == "" or len(g) <= 1:
                continue
            temp.loc[g.index, 'group_id'] = gid
            gid += 1

    ungroupped = temp[temp['group_id'].isna()]
    grp2 = ungroupped.groupby("_item_name_norm", dropna=False)
    for key, g in grp2:
        if not key or key == "" or len(g) <= 1:
            continue
        gid = (temp['group_id'].max() or 0) + 1
        temp.loc[g.index, 'group_id'] = gid

    return temp

# ================================================================
# Interface Streamlit
# ================================================================
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
        .connection-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem;
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
        .dataframe-container {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }
        .group-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🏢 Détection de Doublons - Netis Group DWH</h1>', unsafe_allow_html=True)

    init_connection_params()

    with st.sidebar:
        st.markdown("### 🔌 Connexion Netis Group DWH")
        
        st.info(f"""
        **Base:** {st.session_state.db_config['database']}  
        **Schéma:** SCM  
        **Hôte:** {st.session_state.db_config['host']}  
        **Utilisateur:** {st.session_state.db_config['username']}
        """)
        
        with st.expander("📝 Personnaliser la requête"):
            custom_query = st.text_area(
                "Requête SQL",
                value=st.session_state.db_config['query'],
                height=150,
                help="Modifiez la requête selon vos besoins"
            )
            
            if st.button("Mettre à jour la requête"):
                st.session_state.db_config['query'] = custom_query
                st.success("Requête mise à jour !")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Tester", use_container_width=True):
                success, message = test_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)

        with col2:
            if st.button("📥 Charger", type="primary", use_container_width=True):
                with st.spinner("Connexion à Netis Group DWH..."):
                    df_raw = load_data_from_dwh()
                    
                    if not df_raw.empty:
                        with st.spinner("Normalisation des données..."):
                            df, cache = normalize_dataframe(df_raw)
                            
                            st.session_state.df = df
                            st.session_state.cache = cache
                            st.session_state.data_source = "dwh"
                            st.session_state.last_load = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            for k in ("groups_df", "members_df"):
                                if k in st.session_state:
                                    del st.session_state[k]
                            if 'selected_group' in st.session_state:
                                del st.session_state.selected_group
                            
                            st.success(f"✅ {len(df):,} lignes chargées")
                            st.rerun()

        if st.button("🔄 Rafraîchir", use_container_width=True):
            refresh_data()

        if 'df' in st.session_state:
            st.markdown("---")
            st.markdown("### 📈 Statistiques")
            df_stats = st.session_state.df
            total_items = len(df_stats)

            if total_items > 0:
                unique_names = df_stats['item_name'].nunique()
                duplicate_rate = ((total_items - unique_names) / total_items * 100) if total_items > 0 else 0

                st.metric("Items total", f"{total_items:,}")
                st.metric("Items uniques", f"{unique_names:,}")
                st.metric("Taux doublons", f"{duplicate_rate:.1f}%")
                
                if 'last_load' in st.session_state:
                    st.caption(f"Dernier chargement : {st.session_state.last_load}")

    if 'df' not in st.session_state:
        st.info("👈 Cliquez sur **Charger** pour vous connecter à Netis Group DWH")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🔌 Connexion automatique
            - **Base:** Netis-DW
            - **Schéma:** SCM
            - **Authentification:** it_powerbi_user
            - **Port:** 8822
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Données chargées
            - Requête par défaut sur la table `API_Items`
            - Filtre: created_at >= '2024-01-01'
            - Filtre: status = 'Qualified'
            """)
        
        with st.expander("ℹ️ Prérequis"):
            st.markdown("""
            **Assurez-vous d'être connecté au VPN Netis Group**
            
            **Installation des dépendances :**
            ```bash
            pip install sqlalchemy psycopg2-binary sentence-transformers pandas streamlit openpyxl
            ```
            """)
        return

    df = st.session_state.df
    cache = st.session_state.cache

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analyse Globale",
        "📝 Saisie & Vérification",
        "📊 Statistiques",
        "📤 Export"
    ])

    # === Tab 1 : Analyse Globale (CORRIGÉE avec persistance) ===
    with tab1:
        st.header("🧹 Détection globale des doublons")
        
        col_config1, col_config2, col_config3 = st.columns(3)

        with col_config1:
            available_cols = [col for col in [
                'item_name', 'company_name', 'type_name', 'category_name',
                'sub_category_name', 'uom_name'
            ] if col in df.columns]

            default_blocks = ['item_name', 'type_name', 'category_name']
            default_blocks = [col for col in default_blocks if col in available_cols]

            block_cols = st.multiselect(
                "Colonnes de blocage",
                options=available_cols,
                default=default_blocks
            )

        with col_config2:
            threshold = st.slider(
                "Seuil de similarité",
                min_value=0.60, max_value=0.95, value=0.82, step=0.01
            )
            max_block = st.number_input(
                "Taille max par bloc",
                min_value=100, max_value=5000, value=1000, step=100
            )

        with col_config3:
            use_semantic_global = st.checkbox(
                "🤖 Activer IA sémantique",
                value=True,
                help="Plus précis mais plus coûteux"
            )
            
            sampling = st.checkbox("Échantillonnage", value=True)
            if sampling and len(df) > 5000:
                sample_size = st.slider(
                    "Taille échantillon",
                    min_value=1000,
                    max_value=min(10000, len(df)),
                    value=min(5000, len(df)),
                    step=500
                )
            else:
                sample_size = len(df)

        if st.button("🚀 Lancer l'analyse globale", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                progress_bar = st.progress(0)
                
                analysis_df = df.sample(sample_size, random_state=42).copy() if sample_size < len(df) else df.copy()
                if sample_size < len(df):
                    st.info(f"🔬 Analyse sur échantillon de {sample_size} items")
                
                progress_bar.progress(30)
                groups_df, members_df = detect_global_duplicates_optimized(
                    analysis_df, cache, block_cols, threshold, max_block, use_semantic=use_semantic_global
                )
                progress_bar.progress(100)

                if len(groups_df) == 0:
                    st.success("🎉 Aucun doublon détecté !")
                else:
                    st.session_state.groups_df = groups_df.copy()
                    st.session_state.members_df = members_df.copy()
                    st.session_state.selected_group = None  # Reset de la sélection
                    st.rerun()

        # Afficher les résultats si disponibles
        if 'groups_df' in st.session_state and len(st.session_state.groups_df) > 0:
            groups_df = st.session_state.groups_df
            
            # ===== STATISTIQUES GLOBALES =====
            st.markdown("### 📊 Résumé des doublons")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Groupes de doublons", len(groups_df))
            with col_stat2:
                total_dupes = groups_df['size'].sum() - len(groups_df)
                st.metric("Items en doublon", int(total_dupes))
            with col_stat3:
                avg_group_size = groups_df['size'].mean()
                st.metric("Taille moyenne", f"{avg_group_size:.1f}")
            with col_stat4:
                max_group_size = groups_df['size'].max()
                st.metric("Plus grand groupe", int(max_group_size))

            # ===== LISTE DES GROUPES =====
            st.markdown("### 📋 Liste des groupes de doublons")
            
            display_cols = ['group_id', 'size', 'representative_name',
                           'representative_reference', 'avg_score', 'rule']
            display_cols = [c for c in display_cols if c in groups_df.columns]

            groups_display = groups_df[display_cols].copy()
            groups_display['avg_score'] = groups_display['avg_score'].round(3)
            
            st.dataframe(
                groups_display,
                use_container_width=True,
                column_config={
                    'group_id': "Groupe",
                    'size': "Taille",
                    'representative_name': "Nom représentant",
                    'representative_reference': "Référence",
                    'avg_score': st.column_config.ProgressColumn(
                        "Similarité",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=1.0
                    ),
                    'rule': "Règle"
                },
                height=400
            )

            # ===== DÉTAIL DES DOUBLONS PAR GROUPE (AVEC PERSISTANCE) =====
            if 'members_df' in st.session_state and len(st.session_state.members_df) > 0:
                st.markdown("### 👥 Détail des doublons par groupe")
                st.markdown("Sélectionnez un groupe pour voir tous ses membres")
                
                # Initialiser la sélection si elle n'existe pas
                if 'selected_group' not in st.session_state:
                    st.session_state.selected_group = None
                
                # Filtres
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    min_size = st.slider(
                        "Taille minimum du groupe",
                        min_value=2,
                        max_value=int(groups_df['size'].max()),
                        value=2,
                        key="min_size_slider"
                    )
                    filtered_groups = groups_df[groups_df['size'] >= min_size]
                
                with col_filter2:
                    search_term = st.text_input("🔍 Rechercher un groupe", 
                                               placeholder="Ex: TONER, CABLE...",
                                               key="group_search")
                    if search_term:
                        filtered_groups = filtered_groups[
                            filtered_groups['representative_name'].str.contains(search_term, case=False, na=False)
                        ]
                
                # Créer les options du selectbox
                if len(filtered_groups) > 0:
                    group_options = {
                        row['group_id']: f"Groupe {row['group_id']} - {row['representative_name']} ({row['size']} items)"
                        for _, row in filtered_groups.iterrows()
                    }
                    
                    # Déterminer l'index par défaut
                    group_ids = list(group_options.keys())
                    default_index = 0
                    if st.session_state.selected_group and st.session_state.selected_group in group_ids:
                        default_index = group_ids.index(st.session_state.selected_group)
                    
                    # Selectbox avec persistance
                    selected_group = st.selectbox(
                        "Choisir un groupe",
                        options=group_ids,
                        format_func=lambda x: group_options[x],
                        index=default_index,
                        key="group_selector"
                    )
                    
                    # Mettre à jour la session state
                    if selected_group != st.session_state.selected_group:
                        st.session_state.selected_group = selected_group
                        st.rerun()
                    
                    # Afficher les détails du groupe sélectionné
                    if st.session_state.selected_group:
                        group_members = st.session_state.members_df[
                            st.session_state.members_df['group_id'] == st.session_state.selected_group
                        ]
                        
                        group_info = groups_df[groups_df['group_id'] == st.session_state.selected_group].iloc[0]
                        
                        st.markdown(f"""
                        <div class="group-card">
                            <h4>Groupe {st.session_state.selected_group}</h4>
                            <p><strong>{group_info['representative_name']}</strong></p>
                            <p>Taille: {group_info['size']} items | Similarité: {group_info['avg_score']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Colonnes à afficher
                        member_cols = ['id', 'item_name', 'reference', 'type_name', 
                                      'category_name', 'company_name', 'created_at']
                        member_cols = [c for c in member_cols if c in group_members.columns]
                        
                        if 'code' in group_members.columns:
                            member_cols.insert(1, 'code')
                        
                        st.markdown(f"**{len(group_members)} items dans ce groupe**")
                        
                        st.dataframe(
                            group_members[member_cols],
                            use_container_width=True,
                            column_config={
                                'item_name': "Nom de l'item",
                                'reference': "Référence",
                                'code': "Code",
                                'type_name': "Type",
                                'category_name': "Catégorie",
                                'company_name': "Société",
                                'created_at': "Créé le"
                            },
                            height=300
                        )
                        
                        # Options d'export
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            csv_group = group_members.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                f"📥 Télécharger le groupe {st.session_state.selected_group} (CSV)",
                                data=csv_group,
                                file_name=f"groupe_{st.session_state.selected_group}_{group_info['representative_name'][:30]}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col_exp2:
                            if st.checkbox("Voir en plein écran", key="fullscreen"):
                                st.dataframe(group_members, use_container_width=True, height=600)
                else:
                    st.warning("Aucun groupe ne correspond aux critères")

                # ===== APERÇU GLOBAL DES DOUBLONS =====
                with st.expander("📋 Aperçu global des doublons (tous groupes confondus)"):
                    preview_cols = ['group_id', 'item_name', 'reference', 'category_name', 'type_name']
                    preview_cols = [c for c in preview_cols if c in st.session_state.members_df.columns]
                    
                    preview_df = st.session_state.members_df[preview_cols].sort_values(['group_id', 'item_name'])
                    
                    st.dataframe(
                        preview_df.head(200),
                        use_container_width=True,
                        column_config={
                            'group_id': "Groupe",
                            'item_name': "Nom",
                            'reference': "Réf",
                            'category_name': "Catégorie"
                        }
                    )
                    st.caption(f"Affichage des 200 premiers sur {len(preview_df)} doublons")

                # ===== EXPORT GLOBAL =====
                st.markdown("### 💾 Export des résultats")
                
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    csv_groups = groups_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Groupes (CSV)",
                        data=csv_groups,
                        file_name=f"groupes_doublons_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export2:
                    csv_members = st.session_state.members_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tous les doublons (CSV)",
                        data=csv_members,
                        file_name=f"tous_doublons_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export3:
                    try:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            groups_df.to_excel(writer, sheet_name='Groupes', index=False)
                            st.session_state.members_df.to_excel(writer, sheet_name='Doublons', index=False)
                            
                            stats_df = pd.DataFrame([
                                ["Total groupes", len(groups_df)],
                                ["Total doublons", groups_df['size'].sum() - len(groups_df)],
                                ["Taille moyenne", f"{groups_df['size'].mean():.1f}"],
                                ["Plus grand groupe", groups_df['size'].max()]
                            ], columns=["Indicateur", "Valeur"])
                            stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
                        
                        st.download_button(
                            "📥 Excel complet",
                            data=excel_buffer.getvalue(),
                            file_name=f"analyse_doublons_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning("Export Excel non disponible")

    # === Tab 2 : Saisie & Vérification ===
    with tab2:
        st.header("📝 Vérifier un nouvel item")

        with st.form("nouvel_item_form"):
            col_left, col_right = st.columns(2)

            with col_left:
                item_name = st.text_input("Nom de l'item *", "")
                french_name = st.text_input("Nom français", "")
                reference = st.text_input("Référence", "")
                uom_name = st.text_input("Unité de mesure", "")

            with col_right:
                type_name = st.text_input("Type", "")
                sub_category_name = st.text_input("Sous-catégorie", "")
                category_name = st.text_input("Catégorie", "")
                company_name = st.text_input("Société", "")

            st.markdown("---")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                topn = st.slider("Nombre de résultats", 3, 20, 8)
            with col_opt2:
                threshold_check = st.slider("Seuil de similarité", 0.60, 0.95, 0.82, 0.01)

            submitted = st.form_submit_button("🔍 Vérifier les doublons", type="primary")

        if submitted and item_name:
            with st.spinner("Recherche de doublons..."):
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
                semantic_input = f"{item_name} {french_name} {category_name} {type_name}"
                new_item.embedding = embed_texts([semantic_input])[0]

                duplicates = find_duplicates_fast(cache, new_item, topn, threshold_check)

                if not duplicates:
                    st.success("✅ Aucun doublon détecté !")
                else:
                    st.markdown(f"### 🔍 {len(duplicates)} doublon(s) potentiel(s) trouvé(s)")
                    
                    for item, score, rule in duplicates:
                        if rule == "exact_name":
                            badge_class = "badge-danger"
                            rule_text = "Exact"
                        elif rule == "same_ref":
                            badge_class = "badge-warning"
                            rule_text = "Même référence"
                        elif rule == "semantic_ai":
                            badge_class = "badge-info"
                            rule_text = "IA Sémantique"
                        else:
                            badge_class = "badge-info"
                            rule_text = "Similaire"
                        
                        st.markdown(f"""
                        <div class="duplicate-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{item.item_name}</strong><br>
                                    <small>Réf: {item.reference} | Cat: {item.category_name} | Type: {item.type_name}</small>
                                </div>
                                <div style="text-align: right;">
                                    <span class="{badge_class}">{score:.0%}</span><br>
                                    <small>{rule_text}</small>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # === Tab 3 : Statistiques ===
    with tab3:
        st.header("📊 Statistiques des données")
        
        if 'df' in st.session_state:
            total_items = len(df)
            unique_names = df['item_name'].nunique()
            unique_refs = df['reference'].nunique() if 'reference' in df.columns else 0

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total items", f"{total_items:,}")
            with col_stat2:
                st.metric("Noms uniques", f"{unique_names:,}")
            with col_stat3:
                duplicate_rate = ((total_items - unique_names) / total_items * 100) if total_items > 0 else 0
                st.metric("Taux doublons", f"{duplicate_rate:.1f}%")
            with col_stat4:
                st.metric("Références uniques", f"{unique_refs:,}")

            if 'category_name' in df.columns:
                st.markdown("### 📈 Distribution par catégorie")
                cat_dist = df['category_name'].value_counts().head(15)
                st.bar_chart(cat_dist)

            if 'type_name' in df.columns:
                st.markdown("### 📊 Distribution par type")
                type_dist = df['type_name'].value_counts().head(10)
                st.bar_chart(type_dist)

            if 'created_at' in df.columns:
                st.markdown("### 📅 Évolution temporelle")
                try:
                    df_time = df.copy()
                    df_time['created_at'] = pd.to_datetime(df_time['created_at'], errors='coerce')
                    df_time = df_time.dropna(subset=['created_at'])
                    if len(df_time) > 0:
                        time_series = df_time.set_index('created_at').resample('M').size()
                        st.line_chart(time_series)
                except:
                    pass

            if 'groups_df' in st.session_state:
                st.markdown("### 🎯 Statistiques des doublons")
                
                groups_df = st.session_state.groups_df
                
                col_dup1, col_dup2, col_dup3 = st.columns(3)
                with col_dup1:
                    st.metric("Groupes de doublons", len(groups_df))
                with col_dup2:
                    total_dupes = groups_df['size'].sum() - len(groups_df)
                    st.metric("Items en doublon", int(total_dupes))
                with col_dup3:
                    pct_dupes = (total_dupes / total_items * 100) if total_items > 0 else 0
                    st.metric("% en doublon", f"{pct_dupes:.1f}%")

    # === Tab 4 : Export ===
    with tab4:
        st.header("📤 Export des données")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("### 💾 Export standard")
            export_format = st.radio(
                "Format",
                ["CSV", "Excel", "JSON"],
                horizontal=True,
                key="export_format"
            )
            
            if export_format == "CSV":
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"items_netis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            elif export_format == "Excel":
                try:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Items')
                    st.download_button(
                        "📥 Télécharger Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"items_netis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except:
                    st.warning("Export Excel non disponible")
            else:
                json_data = df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    "📥 Télécharger JSON",
                    data=json_data,
                    file_name=f"items_netis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with col_exp2:
            st.markdown("### 📤 Export par Business Line")
            
            bl_options = []
            if 'company_name' in df.columns:
                bl_options.append(('company_name', 'Société'))
            if 'department_name' in df.columns:
                bl_options.append(('department_name', 'Département'))

            if bl_options:
                bl_key = st.radio(
                    "Clé de regroupement",
                    options=[k for k, _ in bl_options],
                    format_func=lambda x: dict(bl_options)[x],
                    key="bl_key"
                )

                if st.button("📦 Générer ZIP des Business Lines", use_container_width=True):
                    df_export = df.copy()

                    if 'members_df' in st.session_state and len(st.session_state.members_df) > 0:
                        df_groups = st.session_state.members_df[['id', 'group_id']].drop_duplicates()
                        df_export = df_export.merge(df_groups, on='id', how='left')
                    else:
                        tmp = build_temp_groups(df_export)
                        df_export['group_id'] = tmp['group_id']

                    if bl_key in df_export.columns:
                        keys = df_export[bl_key].fillna("Sans_BL").astype(str).unique().tolist()
                        
                        if keys:
                            progress = st.progress(0)
                            zip_buffer = BytesIO()

                            try:
                                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                                    for i, bl_value in enumerate(sorted(keys), 1):
                                        df_bl = df_export[df_export[bl_key].astype(str) == bl_value]
                                        
                                        excel_buffer = BytesIO()
                                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                                            df_bl.to_excel(writer, index=False, sheet_name="Items")
                                            
                                            df_dups = df_bl[df_bl["group_id"].notna()]
                                            if len(df_dups) > 0:
                                                df_dups.to_excel(writer, index=False, sheet_name="Doublons")
                                            
                                            total = len(df_bl)
                                            dups = len(df_dups)
                                            rate = f"{(dups/total*100):.1f}%" if total else "0.0%"
                                            
                                            summary = pd.DataFrame([
                                                ("Total items", total),
                                                ("Doublons", dups),
                                                ("Taux doublons", rate)
                                            ], columns=["Indicateur", "Valeur"])
                                            summary.to_excel(writer, index=False, sheet_name="Résumé")
                                            
                                            if 'category_name' in df_bl.columns:
                                                cat_counts = df_bl['category_name'].value_counts().reset_index()
                                                cat_counts.columns = ['Catégorie', "Nombre d'items"]
                                                cat_counts.to_excel(writer, index=False, sheet_name="Top_Catégories")
                                        
                                        file_name = f"{safe_filename(bl_value)}.xlsx"
                                        zipf.writestr(file_name, excel_buffer.getvalue())
                                        progress.progress(int(i / len(keys) * 100))

                                st.download_button(
                                    "📥 Télécharger ZIP",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"business_lines_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                                st.success("Export terminé !")
                            except Exception as e:
                                st.error(f"Erreur : {e}")
                        else:
                            st.warning("Aucune Business Line à exporter")
                    else:
                        st.warning(f"Colonne '{bl_key}' non trouvée")
            else:
                st.info("Aucune colonne Business Line détectée")

if __name__ == "__main__":
    main()