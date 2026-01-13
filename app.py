# ================================================================
# Items — Upload CSV → Détection de doublons → Saisie Multiple
# Auteur : Zineb FAKKAR – Janv 2026
# Optimisations : Cache, Vectorisation, Multithreading, Saisie multiple
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from io import BytesIO, StringIO
from datetime import datetime
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Any
import warnings
warnings.filterwarnings('ignore')
import logging
from enum import Enum
import json
import base64

# -------- Configuration des logs --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Enum pour les types de correspondance --------
class MatchType(Enum):
    EXACT_REF = "référence_identique"
    EXACT_NAME = "nom_identique"
    SIMILAR_REF = "référence_similaire"
    TEXT_SIMILARITY = "similarité_textuelle"
    LIGHT_SIMILARITY = "légèrement_similaire"

# -------- Constants --------
class Constants:
    DEFAULT_SIMILARITY_THRESHOLD = 0.82
    MAX_BLOCK_SIZE = 2500
    MAX_RESULTS = 10
    CACHE_TTL = 3600
    MAX_ITEMS_BATCH = 50

# -------- Données locales par défaut (MODIF NOUVELLE) --------
DEFAULT_FILE_NAME = "export.csv"
DEFAULT_FILE_PATH = Path(__file__).parent / DEFAULT_FILE_NAME  # IAAPP/ITEMAPP/export.csv

# -------- Configuration Streamlit --------
st.set_page_config(
    page_title="Items — Doublons Optimisé avec Saisie Multiple",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Dataclasses --------
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

# -------- Classes de cache --------
class ItemCache:
    """Cache pour les items avec indexation rapide"""
    def __init__(self):
        self._items_by_id = {}
        self._items_by_name_norm = {}
        self._items_by_ref_root = {}
        self._items_by_exact_ref = {}
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
        self._items_by_exact_ref = {}
        for item in items:
            if item.ref_root:
                self._items_by_ref_root.setdefault(item.ref_root, []).append(item)

            if item.reference:
                ref_clean = clean_reference(item.reference)
                if ref_clean:
                    self._items_by_exact_ref.setdefault(ref_clean, []).append(item)

    def get_by_id(self, item_id: str) -> Optional[Item]:
        """Récupère un item par son ID"""
        return self._items_by_id.get(item_id)

    def get_by_name_norm(self, name_norm: str) -> List[Item]:
        """Récupère tous les items avec un nom normalisé"""
        return self._items_by_name_norm.get(name_norm, [])

    def get_by_ref_root(self, ref_root: str) -> List[Item]:
        """Récupère tous les items avec une référence racine"""
        return self._items_by_ref_root.get(ref_root, [])

    def get_by_exact_reference(self, reference: str) -> List[Item]:
        """Récupère tous les items avec une référence exacte"""
        if not reference:
            return []
        ref_clean = clean_reference(reference)
        return self._items_by_exact_ref.get(ref_clean, [])

    @property
    def all_items(self) -> List[Item]:
        return self._all_items

    @property
    def search_texts(self) -> List[str]:
        return self._search_texts

# -------- Fonctions utilitaires --------
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

def clean_reference(ref: str) -> str:
    """Nettoie une référence pour comparaison"""
    if not ref:
        return ""
    ref = str(ref).strip().lower()
    ref = re.sub(r'[\s\-_\./]', '', ref)
    return ref

# -------- Lecture CSV --------
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

    sep = auto_detect_sep(uploaded_file_bytes)

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

    # Calculer les champs normalisés
    df["_item_name_norm"] = clean_text_batch(df["item_name"].tolist())
    df["_ref_root"] = ref_root_batch(df["reference"].tolist())
    df["_ref_clean"] = [clean_reference(ref) for ref in df["reference"].tolist()]

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

# -------- Similarité rapide --------
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

# -------- Détection de doublons --------
def find_duplicates_fast(cache: ItemCache, new_item: Item,
                        topn: int = 10, threshold: float = 0.82) -> List[Tuple[Item, float, str]]:
    """Trouve les doublons potentiels rapidement"""
    results = []
    processed_ids = set()

    # 1. Recherche par référence EXACTE
    if new_item.reference:
        ref_clean = clean_reference(new_item.reference)
        exact_ref_matches = [item for item in cache.all_items
                           if clean_reference(item.reference) == ref_clean
                           and item.id != new_item.id]

        for item in exact_ref_matches:
            if item.id not in processed_ids:
                results.append((item, 1.0, MatchType.EXACT_REF.value))
                processed_ids.add(item.id)

    # 2. Recherche exacte par nom normalisé
    if new_item.item_name_norm:
        exact_matches = cache.get_by_name_norm(new_item.item_name_norm)
        for item in exact_matches:
            if item.id != new_item.id and item.id not in processed_ids:
                results.append((item, 1.0, MatchType.EXACT_NAME.value))
                processed_ids.add(item.id)

    if len(results) >= topn:
        return sorted(results, key=lambda x: x[1], reverse=True)[:topn]

    # 3. Recherche par référence racine
    ref_matches = cache.get_by_ref_root(new_item.ref_root)
    for item in ref_matches:
        if item.id != new_item.id and item.id not in processed_ids:
            score = 0.9
            results.append((item, score, MatchType.SIMILAR_REF.value))
            processed_ids.add(item.id)

    # 4. Recherche par similarité textuelle
    new_dupe_text = new_item.dupe_text or new_item.search_text
    if new_dupe_text and len(cache.search_texts) > 0:
        max_samples = min(1000, len(cache.all_items))
        sample_indices = np.random.choice(len(cache.all_items), max_samples, replace=False)

        for idx in sample_indices:
            item = cache.all_items[idx]
            if item.id == new_item.id or item.id in processed_ids:
                continue

            jaccard_sim = FastSimilarity.jaccard_similarity(new_dupe_text, item.dupe_text or item.search_text)

            if jaccard_sim >= threshold:
                bonus = 0.1 if new_item.category_name == item.category_name else 0
                bonus += 0.05 if new_item.type_name == item.type_name else 0
                final_score = min(jaccard_sim + bonus, 1.0)

                match_type = MatchType.TEXT_SIMILARITY.value
                if final_score <= 0.8:
                    match_type = MatchType.LIGHT_SIMILARITY.value

                results.append((item, final_score, match_type))
                processed_ids.add(item.id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topn]

# -------- Détection globale --------
def detect_global_duplicates_optimized(df: pd.DataFrame, cache: ItemCache,
                                      block_cols: List[str],
                                      threshold: float = 0.82,
                                      max_block_size: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Détection globale de doublons"""
    start_time = time.time()

    if len(df) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    # Détecter les références identiques
    ref_duplicates = {}
    ref_to_items = {}

    for _, row in df.iterrows():
        ref_clean = clean_reference(row.get('reference', ''))
        if ref_clean:
            ref_to_items.setdefault(ref_clean, []).append(row.to_dict())

    group_id = 1
    duplicate_refs = {ref: items for ref, items in ref_to_items.items() if len(items) > 1}

    if duplicate_refs:
        groups_data = []
        members_data = []

        for ref_clean, items in duplicate_refs.items():
            group_size = len(items)
            representative = items[0]

            groups_data.append({
                'group_id': group_id,
                'size': group_size,
                'representative_reference': representative.get('reference', ''),
                'representative_name': representative.get('item_name', ''),
                'rule': MatchType.EXACT_REF.value,
                'avg_score': 1.0,
                'reference': ref_clean
            })

            for i, item in enumerate(items):
                item_df = pd.DataFrame([item])
                item_df.insert(0, 'group_id', group_id)
                item_df.insert(1, 'is_representative', 1 if i == 0 else 0)
                members_data.append(item_df)

            group_id += 1

        if groups_data:
            ref_groups_df = pd.DataFrame(groups_data)
            ref_members_df = pd.concat(members_data, ignore_index=True)

    # Détection classique pour autres doublons
    work_df = df.copy()

    if block_cols:
        work_df["_block_key"] = work_df[block_cols].fillna("").astype(str).agg("|".join, axis=1)
    else:
        work_df["_block_key"] = "ALL"

    groups = []
    all_members = []
    blocks = list(work_df.groupby("_block_key"))

    def process_block(block_key: str, block_data: pd.DataFrame) -> List[Dict]:
        """Traite un bloc de données"""
        block_results = []

        if len(block_data) <= 1:
            return block_results

        if len(block_data) > max_block_size:
            sample_size = min(max_block_size, len(block_data))
            block_data = block_data.sample(sample_size, random_state=42)

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

        n = len(block_items)
        processed_pairs = set()

        for i in range(n):
            item_i = block_items[i]

            for j in range(i + 1, n):
                item_j = block_items[j]

                if clean_reference(item_i.reference) == clean_reference(item_j.reference):
                    continue

                pair_key = tuple(sorted([item_i.id, item_j.id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                sim = FastSimilarity.jaccard_similarity(
                    item_i.dupe_text or item_i.search_text,
                    item_j.dupe_text or item_j.search_text
                )

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
        futures = {executor.submit(process_block, key, data): (key, data)
                  for key, data in blocks if len(data) > 1}

        for future in as_completed(futures):
            block_results = future.result()
            all_pairs.extend(block_results)

    if all_pairs:
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

        all_items_set = set()
        for pair in all_pairs:
            all_items_set.add(pair['item_i'].id)
            all_items_set.add(pair['item_j'].id)

        for item_id in all_items_set:
            parent[item_id] = item_id
            rank[item_id] = 0

        for pair in all_pairs:
            union(pair['item_i'].id, pair['item_j'].id)

        components = {}
        for item_id in all_items_set:
            root = find(item_id)
            components.setdefault(root, []).append(item_id)

        for root, item_ids in components.items():
            if len(item_ids) <= 1:
                continue

            group_df = work_df[work_df['id'].isin(item_ids)].copy()
            group_df['ref_len'] = group_df['reference'].str.len()
            rep_idx = group_df['ref_len'].idxmax()
            representative = group_df.loc[rep_idx]

            group_scores = [p['score'] for p in all_pairs
                          if p['item_i'].id in item_ids or p['item_j'].id in item_ids]
            avg_score = np.mean(group_scores) if group_scores else 0

            groups_data.append({
                'group_id': group_id,
                'size': len(group_df),
                'representative_reference': representative.get('reference', ''),
                'representative_name': representative.get('item_name', ''),
                'rule': MatchType.TEXT_SIMILARITY.value,
                'avg_score': avg_score,
                'reference': 'multiple'
            })

            group_df = group_df.drop(columns=['ref_len', '_block_key'], errors='ignore')
            group_df.insert(0, 'group_id', group_id)
            group_df.insert(1, 'is_representative', group_df.index == rep_idx)
            members_data.append(group_df)

            group_id += 1

    if 'groups_data' in locals() and groups_data:
        all_groups_df = pd.DataFrame(groups_data)
        all_members_df = pd.concat(members_data, ignore_index=True) if members_data else pd.DataFrame()

        all_groups_df['sort_key'] = all_groups_df['rule'].apply(
            lambda x: 0 if x == MatchType.EXACT_REF.value else 1
        )
        all_groups_df = all_groups_df.sort_values(['sort_key', 'size', 'avg_score'],
                                                 ascending=[True, False, False])
        all_groups_df = all_groups_df.drop(columns=['sort_key'])

        execution_time = time.time() - start_time
        st.success(f"✅ Analyse terminée en {execution_time:.2f} secondes")
        st.info(f"📊 Total: {len(all_groups_df)} groupes de doublons détectés")

        return all_groups_df, all_members_df

    return pd.DataFrame(), pd.DataFrame()

# -------- Nouvelle fonction pour saisie multiple --------
def batch_verify_items(items_list: List[Dict], cache: ItemCache, threshold: float = 0.82) -> Dict[str, Any]:
    """Vérifie une liste d'items en lot"""
    results = {
        'total': len(items_list),
        'duplicates': 0,
        'uniques': 0,
        'items': []
    }

    for item_data in items_list:
        new_item = Item(
            id=f"BATCH_{hash(str(item_data))}",
            item_name=item_data.get('item_name', ''),
            french_name=item_data.get('french_name', ''),
            reference=item_data.get('reference', ''),
            type_name=item_data.get('type_name', ''),
            category_name=item_data.get('category_name', ''),
            company_name=item_data.get('company_name', ''),
            item_name_norm=clean_text_batch([item_data.get('item_name', '')])[0],
            ref_root=ref_root_batch([item_data.get('reference', '')])[0] if item_data.get('reference') else "",
            dupe_text=clean_text_batch([
                f"{item_data.get('item_name', '')} {item_data.get('french_name', '')} "
                f"{item_data.get('reference', '')} {item_data.get('type_name', '')} "
                f"{item_data.get('category_name', '')}"
            ])[0]
        )

        duplicates = find_duplicates_fast(cache, new_item, topn=5, threshold=threshold)

        is_duplicate = any(
            match_type in [MatchType.EXACT_REF.value, MatchType.EXACT_NAME.value]
            for _, _, match_type in duplicates
        )

        item_result = {
            'input_data': item_data,
            'item': new_item,
            'duplicates': duplicates,
            'is_duplicate': is_duplicate,
            'match_count': len(duplicates),
            'exact_matches': [d for d in duplicates if d[2] in [MatchType.EXACT_REF.value, MatchType.EXACT_NAME.value]],
            'status': 'doublon' if is_duplicate else 'unique'
        }

        results['items'].append(item_result)

        if is_duplicate:
            results['duplicates'] += 1
        else:
            results['uniques'] += 1

    return results

# -------- Fonction pour importer une liste --------
def import_items_from_text(text: str) -> List[Dict]:
    """Importe une liste d'items depuis un texte"""
    items = []
    lines = text.strip().split('\n')

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue

        parts = [part.strip() for part in line.split(';')]

        if len(parts) >= 2:
            item = {
                'id': f"import_{line_num}",
                'reference': parts[0] if len(parts) > 0 else '',
                'item_name': parts[1] if len(parts) > 1 else '',
                'french_name': parts[2] if len(parts) > 2 else '',
                'type_name': parts[3] if len(parts) > 3 else '',
                'category_name': parts[4] if len(parts) > 4 else '',
                'company_name': parts[5] if len(parts) > 5 else '',
                'status': 'non_verifie'
            }
            items.append(item)
        elif len(parts) == 1:
            item = {
                'id': f"import_{line_num}",
                'item_name': parts[0],
                'reference': '',
                'french_name': '',
                'type_name': '',
                'category_name': '',
                'company_name': '',
                'status': 'non_verifie'
            }
            items.append(item)

    return items

# -------- Fonction pour détecter les colonnes --------
def detect_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """Détecte automatiquement les noms de colonnes dans le CSV"""
    column_mapping = {}

    # Dictionnaire des noms possibles pour chaque colonne attendue
    possible_names = {
        'reference': ['reference', 'ref', 'référence', 'code', 'id', 'sku', 'article', 'numéro', 'n°', 'no'],
        'item_name': ['item_name', 'nom', 'name', 'article', 'description', 'désignation', 'produit', 'libellé', 'libelle'],
        'french_name': ['french_name', 'nom_français', 'nom_fr', 'libelle_fr', 'description_fr'],
        'type_name': ['type_name', 'type', 'catégorie', 'family', 'famille'],
        'category_name': ['category_name', 'categorie', 'catégorie', 'groupe', 'family'],
        'company_name': ['company_name', 'société', 'fournisseur', 'supplier', 'vendor']
    }

    # Convertir tous les noms de colonnes en minuscules sans accents
    df_columns_lower = {}
    for col in df.columns:
        # Nettoyer le nom de colonne
        col_clean = str(col).lower().strip()
        col_clean = unicodedata.normalize('NFKD', col_clean)
        col_clean = ''.join(c for c in col_clean if not unicodedata.combining(c))
        df_columns_lower[col_clean] = col

    for target_col, possible_list in possible_names.items():
        found = False
        for possible_name in possible_list:
            possible_clean = possible_name.lower()
            if possible_clean in df_columns_lower:
                column_mapping[target_col] = df_columns_lower[possible_clean]
                found = True
                break

        # Recherche partielle
        if not found:
            for col_clean, original_col in df_columns_lower.items():
                for possible_name in possible_list:
                    if possible_name in col_clean or col_clean in possible_name:
                        column_mapping[target_col] = original_col
                        found = True
                        break
                if found:
                    break

        if not found:
            column_mapping[target_col] = None

    return column_mapping

# -------- Chargement local si disponible (MODIF NOUVELLE) --------
def load_local_csv_if_available() -> Optional[Tuple[pd.DataFrame, ItemCache, str]]:
    """
    Charge export.csv localement s'il existe. Retourne (df, cache, filename) ou None si absent.
    """
    try:
        if DEFAULT_FILE_PATH.exists() and DEFAULT_FILE_PATH.is_file():
            file_bytes = DEFAULT_FILE_PATH.read_bytes()
            df, cache = read_and_normalize_df(file_bytes, DEFAULT_FILE_PATH.name)
            return df, cache, DEFAULT_FILE_PATH.name
    except Exception as e:
        st.warning(f"Impossible de lire le fichier local {DEFAULT_FILE_PATH}: {e}")
    return None

# -------- Interface Streamlit --------
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

        .duplicate-highlight {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }

        .exact-match {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }

        .ref-match {
            background-color: #d1ecf1;
            border-left: 4px solid #0dcaf0;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }

        .similar-match {
            background-color: #e2e3e5;
            border-left: 4px solid #6c757d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }

        .badge-ref {
            background: #0dcaf0;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .badge-exact {
            background: #dc3545;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .badge-similar {
            background: #6c757d;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .item-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .unique-item {
            border-left: 5px solid #28a745;
        }

        .duplicate-item {
            border-left: 5px solid #dc3545;
        }

        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            display: inline-block;
        }

        .status-unique {
            background-color: #d4edda;
            color: #155724;
        }

        .status-duplicate {
            background-color: #f8d7da;
            color: #721c24;
        }

        .stSelectbox label {
            font-size: 14px;
            font-weight: 500;
        }

        .column-match {
            background-color: #e8f4fd;
            padding: 8px;
            border-radius: 6px;
            margin: 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🚀 Détection de Doublons - Saisie Multiple</h1>', unsafe_allow_html=True)

    # Sidebar - Source des données (MODIF NOUVELLE)
    with st.sidebar:
        st.markdown("### 📤 Source des données")

        local_file_available = DEFAULT_FILE_PATH.exists()
        use_local = False

        if local_file_available:
            # Toggle pour utiliser le fichier local
            use_local = st.checkbox(
                f"Utiliser le fichier local **{DEFAULT_FILE_NAME}**",
                value=True,
                help=f"Charge {DEFAULT_FILE_NAME} depuis {DEFAULT_FILE_PATH}"
            )
        else:
            st.info(f"ℹ️ Fichier local {DEFAULT_FILE_NAME} introuvable à {DEFAULT_FILE_PATH}. Utilisez l'upload ci-dessous.")

        uploaded_file = None
        if not use_local:
            uploaded_file = st.file_uploader(
                "Téléversez votre fichier CSV",
                type=['csv'],
                help="Format CSV avec UTF-8 ou Latin-1"
            )

        # Chargement des données (local prioritaire)
        if use_local:
            if 'file_hash' not in st.session_state or st.session_state.get('source') != 'local':
                with st.spinner(f"Chargement du fichier local {DEFAULT_FILE_NAME}..."):
                    res = load_local_csv_if_available()
                    if res is not None:
                        df, cache, filename = res
                        st.session_state.df = df
                        st.session_state.cache = cache
                        try:
                            st.session_state.file_hash = hashlib.md5(DEFAULT_FILE_PATH.read_bytes()).hexdigest()
                        except Exception:
                            st.session_state.file_hash = f"local_{int(time.time())}"
                        st.session_state.filename = filename
                        st.session_state.source = 'local'

                        st.success(f"✅ Fichier local chargé : {filename}")
                        st.metric("Lignes", len(df))
                        st.metric("Colonnes", len(df.columns))
                    else:
                        st.error("❌ Échec du chargement du fichier local. Veuillez utiliser l'upload.")
        else:
            # Mode upload (comme avant)
            if uploaded_file:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()

                if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash or st.session_state.get('source') != 'upload':
                    with st.spinner("Chargement et optimisation en cours..."):
                        df, cache = read_and_normalize_df(file_bytes, uploaded_file.name)
                        st.session_state.df = df
                        st.session_state.cache = cache
                        st.session_state.file_hash = file_hash
                        st.session_state.filename = uploaded_file.name
                        st.session_state.source = 'upload'

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
                # Préserver la source (local/upload)
                source = st.session_state.get('source')
                for key in list(st.session_state.keys()):
                    if key not in ('source',):
                        del st.session_state[key]
                st.session_state.source = source
                st.rerun()

        with col2:
            if st.button("🧹 Nettoyer saisie", use_container_width=True):
                if 'items_a_verifier' in st.session_state:
                    st.session_state.items_a_verifier = []
                if 'batch_results' in st.session_state:
                    del st.session_state.batch_results
                st.rerun()

    # Initialisation de la session
    if 'items_a_verifier' not in st.session_state:
        st.session_state.items_a_verifier = []

    # Main content
    if 'df' not in st.session_state:
        st.info("👈 Chargez un fichier (local ou upload) dans la barre latérale pour commencer.")
        return

    # Données chargées
    df = st.session_state.df
    cache = st.session_state.cache

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs([
        "🔍 Analyse Globale",
        "📝 Saisie Multiple",
        "📊 Statistiques & Export"
    ])

    # Tab 1 - Analyse Globale
    with tab1:
        st.header("🧹 Détection globale des doublons")

        col_config1, col_config2 = st.columns(2)

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
                default=default_blocks,
                help="Réduit les comparaisons aux items similaires"
            )

        with col_config2:
            threshold = st.slider(
                "Seuil de similarité textuelle",
                min_value=0.60,
                max_value=0.95,
                value=Constants.DEFAULT_SIMILARITY_THRESHOLD,
                step=0.01
            )

            max_block = st.number_input(
                "Taille max par bloc",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )

        if st.button("🚀 Lancer l'analyse globale", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                groups_df, members_df = detect_global_duplicates_optimized(
                    df, cache, block_cols, threshold, max_block
                )

                if len(groups_df) == 0:
                    st.success("🎉 Aucun doublon détecté !")
                else:
                    st.markdown(f"### 📊 Résultats : {len(groups_df)} groupes de doublons détectés")

                    ref_identical = len(groups_df[groups_df['rule'] == MatchType.EXACT_REF.value])
                    text_similar = len(groups_df[groups_df['rule'] == MatchType.TEXT_SIMILARITY.value])

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Groupes", len(groups_df))
                    with col_stat2:
                        st.metric("Références identiques", ref_identical)
                    with col_stat3:
                        st.metric("Similitudes textuelles", text_similar)

                    with st.expander("📋 Liste des groupes", expanded=True):
                        for _, group in groups_df.iterrows():
                            rule = group['rule']
                            size = group['size']
                            ref = group['representative_reference']
                            name = group['representative_name']
                            score = group['avg_score']

                            if rule == MatchType.EXACT_REF.value:
                                badge = '<span class="badge-exact">Référence identique</span>'
                                css_class = "exact-match"
                            else:
                                badge = '<span class="badge-similar">Similarité textuelle</span>'
                                css_class = "similar-match"

                            st.markdown(f"""
                            <div class="{css_class}">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>
                                        <strong>Groupe {group['group_id']} ({size} items)</strong><br>
                                        {name}<br>
                                        <small>Référence: {ref}</small>
                                    </div>
                                    <div style="text-align: right;">
                                        {badge}<br>
                                        <strong>Score: {score:.1%}</strong>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("### 💾 Export des résultats")
                    col_export1, col_export2 = st.columns(2)

                    with col_export1:
                        csv_groups = groups_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "📥 Télécharger les groupes",
                            data=csv_groups,
                            file_name="groupes_doublons.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col_export2:
                        if len(members_df) > 0:
                            csv_members = members_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                "📥 Télécharger tous les membres",
                                data=csv_members,
                                file_name="membres_doublons.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

    # Tab 2 - Saisie Multiple (AMÉLIORÉ)
    with tab2:
        st.header("📝 Vérification de plusieurs items")

        # Mode de saisie
        saisie_mode = st.radio(
            "Mode de saisie :",
            ["📝 Saisie manuelle", "📋 Coller une liste", "📁 Importer CSV"],
            horizontal=True
        )

        if saisie_mode == "📝 Saisie manuelle":
            # Interface de saisie multiple
            st.markdown("### 🔢 Ajouter plusieurs items")

            # Boutons d'action
            col_add, col_clear, col_sample = st.columns([2, 1, 1])
            with col_add:
                if st.button("➕ Ajouter un item", use_container_width=True):
                    new_item = {
                        'id': f"item_{len(st.session_state.items_a_verifier)}_{int(time.time())}",
                        'item_name': '',
                        'reference': '',
                        'french_name': '',
                        'type_name': '',
                        'category_name': '',
                        'company_name': '',
                        'status': 'non_verifie'
                    }
                    st.session_state.items_a_verifier.append(new_item)
                    st.rerun()

            with col_clear:
                if st.button("🗑️ Tout effacer", use_container_width=True):
                    st.session_state.items_a_verifier = []
                    if 'batch_results' in st.session_state:
                        del st.session_state.batch_results
                    st.rerun()

            with col_sample:
                if st.button("🎯 Ajouter exemple", use_container_width=True):
                    sample_items = [
                        {
                            'id': f"sample_{int(time.time())}_1",
                            'item_name': 'Vis à tête hexagonale 10mm',
                            'reference': 'VTH-10',
                            'french_name': 'Vis hexagonale',
                            'type_name': 'Vis',
                            'category_name': 'Quincaillerie',
                            'company_name': 'Fournisseur A',
                            'status': 'non_verifie'
                        },
                        {
                            'id': f"sample_{int(time.time())}_2",
                            'item_name': 'Tube PVC diamètre 20mm',
                            'reference': 'TPVC-20',
                            'french_name': 'Tuyau PVC',
                            'type_name': 'Tuyauterie',
                            'category_name': 'Plomberie',
                            'company_name': 'Fournisseur B',
                            'status': 'non_verifie'
                        }
                    ]
                    st.session_state.items_a_verifier.extend(sample_items)
                    st.rerun()

            # Affichage des items existants
            if st.session_state.items_a_verifier:
                st.markdown(f"### 📋 Liste à vérifier ({len(st.session_state.items_a_verifier)} items)")

                # Formulaire pour chaque item
                for idx, item_data in enumerate(st.session_state.items_a_verifier):
                    with st.expander(f"Item #{idx+1} - {item_data['item_name'][:30] or 'Nouvel item'}",
                                     expanded=idx < 3):
                        col1, col2 = st.columns(2)

                        with col1:
                            item_name = st.text_input(
                                "Nom de l'item *",
                                value=item_data['item_name'],
                                key=f"name_{item_data['id']}",
                                placeholder="Ex: Vis à tête hexagonale 10mm"
                            )
                            french_name = st.text_input(
                                "Nom français",
                                value=item_data.get('french_name', ''),
                                key=f"french_{item_data['id']}",
                                placeholder="Ex: Vis hexagonale"
                            )
                            reference = st.text_input(
                                "Référence *",
                                value=item_data.get('reference', ''),
                                key=f"ref_{item_data['id']}",
                                placeholder="Ex: VTH-10"
                            )

                        with col2:
                            type_name = st.text_input(
                                "Type",
                                value=item_data.get('type_name', ''),
                                key=f"type_{item_data['id']}",
                                placeholder="Ex: Vis"
                            )
                            category_name = st.text_input(
                                "Catégorie",
                                value=item_data.get('category_name', ''),
                                key=f"category_{item_data['id']}",
                                placeholder="Ex: Quincaillerie"
                            )
                            company_name = st.text_input(
                                "Société",
                                value=item_data.get('company_name', ''),
                                key=f"company_{item_data['id']}",
                                placeholder="Ex: Fournisseur A"
                            )

                        # Boutons d'action pour cet item
                        col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
                        with col_act1:
                            if st.button("🗑️", key=f"del_{item_data['id']}",
                                         help="Supprimer cet item"):
                                st.session_state.items_a_verifier.pop(idx)
                                st.rerun()

                        with col_act2:
                            if st.button("📋", key=f"copy_{item_data['id']}",
                                         help="Dupliquer cet item"):
                                new_item = item_data.copy()
                                new_item['id'] = f"copy_{item_data['id']}_{int(time.time())}"
                                st.session_state.items_a_verifier.append(new_item)
                                st.rerun()

                        with col_act3:
                            # Afficher le statut actuel
                            if item_data.get('status') == 'doublon':
                                st.warning("⚠️ Doublon détecté")
                            elif item_data.get('status') == 'unique':
                                st.success("✅ Unique")

                        # Mettre à jour l'item
                        item_data.update({
                            'item_name': item_name,
                            'reference': reference,
                            'french_name': french_name,
                            'type_name': type_name,
                            'category_name': category_name,
                            'company_name': company_name
                        })

            # Paramètres de vérification
            st.markdown("---")
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                threshold_check = st.slider(
                    "Seuil de similarité",
                    min_value=0.60,
                    max_value=0.95,
                    value=Constants.DEFAULT_SIMILARITY_THRESHOLD,
                    step=0.01,
                    key="threshold_batch"
                )

            with col_param2:
                max_results = st.slider(
                    "Max résultats par item",
                    min_value=3,
                    max_value=20,
                    value=5,
                    step=1
                )

            # Bouton de vérification globale
            if st.session_state.items_a_verifier:
                col_check, col_export_list = st.columns([2, 1])

                with col_check:
                    if st.button("🔍 Vérifier TOUS les items", type="primary", use_container_width=True):
                        with st.spinner(f"Vérification de {len(st.session_state.items_a_verifier)} items..."):
                            # Vérifier tous les items
                            results = batch_verify_items(
                                st.session_state.items_a_verifier,
                                cache,
                                threshold_check
                            )

                            # Mettre à jour les statuts
                            for idx, item_data in enumerate(st.session_state.items_a_verifier):
                                for item_result in results['items']:
                                    if item_result['input_data']['id'] == item_data['id']:
                                        item_data['status'] = item_result['status']
                                        break

                            st.session_state.batch_results = results
                            st.success(f"✅ Vérification terminée !")

                with col_export_list:
                    # Exporter la liste
                    export_df = pd.DataFrame(st.session_state.items_a_verifier)
                    csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Exporter la liste",
                        data=csv_data,
                        file_name="liste_items_a_verifier.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Afficher les résultats
                if 'batch_results' in st.session_state:
                    results = st.session_state.batch_results

                    st.markdown("### 📊 Résultats de la vérification")

                    # Statistiques
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total items", results['total'])
                    with col_stat2:
                        st.metric("Doublons détectés", results['duplicates'],
                                  delta_color="inverse")
                    with col_stat3:
                        st.metric("Items uniques", results['uniques'])

                    # Onglets de résultats
                    tab_res1, tab_res2 = st.tabs(["📋 Vue synthèse", "🔍 Détails complets"])

                    with tab_res1:
                        # Tableau récapitulatif
                        summary_data = []
                        for item_result in results['items']:
                            input_data = item_result['input_data']
                            summary_data.append({
                                'Nom': input_data['item_name'],
                                'Référence': input_data['reference'],
                                'Catégorie': input_data['category_name'],
                                'Statut': '🚨 Doublon' if item_result['is_duplicate'] else '✅ Unique',
                                'Matches trouvés': item_result['match_count'],
                                'Matches exacts': len(item_result['exact_matches'])
                            })

                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(
                                summary_df,
                                use_container_width=True,
                                hide_index=True
                            )

                    with tab_res2:
                        # Détails par item
                        for idx, item_result in enumerate(results['items'], 1):
                            input_data = item_result['input_data']
                            status_class = "duplicate-item" if item_result['is_duplicate'] else "unique-item"
                            status_badge = "status-duplicate" if item_result['is_duplicate'] else "status-unique"
                            status_text = "DOUBLON" if item_result['is_duplicate'] else "UNIQUE"

                            with st.container():
                                st.markdown(f"""
                                <div class="item-card {status_class}">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <h4>Item #{idx}: {input_data['item_name']}</h4>
                                            <p><strong>Référence:</strong> {input_data['reference']}</p>
                                            <p><strong>Catégorie:</strong> {input_data['category_name']}</p>
                                        </div>
                                        <div>
                                            <span class="status-badge {status_badge}">{status_text}</span>
                                            <p style="text-align: center; margin-top: 5px;">
                                                <strong>{item_result['match_count']}</strong> match(s)
                                            </p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                                # Afficher les doublons détectés
                                if item_result['duplicates']:
                                    with st.expander(f"Voir les {len(item_result['duplicates'])} match(s)"):
                                        for match_idx, (item, score, match_type) in enumerate(item_result['duplicates'], 1):
                                            if match_type == MatchType.EXACT_REF.value:
                                                css_class = "exact-match"
                                            elif match_type == MatchType.EXACT_NAME.value:
                                                css_class = "exact-match"
                                            elif match_type == MatchType.SIMILAR_REF.value:
                                                css_class = "ref-match"
                                            else:
                                                css_class = "similar-match"

                                            st.markdown(f"""
                                            <div class="{css_class}" style="margin: 5px 0;">
                                                <div style="display: flex; justify-content: space-between;">
                                                    <div>
                                                        <strong>{item.item_name}</strong><br>
                                                        <small>Réf: {item.reference} | Cat: {item.category_name}</small>
                                                    </div>
                                                    <div>
                                                        <strong>{score:.0%}</strong><br>
                                                        <small>{match_type}</small>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                st.markdown("</div>", unsafe_allow_html=True)

                    # Export des résultats détaillés
                    st.markdown("---")
                    st.markdown("### 💾 Export des résultats")

                    if st.button("📥 Exporter tous les résultats", use_container_width=True):
                        # Préparer les données d'export
                        export_data = []
                        for item_result in results['items']:
                            input_data = item_result['input_data']
                            for match_idx, (item, score, match_type) in enumerate(item_result['duplicates'], 1):
                                export_data.append({
                                    'Item_Nom': input_data['item_name'],
                                    'Item_Reference': input_data['reference'],
                                    'Item_Categorie': input_data['category_name'],
                                    'Match_Nom': item.item_name,
                                    'Match_Reference': item.reference,
                                    'Match_Categorie': item.category_name,
                                    'Score': score,
                                    'Type_Match': match_type,
                                    'Est_Doublon': 'OUI' if item_result['is_duplicate'] else 'NON'
                                })

                        if export_data:
                            export_df = pd.DataFrame(export_data)
                            csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                "📥 Télécharger résultats détaillés",
                                data=csv_data,
                                file_name="resultats_doublons_detailles.csv",
                                mime="text/csv"
                            )

        elif saisie_mode == "📋 Coller une liste":
            st.markdown("### 📋 Coller une liste d'items")
            st.markdown("""
            **Format attendu :** Chaque ligne = un item, séparé par des points-virgules
            ```
            Référence;Nom;Nom français;Type;Catégorie;Société
            VTH-10;Vis à tête hexagonale 10mm;Vis hexagonale;Vis;Quincaillerie;Fournisseur A
            TPVC-20;Tube PVC diamètre 20mm;Tuyau PVC;Tuyauterie;Plomberie;Fournisseur B
            ```
            *Ou simplement les noms sur chaque ligne*
            """)

            text_input = st.text_area(
                "Collez votre liste ici :",
                height=200,
                placeholder="Exemple:\nVTH-10;Vis à tête hexagonale 10mm;Vis;Quincaillerie\nTPVC-20;Tube PVC 20mm;Tuyauterie;Plomberie"
            )

            if st.button("📥 Importer la liste", use_container_width=True):
                if text_input.strip():
                    imported_items = import_items_from_text(text_input)
                    st.session_state.items_a_verifier.extend(imported_items)
                    st.success(f"✅ {len(imported_items)} items importés !")
                    st.rerun()
                else:
                    st.error("Veuillez coller une liste d'items")

        elif saisie_mode == "📁 Importer CSV":
            st.markdown("### 📁 Importer un fichier CSV")
            st.markdown("""
            **Format flexible :** Le programme reconnaîtra automatiquement vos colonnes
            """)

            uploaded_list = st.file_uploader(
                "Téléversez votre fichier CSV",
                type=['csv'],
                key="batch_csv"
            )

            if uploaded_list:
                try:
                    # Essayer différents encodages
                    try:
                        df_import = pd.read_csv(uploaded_list, dtype=str, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df_import = pd.read_csv(uploaded_list, dtype=str, encoding='latin-1')
                        except:
                            df_import = pd.read_csv(uploaded_list, dtype=str, encoding='iso-8859-1')

                    # Nettoyer les noms de colonnes
                    df_import.columns = df_import.columns.str.strip().str.lower()

                    # Afficher les colonnes détectées
                    st.success(f"✅ Fichier chargé : {len(df_import)} lignes, {len(df_import.columns)} colonnes")

                    with st.expander("👁️ Colonnes détectées dans votre fichier"):
                        cols_df = pd.DataFrame({
                            'Nom de colonne': df_import.columns,
                            'Type': df_import.dtypes.astype(str).values,
                            'Valeurs uniques': df_import.nunique().values,
                            'Exemple': df_import.iloc[0].apply(lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x))
                        })
                        st.dataframe(cols_df, use_container_width=True, hide_index=True)

                    # Détection automatique des colonnes
                    column_mapping = detect_column_names(df_import)

                    st.markdown("### 🗺️ Correspondance des colonnes détectées")

                    # Afficher la correspondance
                    mapping_display = []
                    for target_col in ['reference', 'item_name', 'french_name', 'type_name', 'category_name', 'company_name']:
                        source_col = column_mapping.get(target_col)
                        if source_col:
                            status = "✅"
                            example = str(df_import[source_col].iloc[0])[:50] if not df_import.empty else ""
                        else:
                            status = "❌"
                            example = "Non détecté"

                        mapping_display.append({
                            'Colonne attendue': target_col,
                            'Colonne détectée': source_col or "NON DÉTECTÉE",
                            'Statut': status,
                            'Exemple': example
                        })

                    mapping_df = pd.DataFrame(mapping_display)
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

                    # Vérifier les colonnes obligatoires
                    required_missing = []
                    if not column_mapping.get('reference'):
                        required_missing.append('reference')
                    if not column_mapping.get('item_name'):
                        required_missing.append('item_name')

                    if required_missing:
                        st.error(f"⚠️ Colonnes obligatoires non détectées : {', '.join(required_missing)}")
                        st.info("""
                        **Solutions :**
                        1. **Renommez vos colonnes** dans Excel avant d'exporter
                        2. **Utilisez le mode 'Coller une liste'** pour coller directement vos données
                        3. **Modifiez manuellement** les noms de colonnes ci-dessous
                        """)

                        # Interface de mapping manuel
                        st.markdown("### 🛠️ Mapping manuel des colonnes")

                        col_map1, col_map2 = st.columns(2)

                        with col_map1:
                            manual_ref = st.selectbox(
                                "Colonne pour **Référence**",
                                options=['--- Sélectionnez ---'] + list(df_import.columns),
                                index=0
                            )
                            manual_name = st.selectbox(
                                "Colonne pour **Nom produit**",
                                options=['--- Sélectionnez ---'] + list(df_import.columns),
                                index=0
                            )

                        with col_map2:
                            manual_type = st.selectbox(
                                "Colonne pour **Type** (optionnel)",
                                options=['--- Non utilisé ---'] + list(df_import.columns),
                                index=0
                            )
                            manual_cat = st.selectbox(
                                "Colonne pour **Catégorie** (optionnel)",
                                options=['--- Non utilisé ---'] + list(df_import.columns),
                                index=0
                            )

                        # Mettre à jour le mapping manuel
                        if manual_ref != '--- Sélectionnez ---':
                            column_mapping['reference'] = manual_ref
                        if manual_name != '--- Sélectionnez ---':
                            column_mapping['item_name'] = manual_name
                        if manual_type != '--- Non utilisé ---':
                            column_mapping['type_name'] = manual_type
                        if manual_cat != '--- Non utilisé ---':
                            column_mapping['category_name'] = manual_cat

                    # Vérifier si on peut importer
                    can_import = column_mapping.get('reference') and column_mapping.get('item_name')

                    if can_import:
                        # Bouton d'import
                        if st.button("📥 Importer les données", type="primary", use_container_width=True):
                            # Préparer les données
                            imported_items = []

                            for idx, row in df_import.iterrows():
                                item = {
                                    'id': f"import_{idx}_{int(time.time())}",
                                    'reference': str(row[column_mapping['reference']]).strip() if column_mapping['reference'] in row.index and pd.notna(row[column_mapping['reference']]) else '',
                                    'item_name': str(row[column_mapping['item_name']]).strip() if column_mapping['item_name'] in row.index and pd.notna(row[column_mapping['item_name']]) else '',
                                    'french_name': str(row[column_mapping.get('french_name')]).strip() if column_mapping.get('french_name') and column_mapping['french_name'] in row.index and pd.notna(row.get(column_mapping['french_name'])) else '',
                                    'type_name': str(row[column_mapping.get('type_name')]).strip() if column_mapping.get('type_name') and column_mapping['type_name'] in row.index and pd.notna(row.get(column_mapping['type_name'])) else '',
                                    'category_name': str(row[column_mapping.get('category_name')]).strip() if column_mapping.get('category_name') and column_mapping['category_name'] in row.index and pd.notna(row.get(column_mapping['category_name'])) else '',
                                    'company_name': str(row[column_mapping.get('company_name')]).strip() if column_mapping.get('company_name') and column_mapping['company_name'] in row.index and pd.notna(row.get(column_mapping['company_name'])) else '',
                                    'status': 'non_verifie'
                                }

                                # Vérifier que l'item a au moins une référence ou un nom
                                if item['reference'] or item['item_name']:
                                    imported_items.append(item)

                            # Ajouter à la liste
                            st.session_state.items_a_verifier.extend(imported_items)

                            # Statistiques
                            valid_items = sum(1 for item in imported_items if item['reference'] and item['item_name'])

                            st.success(f"""
                            ✅ Import réussi !
                            - {len(imported_items)} items importés
                            - {valid_items} items complets (référence + nom)
                            """)

                            # Aperçu
                            with st.expander("👁️ Aperçu des 5 premiers items importés"):
                                preview_items = imported_items[:5]
                                preview_df = pd.DataFrame(preview_items)
                                st.dataframe(preview_df[['reference', 'item_name', 'type_name', 'category_name']],
                                           use_container_width=True)

                            st.rerun()
                    else:
                        st.warning("⚠️ Impossible d'importer sans les colonnes Référence et Nom")

                        # Télécharger un template
                        with st.expander("📋 Télécharger un template CSV"):
                            template_data = {
                                'reference': ['REF-001', 'REF-002', 'REF-003'],
                                'item_name': ['Vis à tête plate 10mm', 'Tube PVC 20mm', 'Interrupteur simple'],
                                'type_name': ['Vis', 'Tuyauterie', 'Électricité'],
                                'category_name': ['Quincaillerie', 'Plomberie', 'Électricité'],
                                'company_name': ['Fournisseur A', 'Fournisseur B', 'Fournisseur C']
                            }
                            template_df = pd.DataFrame(template_data)
                            csv_template = template_df.to_csv(index=False, encoding='utf-8-sig')

                            st.download_button(
                                "📥 Télécharger template",
                                data=csv_template,
                                file_name="template_items.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")

    # Tab 3 - Statistiques & Export
    with tab3:
        st.header("📊 Statistiques détaillées")

        if 'df' in st.session_state:
            # Calcul des statistiques
            total_items = len(df)
            unique_names = df['item_name'].nunique()

            if 'reference' in df.columns:
                ref_counts = df['reference'].value_counts()
                duplicate_refs = ref_counts[ref_counts > 1]
                duplicate_ref_count = len(duplicate_refs)
                items_with_duplicate_refs = duplicate_refs.sum() if not duplicate_refs.empty else 0

            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total items", total_items)
            with col2:
                st.metric("Noms uniques", unique_names)
            with col3:
                if 'reference' in df.columns:
                    st.metric("Références en doublon", duplicate_ref_count)
            with col4:
                if 'reference' in df.columns:
                    duplicate_percentage = (items_with_duplicate_refs / total_items * 100) if total_items > 0 else 0
                    st.metric("Items avec ref en doublon", f"{duplicate_percentage:.1f}%")

            # Visualisations
            st.markdown("---")
            st.markdown("### 📈 Visualisations")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Top 10 des références les plus communes
                if 'reference' in df.columns:
                    top_refs = df['reference'].value_counts().head(10)
                    if not top_refs.empty:
                        st.bar_chart(top_refs)
                        st.caption("Top 10 des références les plus courantes")

            with viz_col2:
                # Distribution par catégorie
                if 'category_name' in df.columns:
                    cat_counts = df['category_name'].value_counts().head(15)
                    if not cat_counts.empty:
                        st.dataframe(cat_counts, use_container_width=True)
                        st.caption("Top 15 des catégories")

            # Export des données
            st.markdown("---")
            st.markdown("### 💾 Export des données")

            export_format = st.radio(
                "Format d'export",
                ["CSV", "Excel", "JSON"],
                horizontal=True
            )

            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                if export_format == "CSV":
                    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Télécharger toutes les données",
                        data=csv_data,
                        file_name="items_complet.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                elif export_format == "Excel":
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Données')
                    st.download_button(
                        "📥 Télécharger Excel",
                        data=buffer.getvalue(),
                        file_name="items_complet.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                elif export_format == "JSON":
                    json_data = df.to_json(orient='records', force_ascii=False)
                    st.download_button(
                        "📥 Télécharger JSON",
                        data=json_data,
                        file_name="items_complet.json",
                        mime="application/json",
                        use_container_width=True
                    )

            with col_exp2:
                # Export des items avec références uniques
                if 'reference' in df.columns:
                    unique_ref_df = df.drop_duplicates(subset=['reference'], keep='first')

                    if export_format == "CSV":
                        csv_unique = unique_ref_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "📥 Items avec références uniques",
                            data=csv_unique,
                            file_name="items_references_uniques.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    elif export_format == "Excel":
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            unique_ref_df.to_excel(writer, index=False, sheet_name='Références uniques')
                        st.download_button(
                            "📥 Items avec références uniques",
                            data=buffer.getvalue(),
                            file_name="items_references_uniques.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

if __name__ == "__main__":
    main()
