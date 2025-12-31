# ================================================================
# Assistant IA Items – Version Optimisée pour Streamlit Cloud
# Auteur : Zineb FAKKAR – Janv 2026
# ================================================================

# Core imports (gardés au minimum)
import streamlit as st
import pandas as pd
import re
import unicodedata
from pathlib import Path
from io import BytesIO
import logging
import traceback
from contextlib import contextmanager
from datetime import datetime

# ============= Configuration early ==================================
st.set_page_config(
    page_title="Assistant IA de sélection d'item",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CSS personnalisé =====================================
st.markdown("""
<style>
:root{
  --bg:#f8fafc; --card:#ffffff; --border:#e6eaf0; --txt:#1f2937; --muted:#6b7280;
  --brand:#3E5FFF; --brand-2:#EEF6FF; --ok:#16a34a; --warn:#a16207;
}
* { font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", sans-serif; }
footer {visibility: hidden;}

.big {font-size:28px;font-weight:800;color:var(--txt);margin:8px 0 2px;}
.sub {color:var(--muted);margin-bottom:12px;}
.card {background:var(--card);border:1px solid var(--border);padding:12px;border-radius:12px;}
.badge-green {background:#E9FBF0;color:#16794D;padding:4px 8px;border:1px solid #BDEECD;border-radius:6px;}
.badge-yellow {background:#FFF8E0;color:#7A5E00;padding:4px 8px;border:1px solid #FFE9B3;border-radius:6px;}
.pill {display:inline-block;background:var(--brand-2);color:var(--brand);border:1px solid #DDE3FF;padding:3px 8px;border-radius:999px;margin-left:8px;}
.section-title {font-weight:700;color:var(--txt);margin:16px 0 6px;}
.stTextInput>div>div>input, .stTextArea textarea, .stSelectbox div, .stSlider { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ============= Anti-crash Logging ==================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

@contextmanager
def show_errors(section):
    try:
        yield
    except Exception as e:
        st.error(f"❌ Erreur dans : {section}")
        st.exception(e)
        logger.error("Exception in %s:\n%s", section, "".join(
            traceback.format_exception(type(e), e, e.__traceback__)
        ))

# ============= Optimisation: Définition précoce des caches ===========
@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def load_csv_cached(file_or_path):
    """Cache fort pour éviter les rechargements"""
    return pd.read_csv(file_or_path, sep=";", dtype=str, encoding="utf-8")

@st.cache_resource(show_spinner=False)
def get_fuzzy_processor():
    """Charge rapide et mise en cache de rapidfuzz"""
    try:
        from rapidfuzz import process, fuzz
        return process, fuzz, True
    except Exception:
        return None, None, False

# ============= Fonctions de base ====================================
def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    return "".join(c for c in s if not unicodedata.combining(c))

def clean_text(s: str) -> str:
    s = strip_accents((s or "").lower())
    s = re.sub(r'[^a-z0-9\s\-\./_]', ' ', s)
    s = re.sub(r'[_:/\\\-]+', ' ', s)  # unifie séparateurs
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def soundex(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r'[^a-z]', '', s)
    if not s: return ""
    first = s[0].upper()
    mapping = {'bfpv': '1', 'cgjkqsxz': '2', 'dt': '3', 'l': '4', 'mn': '5', 'r': '6'}
    def code(ch):
        for k,v in mapping.items():
            if ch in k: return v
        return ''
    digits = []
    prev = ''
    for ch in s[1:]:
        d = code(ch)
        if d and d != prev:
            digits.append(d)
        prev = d
    return (first + "".join(digits) + "0000")[:4]

def token_set(s: str) -> set:
    return set([t for t in clean_text(s).split() if t])

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def ref_root(r: str) -> str:
    r = (r or "").lower().replace(' ', '')
    r = re.sub(r'[-_/\.]', '', r)
    return r

# ============= Fonctions optimisées ==================================
def normalize_columns_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée pour plus de rapidité"""
    # Renommage efficace
    rename_dict = {'name': 'item_name', 'nom': 'item_name', 'libelle': 'item_name'}
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns and v not in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Colonnes standard
    standard_cols = [
        'id','reference','item_name','french_name','uom_name',
        'type_name','sub_category_name','category_name','company_name',
        'last_price','last_use','created_at'
    ]
    
    # Garder uniquement les colonnes existantes
    existing_cols = [c for c in standard_cols if c in df.columns]
    df = df[existing_cols] if existing_cols else df
    
    # Vectoriser les opérations de nettoyage
    for col in df.columns:
        df[col] = df[col].astype(str).fillna("").str.strip()
    
    # Créer le champ de recherche (vectorisé)
    text_cols = ["item_name", "french_name", "reference", "uom_name", 
                 "type_name", "sub_category_name", "category_name"]
    text_cols = [c for c in text_cols if c in df.columns]
    
    if text_cols:
        df["search_text"] = df[text_cols].apply(
            lambda row: " ".join(row.values), axis=1
        ).str.lower()
    else:
        df["search_text"] = ""
    
    return df

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    """Wrapper avec cache optimisé"""
    df = load_csv_cached(file_or_path)
    return normalize_columns_optimized(df)

# ============= Search Engine Optimisé ================================
class SearchEngine:
    """Moteur de recherche optimisé avec cache"""
    def __init__(self, df):
        self.df = df
        self._cache = {}
        
    @st.cache_data(ttl=300, show_spinner=False)
    def preprocess_search_texts(_self, texts):
        """Pré-calcul des textes nettoyés"""
        return [clean_text(t) for t in texts]
    
    def fuzzy_search_optimized(self, query: str, topn=10) -> pd.DataFrame:
        """Recherche optimisée avec cache multi-niveaux"""
        cache_key = f"{query}_{topn}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if len(self.df) == 0:
            return pd.DataFrame()
        
        process, fuzz, FUZZY_OK = get_fuzzy_processor()
        if not FUZZY_OK:
            return pd.DataFrame()
        
        q = expand_query(query)
        if not q:
            return pd.DataFrame()
        
        # Pré-calcul des choix
        choices = self.df["search_text"].tolist()
        cleaned_choices = self.preprocess_search_texts(choices)
        
        # Recherche avec limite intelligente
        limit = min(topn * 3, len(choices))  # Recherche large puis filtrage
        
        matches = process.extract(
            q, 
            cleaned_choices, 
            scorer=fuzz.token_set_ratio, 
            limit=limit,
            score_cutoff=50  # Élimine les très mauvais matches
        )
        
        if not matches:
            return pd.DataFrame()
        
        # Trier et limiter
        matches.sort(key=lambda x: x[1], reverse=True)
        idxs = [m[2] for m in matches[:topn]]
        scores = [m[1]/100 for m in matches[:topn]]
        
        res = self.df.iloc[idxs].copy()
        res["score"] = scores
        
        # Cache le résultat
        self._cache[cache_key] = res
        
        return res.sort_values("score", ascending=False)

# ============= Initialisation session state =========================
def init_session_state():
    """Initialise toutes les variables de session"""
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None
    if 'df_current' not in st.session_state:
        st.session_state.df_current = pd.DataFrame()
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = pd.DataFrame()
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'last_file_hash' not in st.session_state:
        st.session_state.last_file_hash = None

# ============= Synonymes ============================================
SYNONYMS = {
    "otdr": ["optical time domain reflectometer","reflectometre optique","réflectomètre optique"],
    "splice": ["fusion splicing","épissure","épissurage"],
    "cleaner": ["nettoyant","kit nettoyage","cleaning kit"],
    "power cable": ["câble électrique","cable d'alimentation","cable power"],
    "cable": ["câble","cordon"],
    "1.5mm": ["1.5 mm","1.5mm²","1.5 mm2"],
    "2.5mm": ["2.5 mm","2.5mm²","2.5 mm2"],
    "kit": ["set","ensemble"],
}

def expand_query(q: str) -> str:
    ql = clean_text(q)
    extra=[]
    for k, vals in SYNONYMS.items():
        if k in ql or any(v in ql for v in vals):
            extra.extend([k]+vals)
    # Note: f_company, f_type, f_cat sont définis plus tard
    return (ql + " " + " ".join(sorted(set(extra)))).strip()

# ============= DSU pour détection doublons =========================
class DSU:
    def __init__(self, n): 
        self.p = list(range(n))
        self.r = [0]*n
    
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: 
            return
        if self.r[ra] < self.r[rb]: 
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: 
            self.p[rb] = ra
        else: 
            self.p[rb] = ra
            self.r[ra] += 1

def build_dupe_text(row: pd.Series) -> str:
    parts = [
        row.get('item_name',''), row.get('french_name',''), row.get('reference',''),
        row.get('uom_name',''), row.get('type_name',''), row.get('sub_category_name',''),
        row.get('category_name','')
    ]
    return clean_text(" ".join([p for p in parts if p]))

# ============= Détection incohérences ==============================
CALIBER_RE = r'(\d+(?:\.\d+)?)\s*mm(?:2|²)?'
LENGTH_RE  = r'(\d+(?:\.\d+)?)\s*(m|meter|metre|mètre|meters)\b'
COLOR_WORDS = ["noir","black","rouge","red","bleu","blue","vert","green","jaune","yellow","blanc","white","gris","gray","grey"]

def grab_float(pattern, text):
    m = re.search(pattern, clean_text(text))
    if not m: 
        return None
    try: 
        return float(m.group(1))
    except: 
        return None

def detect_issues(user_desc: str, row: pd.Series):
    issues = []
    base = (row.get('item_name','') + ' ' + row.get('french_name',''))
    d_cal, i_cal = grab_float(CALIBER_RE, user_desc), grab_float(CALIBER_RE, base)
    if d_cal is not None and i_cal is not None and abs(d_cal - i_cal) > 0.01:
        issues.append(f"Calibre détecté {d_cal} mm² ≠ {i_cal} mm².")
    d_len, i_len = grab_float(LENGTH_RE, user_desc), grab_float(LENGTH_RE, base)
    if d_len is not None and i_len is not None and abs(d_len - i_len) > 0.01:
        issues.append(f"Longueur détectée {d_len} m ≠ {i_len} m.")
    t_user = clean_text(user_desc)
    t_base = clean_text(base)
    for c in COLOR_WORDS:
        if c in t_user and c not in t_base:
            issues.append(f"Couleur '{c}' mentionnée dans la description mais pas dans l'item.")
    uom = (row.get('uom_name','') or '').lower()
    if uom and uom not in t_user:
        issues.append(f"UoM attendue '{row.get('uom_name','')}' non mentionnée.")
    return issues

# ============= Initialisation ======================================
init_session_state()

# ============= Interface principale ================================
st.markdown("<div class='big'>🧠 Assistant IA – Version Optimisée</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Entrez une description, l'outil propose l'item le plus pertinent.</div>", unsafe_allow_html=True)

# ================================================================
#       Chargement fichier CSV Optimisé
# ================================================================
st.sidebar.markdown("### 📥 Importer un fichier CSV")
uploaded = st.sidebar.file_uploader(
    "Fichier (.csv ;)", 
    type=["csv"],
    help="Format CSV avec séparateur point-virgule"
)

with st.spinner("🔍 Préparation de l'application..."):
    if uploaded:
        # Vérifier si le fichier a changé
        file_hash = hash(uploaded.getvalue())
        if (st.session_state.df_loaded is None or 
            st.session_state.get('last_file_hash') != file_hash):
            
            with st.spinner("📊 Chargement et traitement des données..."):
                df = load_csv(uploaded)
                st.session_state.df_loaded = df
                st.session_state.last_file_hash = file_hash
                st.session_state.search_engine = SearchEngine(df)
                st.success(f"✅ {len(df)} items chargés")
        else:
            df = st.session_state.df_loaded
    
    elif Path("data/export.csv").exists():
        if st.session_state.df_loaded is None:
            with st.spinner("📊 Chargement depuis data/export.csv..."):
                df = load_csv("data/export.csv")
                st.session_state.df_loaded = df
                st.session_state.search_engine = SearchEngine(df)
                st.success(f"✅ {len(df)} items chargés")
        else:
            df = st.session_state.df_loaded
    else:
        df = pd.DataFrame(columns=[
            "id","reference","item_name","french_name","uom_name","type_name",
            "sub_category_name","category_name","company_name","last_price","last_use"
        ])
        st.info("📁 Aucun CSV trouvé. Chargez un fichier via la barre latérale")

# ================================================================
#       Filtres métier
# ================================================================
def uniq(df, col):
    if col not in df.columns: 
        return [""]
    s = df[col].astype(str).fillna("").str.strip()
    vals = sorted([x for x in s.unique() if x])
    return [""] + vals

# Variables globales pour les filtres
f_company = ""
f_type = ""
f_cat = ""

if st.session_state.df_loaded is not None:
    with st.expander("🎚️ Filtres (optionnels)"):
        c1, c2, c3 = st.columns(3)
        f_company = c1.selectbox("Société", uniq(st.session_state.df_loaded,"company_name"), help="Filtrer par filiale/société", key="filter_company")
        f_type = c2.selectbox("Type", uniq(st.session_state.df_loaded,"type_name"), help="Filtrer par type d'item", key="filter_type")
        f_cat = c3.selectbox("Catégorie", uniq(st.session_state.df_loaded,"category_name"), help="Filtrer par catégorie", key="filter_cat")

    def apply_filters(df):
        m = pd.Series([True]*len(df))
        if f_company: 
            m &= (df.get("company_name","") == f_company)
        if f_type:    
            m &= (df.get("type_name","") == f_type)
        if f_cat:     
            m &= (df.get("category_name","") == f_cat)
        return df[m].reset_index(drop=True)

    df_current = apply_filters(st.session_state.df_loaded)
    st.session_state.df_current = df_current
else:
    df_current = pd.DataFrame()
    st.session_state.df_current = df_current

# ================================================================
#       UI principale optimisée
# ================================================================
left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown("<div class='section-title'>✍️ Description</div>", unsafe_allow_html=True)
    
    query = st.text_area(
        "Saisissez ce que vous cherchez :", 
        placeholder="Ex : câble fibre 1.5mm 100m",
        key="search_query"
    )
    
    topn = st.slider("Nombre de suggestions", 3, 30, 8, key="top_n_slider")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        search_clicked = st.button(
            "🔎 Rechercher", 
            type="primary",
            use_container_width=True,
            key="search_button"
        )
    
    if search_clicked and query.strip():
        with st.spinner("🔍 Recherche en cours..."):
            if st.session_state.search_engine:
                suggestions = st.session_state.search_engine.fuzzy_search_optimized(query, topn)
                st.session_state.suggestions = suggestions
            else:
                st.error("⚠️ Moteur de recherche non initialisé")
    
    suggestions = st.session_state.suggestions
    if len(suggestions) > 0:
        st.subheader("🎯 Suggestions")
        view_cols = ['item_name','french_name','reference','uom_name',
                     'type_name','sub_category_name','category_name','last_price','score']
        view_cols = [c for c in view_cols if c in suggestions.columns]
        st.dataframe(suggestions[view_cols], hide_index=True, use_container_width=True)
        
        b = BytesIO()
        suggestions[view_cols].to_csv(b, index=False, encoding="utf-8")
        b.seek(0)
        st.download_button(
            "⬇️ Télécharger les suggestions (CSV)", 
            data=b.getvalue(), 
            file_name="suggestions.csv", 
            mime="text/csv",
            key="download_suggestions"
        )

with right:
    st.markdown("<div class='section-title'>📝 Validation de l'item</div>", unsafe_allow_html=True)
    if len(suggestions) > 0:
        opts = [f"{r.item_name} | {r.reference}" for _, r in suggestions.iterrows()]
        choice = st.selectbox("Choisissez un item", opts, index=0, key="item_choice")
        row = suggestions.iloc[opts.index(choice)]
        
        # Description standard simple
        std = f"{row.get('item_name','')} / {row.get('french_name','')}".strip(" / ")
        st.markdown("**📋 Description standard**")
        st.markdown(f"<div class='card'>{std}</div>", unsafe_allow_html=True)
        
        st.markdown("**🔖 Attributs clés**")
        st.code(
            f"Ref: {row.get('reference','')} | UoM: {row.get('uom_name','')}\n"
            f"Type: {row.get('type_name','')} | Sous-cat: {row.get('sub_category_name','')} | "
            f"Cat: {row.get('category_name','')}"
        )
        
        # Incohérences détectées
        issues = detect_issues(query, row)
        if issues:
            st.markdown("<span class='badge-yellow'>⚠️ Incohérences possibles</span>", unsafe_allow_html=True)
            for i in issues:
                st.write("- " + i)
        else:
            st.markdown("<span class='badge-green'>✅ Description cohérente</span>", unsafe_allow_html=True)
        
        st.text_input("Copier cette description :", value=std, key="copy_desc")

# ================================================================
#       ANALYTICS & PERFORMANCE
# ================================================================
def show_analytics(df):
    """Affiche des statistiques sur les données"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analytics")
    
    if len(df) > 0:
        # Stats rapides
        total_items = len(df)
        unique_companies = df['company_name'].nunique() if 'company_name' in df.columns else 0
        unique_categories = df['category_name'].nunique() if 'category_name' in df.columns else 0
        
        st.sidebar.metric("Items totaux", total_items)
        st.sidebar.metric("Sociétés", unique_companies)
        st.sidebar.metric("Catégories", unique_categories)
        
        # Top catégories
        if 'category_name' in df.columns:
            top_cats = df['category_name'].value_counts().head(5)
            st.sidebar.markdown("**Top catégories:**")
            for cat, count in top_cats.items():
                st.sidebar.text(f"• {cat}: {count}")

# Appeler analytics
if st.session_state.df_loaded is not None:
    show_analytics(st.session_state.df_loaded)

# ================================================================
#       Détection des incohérences – sur tout le dataset
# ================================================================
st.markdown("---")
st.markdown("## ✅ Incohérences dans la base")
st.caption("Repère les UoM incohérentes (m vs Each) et des références ou familles suspectes.")

def detect_quality_issues(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in frame.iterrows():
        txt = " ".join([r.get('item_name',''), r.get('french_name','')])
        uom = r.get('uom_name','')
        len_m = grab_float(LENGTH_RE, txt)
        if len_m is not None and clean_text(uom) in ["each",""]:
            rows.append({"id": r.get('id', i), "issue": "UoM incohérente (texte contient une longueur m)","suggestion":"UoM='m'"})
        cal = grab_float(CALIBER_RE, txt)
        type_n = clean_text(r.get('type_name','')); cat_n = clean_text(r.get('category_name',''))
        if cal is not None and ("electrical" not in type_n and "cable" not in clean_text(txt)):
            rows.append({"id": r.get('id', i), "issue": "Calibre mm² détecté mais type/catégorie non électrique"})
        root = ref_root(r.get('reference',''))
        if root and len(root) <= 4:
            rows.append({"id": r.get('id', i), "issue": "Référence trop courte/générique"})
    return pd.DataFrame(rows)

if st.button("🔍 Scanner les incohérences", key="scan_issues"):
    with show_errors("Scan incohérences"):
        if len(df_current) == 0:
            st.warning("Aucune donnée à analyser")
        else:
            issues_df = detect_quality_issues(df_current)
            if len(issues_df) == 0:
                st.success("✅ Aucune incohérence détectée selon ces règles.")
            else:
                st.dataframe(issues_df, use_container_width=True)
                b = BytesIO()
                issues_df.to_csv(b, index=False, encoding="utf-8")
                b.seek(0)
                st.download_button("⬇️ Exporter (CSV)", data=b.getvalue(), file_name="quality_issues.csv", mime="text/csv", key="export_issues")

# ================================================================
#       Détection de doublons – full fuzzy
# ================================================================
def detect_duplicate_groups_optimized(frame: pd.DataFrame,
                                     block_cols: list[str],
                                     similarity_threshold: float = 0.82,
                                     max_block_size: int = 2500):
    """Version optimisée avec pandas vectorisé"""
    if len(frame) == 0:
        return pd.DataFrame(), {}
    
    process, fuzz, FUZZY_OK = get_fuzzy_processor()
    if not FUZZY_OK:
        return pd.DataFrame(), {}
    
    work = frame.copy().reset_index(drop=True)
    work['_rowid'] = range(len(work))
    
    # Pré-calcul des textes pour comparaison
    work['dupe_text'] = work.apply(build_dupe_text, axis=1)
    
    # Blocage intelligent
    available = [c for c in block_cols if c in work.columns]
    if not available:
        work['_block'] = 'ALL'
        available = ['_block']
    
    groups_map = {}
    group_records = []
    next_gid = 1
    
    # Utiliser pandas groupby optimisé
    for _, block in work.groupby(available, dropna=False):
        block = block.reset_index(drop=True)
        if len(block) <= 1:
            continue
        
        # Détection par lots pour les grands groupes
        if len(block) > max_block_size:
            # Hachage pour subdivision
            block['_hash'] = block['dupe_text'].apply(lambda s: hash(s) % 10)
            subgroups = [g for _, g in block.groupby('_hash')]
        else:
            subgroups = [block]
        
        for sub in subgroups:
            if len(sub) <= 1:
                continue
            
            # Matrice de similarité optimisée
            n = len(sub)
            texts = sub['dupe_text'].tolist() 
            
            # Utiliser DSU optimisé
            dsu = DSU(n)
            
            # Comparaisons par paires avec seuil précoce
            for i in range(n):
                for j in range(i + 1, n):
                    # Vérification rapide des références d'abord
                    ri = ref_root(sub.iloc[i]['reference'])
                    rj = ref_root(sub.iloc[j]['reference'])
                    
                    if ri and rj and ri == rj:
                        dsu.union(i, j)
                        continue
                    
                    # Calcul fuzzy seulement si nécessaire
                    s = fuzz.token_set_ratio(texts[i], texts[j]) / 100.0
                    if s >= similarity_threshold:
                        dsu.union(i, j)
            
            # Compter les composants
            # Compter les composants
            comp = {}
            for i in range(n):
                r = dsu.find(i)
                comp.setdefault(r, []).append(i)
            
            # Créer les groupes
            for members in comp.values():
                if len(members) <= 1:
                    continue
                
                group_df = sub.iloc[members].copy()
                
                # Trouver le représentant (celui avec la référence la plus complète)
                group_df['ref_len'] = group_df['reference'].fillna('').str.len()
                rep = group_df.loc[group_df['ref_len'].idxmax()]
                
                groups_map[next_gid] = group_df['_rowid'].tolist()
                group_records.append({
                    "group_id": next_gid,
                    "size": len(group_df),
                    "representative_reference": rep.get('reference', ''),
                    "representative_name": rep.get('item_name', ''),
                    "representative_id": rep.get('id', '')
                })
                next_gid += 1
    
    groups_df = pd.DataFrame(group_records)
    if len(groups_df) > 0:
        groups_df = groups_df.sort_values('size', ascending=False).reset_index(drop=True)
    
    return groups_df, groups_map

st.markdown("---")
st.markdown("## 🧹 Détection de doublons (IA légère)")
st.caption("Regroupe les items très proches. Ajustez le seuil et les colonnes de blocage.")

with st.expander("⚙️ Paramètres doublons"):  # CORRIGÉ : Supprimé key=
    options_blocks = [c for c in ['company_name','type_name','sub_category_name','category_name','uom_name','item_name'] if c in df_current.columns]
    default_blocks = [c for c in ['type_name','category_name','uom_name'] if c in options_blocks]
    block_cols = st.multiselect("Colonnes de blocage (pour limiter les comparaisons)", options_blocks, default=default_blocks)
    threshold = st.slider("Seuil de similarité", 0.60, 0.95, 0.82, 0.01)
    max_block = st.number_input("Taille max par bloc", 500, 5000, 2500, step=100)

if st.button("🔎 Détecter les doublons"):
    with show_errors("Doublons fuzzy"):
        if len(df_current) == 0:
            st.warning("Aucune donnée à analyser")
        else:
            groups_df, groups_map = detect_duplicate_groups_optimized(df_current, block_cols, threshold, max_block)
            if len(groups_df) == 0:
                st.success("✅ Aucun groupe de doublons avec ces paramètres.")
            else:
                st.subheader(f"Groupes détectés : {len(groups_df)}")
                st.dataframe(groups_df, use_container_width=True)
                b1 = BytesIO()
                groups_df.to_csv(b1, index=False, encoding="utf-8")
                b1.seek(0)
                st.download_button("⬇️ Exporter groupes (CSV)", data=b1.getvalue(), file_name="dupes_groups.csv", mime="text/csv")

                # Membres
                st.markdown("### 👥 Membres de tous les groupes")
                cols_view = [c for c in ['id','reference','item_name','french_name','uom_name','type_name','sub_category_name','category_name','last_price'] if c in df_current.columns]
                all_rows = []
                for gid, idxs in groups_map.items():
                    tmp = df_current.iloc[idxs].copy()
                    tmp.insert(0,'group_id',gid)
                    all_rows.append(tmp)
                members_all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=['group_id']+cols_view)
                st.dataframe(members_all_df[['group_id']+cols_view], use_container_width=True)
                b2 = BytesIO()
                members_all_df[['group_id']+cols_view].to_csv(b2, index=False, encoding="utf-8")
                b2.seek(0)
                st.download_button("⬇️ Exporter membres (CSV)", data=b2.getvalue(), file_name="dupes_members_all.csv", mime="text/csv")

# ================================================================
#       Master Data — Clean & Merge Plan (fuzzy composite)
# ================================================================
def numeric_signature(row: pd.Series) -> dict:
    base = " ".join([
        row.get('item_name',''), row.get('french_name',''), row.get('reference',''),
        row.get('uom_name',''), row.get('type_name',''),
        row.get('sub_category_name',''), row.get('category_name','')
    ])
    return {
        "cal_mm2": grab_float(CALIBER_RE, base),
        "len_m": grab_float(LENGTH_RE, base),
    }

def norm_uom(u: str) -> str:
    u = clean_text(u)
    MAP = {
        "each":"each","ea":"each","piece":"each","pc":"each","pcs":"each",
        "m":"m","meter":"m","metre":"m","mètre":"m","meters":"m",
        "kg":"kg","kilogram":"kg"
    }
    return MAP.get(u, u)

def composite_score_fuzzy(a: pd.Series, b: pd.Series) -> tuple[float, dict]:
    process, fuzz, FUZZY_OK = get_fuzzy_processor()
    if not FUZZY_OK:
        return 0.0, {}
    
    # Fuzzy (search_text)
    s_fuzzy = fuzz.token_set_ratio(
        clean_text(a.get('search_text','')),
        clean_text(b.get('search_text',''))
    ) / 100.0
    
    # Jaccard
    s_jacc = jaccard(
        token_set(a.get('item_name','') + " " + a.get('french_name','')),
        token_set(b.get('item_name','') + " " + b.get('french_name',''))
    )
    
    # Phonétique (premier mot)
    na = clean_text(a.get('item_name',''))
    nb = clean_text(b.get('item_name',''))
    sa = soundex(na.split()[0]) if na else ""
    sb = soundex(nb.split()[0]) if nb else ""
    s_phon = 1.0 if sa and sb and sa == sb else 0.0
    
    # Référence root
    ra = ref_root(a.get('reference',''))
    rb = ref_root(b.get('reference',''))
    s_ref = 1.0 if (ra and rb and ra == rb) else 0.0
    
    # Attributs (type/cat/uom + signatures numériques)
    type_ok = clean_text(a.get('type_name','')) == clean_text(b.get('type_name',''))
    cat_ok = clean_text(a.get('category_name','')) == clean_text(b.get('category_name',''))
    uom_ok = norm_uom(a.get('uom_name','')) == norm_uom(b.get('uom_name',''))
    
    sig_a, sig_b = numeric_signature(a), numeric_signature(b)
    num_ok = True
    if sig_a['cal_mm2'] is not None and sig_b['cal_mm2'] is not None:
        num_ok &= abs(sig_a['cal_mm2'] - sig_b['cal_mm2']) <= 0.01
    if sig_a['len_m'] is not None and sig_b['len_m'] is not None:
        num_ok &= abs(sig_a['len_m'] - sig_b['len_m']) <= 0.01
    
    s_attr = (1.0 if type_ok else 0.0)*0.4 + (1.0 if cat_ok else 0.0)*0.4 + (1.0 if uom_ok else 0.0)*0.2
    if not num_ok: 
        s_attr *= 0.7

    weights = st.session_state.get('md_weights', {
        'fuzzy': 0.40, 'jacc': 0.25, 'phon': 0.15, 'ref': 0.10, 'attr': 0.10
    })
    
    score = (
        weights['fuzzy'] * s_fuzzy + 
        weights['jacc'] * s_jacc + 
        weights['phon'] * s_phon +
        weights['ref'] * s_ref + 
        weights['attr'] * s_attr
    )
    
    details = {
        "s_fuzzy": s_fuzzy,
        "s_jaccard": s_jacc,
        "s_phonetic": s_phon,
        "s_ref": s_ref,
        "s_attr": s_attr
    }
    
    return score, details

def choose_representative(group_df: pd.DataFrame) -> pd.Series:
    g = group_df.copy()
    g['ref_len'] = g['reference'].fillna('').str.len()
    # priorité à la référence renseignée puis au prix non nul
    try:
        g['_price'] = pd.to_numeric(g.get('last_price',''), errors='coerce')
    except:
        g['_price'] = pd.NA
    g['_rep'] = g['ref_len'].fillna(0) + (g['_price'].fillna(0)/1000.0)
    return g.sort_values(['ref_len','_rep'], ascending=[False, False]).iloc[0]

st.markdown("---")
st.markdown("## 🗃️ Master Data — Plan de fusion (proposition)")
st.caption("Score composite : fuzzy + jaccard + phonétique + référence + attributs. Sans ML lourd.")

with st.expander("⚙️ Paramètres Master Data"):  # CORRIGÉ : Supprimé key=
    c1, c2 = st.columns(2)
    with c1:
        threshold_md = st.slider("Seuil de confiance (merge)", 0.60, 0.95, 0.85, 0.01, key="threshold_md")
    with c2:
        st.markdown("**Pondérations du score**")
        fuzzy_w = st.slider("Poids Fuzzy", 0.0, 1.0, 0.40, 0.05, key="fuzzy_w")
        jacc_w = st.slider("Poids Jaccard", 0.0, 1.0, 0.25, 0.05, key="jacc_w")
        phon_w = st.slider("Poids Phonétique", 0.0, 1.0, 0.15, 0.05, key="phon_w")
        ref_w = st.slider("Poids Référence", 0.0, 1.0, 0.10, 0.05, key="ref_w")
        attr_w = st.slider("Poids Attributs", 0.0, 1.0, 0.10, 0.05, key="attr_w")
        total_w = fuzzy_w + jacc_w + phon_w + ref_w + attr_w
        if abs(total_w - 1.0) > 1e-6:
            st.info(f"Somme des poids = {total_w:.2f} (normalisée automatiquement)")
        st.session_state['md_weights'] = {
            'fuzzy': fuzzy_w/total_w, 
            'jacc': jacc_w/total_w, 
            'phon': phon_w/total_w,
            'ref': ref_w/total_w, 
            'attr': attr_w/total_w
        }
    cA, cB = st.columns(2)
    with cA:
        strong_gap = st.slider("Écart pour merge STRONG (vs 2e)", 0.0, 0.5, 0.10, 0.01, key="strong_gap")
    with cB:
        mid_gap = st.slider("Écart pour merge MEDIUM", 0.0, 0.5, 0.05, 0.01, key="mid_gap")

if st.button("🧠 Générer le plan de fusion"):
    with show_errors("Plan Master Data"):
        if len(df_current) < 2:
            st.info("Pas assez d'items pour proposer des fusions.")
        else:
            work = df_current.copy().reset_index(drop=True)

            # groupage simple par racine de référence + famille (BK1/BK3)
            work['BK1'] = work.apply(lambda r: "|".join([
                clean_text(r.get('type_name','')),
                clean_text(r.get('category_name','')),
                norm_uom(r.get('uom_name',''))
            ]) or "BK1", axis=1)
            
            work['BK3'] = work['reference'].apply(ref_root)

            groups_records = []
            members_records = []
            gid = 1
            
            for _, block in work.groupby(['BK1','BK3'], dropna=False):
                block = block.reset_index(drop=True)
                if len(block) < 2: 
                    continue
                
                rep = choose_representative(block)
                rep_id = rep.get('id', str(rep.name))
                rep_ref = rep.get('reference','')
                rep_name = rep.get('item_name','')

                scores = []
                for j, row in block.iterrows():
                    if j == rep.name: 
                        continue
                    s, det = composite_score_fuzzy(rep, row)
                    scores.append((j, s, det))
                scores.sort(key=lambda x: x[1], reverse=True)

                if len(scores) >= 1:
                    best = scores[0][1]
                    second = scores[1][1] if len(scores) > 1 else 0.0
                    gap = best - second
                else:
                    best, second, gap = 0.0, 0.0, 0.0

                groups_records.append({
                    "group_id": gid, 
                    "size": len(block),
                    "representative_id": rep_id, 
                    "representative_reference": rep_ref, 
                    "representative_name": rep_name,
                    "best_score": round(best,4), 
                    "gap_vs_second": round(gap,4),
                    "bk_keys": f"{block['BK1'].iloc[0]} | {block['BK3'].iloc[0]}"
                })

                for j, s, det in scores:
                    row = block.iloc[j]
                    action = "review"
                    reason = []
                    
                    if s >= threshold_md and gap >= strong_gap:
                        action = "merge_strong"
                        reason.append("score>=seuil & gap strong")
                    elif s >= (threshold_md - 0.05) and gap >= mid_gap:
                        action = "merge_medium"
                        reason.append("score proche seuil & gap medium")
                    elif s < (threshold_md - 0.10):
                        action = "skip"
                        reason.append("score faible")
                    else:
                        action = "review"
                        reason.append("incertain")

                    members_records.append({
                        "group_id": gid,
                        "representative_id": rep_id,
                        "representative_reference": rep_ref,
                        "representative_name": rep_name,
                        "item_id": row.get('id', str(row.name)),
                        "item_reference": row.get('reference',''),
                        "item_name": row.get('item_name',''),
                        "uom_name": row.get('uom_name',''),
                        "type_name": row.get('type_name',''),
                        "category_name": row.get('category_name',''),
                        "score": round(s,4),
                        "action": action,
                        "reason": ", ".join(reason),
                        "s_fuzzy": round(det["s_fuzzy"],4),
                        "s_jaccard": round(det["s_jaccard"],4),
                        "s_phonetic": round(det["s_phonetic"],4),
                        "s_ref": round(det["s_ref"],4),
                        "s_attr": round(det["s_attr"],4),
                    })

                gid += 1

            groups_df = pd.DataFrame(groups_records).sort_values('size', ascending=False)
            plan_df = pd.DataFrame(members_records).sort_values(['group_id','score'], ascending=[True, False])

            if len(groups_df) == 0:
                st.success("✅ Aucun candidat à fusion avec ces paramètres.")
            else:
                st.markdown("### 🔎 Groupes (représentants proposés)")
                st.dataframe(groups_df, use_container_width=True)
                st.markdown("### 🧩 Plan de fusion (membres)")
                st.dataframe(plan_df, use_container_width=True)
                
                b1 = BytesIO()
                groups_df.to_csv(b1, index=False, encoding='utf-8')
                b1.seek(0)
                
                b2 = BytesIO()
                plan_df.to_csv(b2, index=False, encoding='utf-8')
                b2.seek(0)
                
                st.download_button("⬇️ Export Groupes (CSV)", data=b1.getvalue(), file_name="md_groups.csv", mime="text/csv")
                st.download_button("⬇️ Export Plan (CSV)", data=b2.getvalue(), file_name="md_to_merge.csv", mime="text/csv")

                st.session_state["md_groups_df"] = groups_df
                st.session_state["md_plan_df"] = plan_df

# ================================================================
#       Obsolescence (items peu/plus utilisés)
# ================================================================
st.markdown("---")
st.markdown("## 🕰️ Obsolescence")
st.caption("Liste les items jamais utilisés récemment (last_use/created_at).")

with st.expander("⚙️ Paramètres obsolescence"):  # CORRIGÉ : Supprimé key=
    months = st.slider("Inactivité (mois)", 3, 36, 12, key="months_inactive")

if st.button("🗑️ Lister les obsolètes"):
    with show_errors("Obsolescence"):
        base = df_current.copy()
        dt_use = pd.to_datetime(base.get('last_use',''), errors='coerce', utc=True)
        dt_created = pd.to_datetime(base.get('created_at',''), errors='coerce', utc=True)
        base['_last_dt'] = dt_use.fillna(dt_created)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=months)
        obso = base[(base['_last_dt'].isna()) | (base['_last_dt'] < cutoff)]
        out = obso.copy()
        
        try:
            out['_last_dt_display'] = out['_last_dt'].dt.tz_localize(None)
        except Exception:
            out['_last_dt_display'] = out['_last_dt']
        
        show_cols = ['id','reference','item_name','last_use','created_at','_last_dt_display']
        cols_present = [c for c in show_cols if c in out.columns]
        
        if len(out) > 0:
            st.dataframe(out[cols_present], use_container_width=True)
            b = BytesIO()
            out.to_csv(b, index=False, encoding='utf-8')
            b.seek(0)
            st.download_button("⬇️ Export obsolètes (CSV)", data=b.getvalue(), file_name="obsolete_items.csv", mime="text/csv")
        else:
            st.success("✅ Aucun item obsolète trouvé.")

# ================================================================
#       Corrections orthographiques & synonymes
# ================================================================
st.markdown("---")
st.markdown("## ✏️ Corrections orthographiques & synonymes")
st.caption("Standardise les libellés (Câble/Cable/calbe…), abréviations et variantes FR/EN.")

ORTHO_REPLACE = {
    "calbe": "cable", "cabl e": "cable", "câbl e": "câble",
    "bat": "battery", "batt": "battery", "psu": "power supply",
    "charg": "charger", "chgr": "charger",
    "mpos": "mpo", "lc/apc": "lc apc", "sc/apc": "sc apc",
}

syn_path = Path("data/synonyms.csv")
if syn_path.exists():
    try:
        syn_df = pd.read_csv(syn_path, dtype=str)
        for _, r in syn_df.iterrows():
            t = str(r.get("term","")).strip().lower()
            rep = str(r.get("replacement","")).strip().lower()
            if t and rep: 
                ORTHO_REPLACE[t] = rep
        st.info(f"Synonymes chargés : {len(syn_df)} (data/synonyms.csv)")
    except Exception as e:
        st.warning(f"Impossible de charger synonyms.csv : {e}")

def suggest_standard_name(name: str) -> str:
    s = clean_text(name or "")
    toks = s.split()
    toks2 = [ORTHO_REPLACE.get(t, t) for t in toks]
    s2 = " ".join(toks2)
    s2 = re.sub(r'\bmm\s*2\b', 'mm²', s2)
    s2 = re.sub(r'\bmm2\b', 'mm²', s2)
    s2 = re.sub(r'\b12v7ah\b', '12v 7ah', s2)
    return s2.strip()

if st.button("🧽 Proposer des libellés standardisés"):
    with show_errors("Standardisation libellés"):
        if len(df_current) == 0:
            st.warning("Aucune donnée à traiter")
        else:
            preview = df_current[['id','item_name','french_name']].copy() if 'id' in df_current.columns else df_current[['item_name','french_name']].copy()
            preview['item_name_std'] = preview['item_name'].apply(suggest_standard_name)
            st.dataframe(preview.head(50), use_container_width=True)
            b = BytesIO()
            preview.to_csv(b, index=False, encoding='utf-8')
            b.seek(0)
            st.download_button("⬇️ Export libellés (CSV)", data=b.getvalue(), file_name="item_names_standard.csv", mime="text/csv")

# ================================================================
#       EXPORT AVANCÉ
# ================================================================
def export_advanced_options():
    """Options d'export avancées"""
    with st.expander("🚀 Export avancé"):  # CORRIGÉ : Supprimé key=
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Format",
                ["CSV", "Excel", "JSON"],
                index=0
            )
        
        with col2:
            encoding = st.selectbox(
                "Encodage",
                ["utf-8", "utf-8-sig", "latin-1"],
                index=0
            )
        
        if st.button("📤 Exporter toutes les données"):
            if len(df_current) > 0:
                buffer = BytesIO()
                
                if export_format == "Excel":
                    df_current.to_excel(buffer, index=False, engine='openpyxl')
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_ext = "xlsx"
                elif export_format == "JSON":
                    buffer.write(df_current.to_json(orient='records', force_ascii=False).encode(encoding))
                    mime_type = "application/json"
                    file_ext = "json"
                else:  # CSV
                    buffer.write(df_current.to_csv(index=False, encoding=encoding).encode(encoding))
                    mime_type = "text/csv"
                    file_ext = "csv"
                
                buffer.seek(0)
                
                st.download_button(
                    label=f"⬇️ Télécharger ({export_format})",
                    data=buffer,
                    file_name=f"export_items.{file_ext}",
                    mime=mime_type
                )

# Export avancé
export_advanced_options()

# ================================================================
#       BATCH PROCESSING
# ================================================================
def batch_processing():
    """Traitement par lots"""
    with st.expander("⚙️ Traitement par lots"):  # CORRIGÉ : Supprimé key=
        st.markdown("**Correction automatique de formats**")
        
        if st.button("🔄 Standardiser toutes les références"):
            if len(df_current) > 0:
                with st.spinner("Standardisation en cours..."):
                    # Exemple: standardisation des références
                    if 'reference' in df_current.columns:
                        df_current['reference'] = df_current['reference'].str.upper().str.strip()
                        st.success("Références standardisées")
        
        if st.button("🧹 Nettoyer les UoM"):
            if len(df_current) > 0 and 'uom_name' in df_current.columns:
                uom_mapping = {
                    'each': 'unité',
                    'unit': 'unité',
                    'pcs': 'unité',
                    'mtr': 'mètre',
                    'mt': 'mètre',
                    'kg': 'kilogramme',
                }
                df_current['uom_name'] = df_current['uom_name'].replace(uom_mapping)
                st.success("UoM nettoyées")

# Batch processing
batch_processing()

# ================================================================
#       Fin
# ================================================================
st.markdown("---")
st.success("Version **Optimisée** chargée ✔️ – Stable pour Streamlit Cloud. Performances améliorées.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 - Assistant IA Items - v2.0")