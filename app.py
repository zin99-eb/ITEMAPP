
# ================================================================
# Items — Upload CSV → Détection de doublons → Saisie + Batch
# Auteur : Zineb FAKKAR – Janv 2026
# ================================================================

import streamlit as st
import pandas as pd
import re
import unicodedata
from pathlib import Path
from io import BytesIO
from datetime import datetime
from rapidfuzz import process, fuzz

# -------- Config UI --------
st.set_page_config(page_title="Items — Upload → Doublons → Saisie", page_icon="🧠", layout="wide")

# -------- Chemins -------- (optionnel, pour chargement manuel)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXPORT_CSV_PATH = DATA_DIR / "export.csv"

# -------- Utilitaires texte --------
def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or "")
    return "".join(c for c in s if not unicodedata.combining(c))

def clean_text(s: str) -> str:
    s = strip_accents((s or "").lower())
    s = re.sub(r'[^a-z0-9\s\-\./_]', ' ', s)
    s = re.sub(r'[_:/\\\-]+', ' ', s)   # uniformiser séparateurs
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def ref_root(r: str) -> str:
    r = (r or "").lower().replace(' ', '')
    r = re.sub(r'[-_/\.]', '', r)
    return r

# -------- Lecture CSV (upload) --------
EXPECTED_COLS = [
    "id","reference","item_name","french_name","uom_name",
    "type_name","sub_category_name","category_name","company_name",
    "last_price","last_use","created_at"
]

RENAME_MAP = {
    # FR courants / variantes
    "nom": "item_name", "name": "item_name",
    "libelle": "french_name", "libellé": "french_name",
    "unite": "uom_name", "uom": "uom_name",
    "type": "type_name",
    "sous_categorie": "sub_category_name", "sous-categorie": "sub_category_name", "sous catégorie": "sub_category_name",
    "categorie": "category_name", "catégorie": "category_name",
    "societe": "company_name", "société": "company_name",
    "prix": "last_price", "dernier_prix": "last_price",
    "derniere_utilisation": "last_use", "dernière_utilisation": "last_use",
    "cree_le": "created_at", "créé_le": "created_at",
    # EN courants
    "unit": "uom_name", "company": "company_name",
    "category": "category_name", "sub_category": "sub_category_name",
    "created": "created_at",
}

def auto_detect_sep(sample_bytes: bytes) -> str:
    """Détecte ';' ou ',' sur les premières lignes."""
    head = sample_bytes[:4096].decode("utf-8", errors="ignore")
    return ";" if head.count(";") >= head.count(",") else ","

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Renommer colonnes si besoin
    to_rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns and v not in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)

    # Assurer la présence des colonnes (schéma base existante)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""

    # Normaliser en strings
    for c in df.columns:
        df[c] = df[c].astype(str).fillna("").str.strip()

    # search_text robuste (Series)
    text_cols = ["item_name","french_name","reference","uom_name","type_name","sub_category_name","category_name"]
    if len(df) > 0:
        tmp = df[text_cols].astype(str).agg(" ".join, axis=1)
        df["search_text"] = tmp.str.lower()
    else:
        df["search_text"] = ""

    # auxiliaires
    df["_item_name_norm"] = df["item_name"].map(lambda x: clean_text(str(x)))
    df["_ref_root"] = df["reference"].map(lambda x: ref_root(str(x)))

    return df

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Lit le fichier uploadé (UTF-8 ⇢ latin-1 ; et séparateur ; ⇢ ,), normalise et renvoie DataFrame."""
    raw = uploaded_file.getvalue()
    sep = auto_detect_sep(raw)
    try:
        df = pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8", sep=sep)
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(raw), dtype=str, encoding="latin-1", sep=sep)
    return normalize_df(df)

def read_local_export_csv() -> pd.DataFrame:
    """Lecture manuelle de data/export.csv si besoin."""
    if not EXPORT_CSV_PATH.exists():
        st.error("❌ data/export.csv introuvable.")
        return pd.DataFrame(columns=EXPECTED_COLS).assign(search_text="", _item_name_norm="", _ref_root="")
    raw = EXPORT_CSV_PATH.read_bytes()
    sep = auto_detect_sep(raw)
    try:
        df = pd.read_csv(EXPORT_CSV_PATH, dtype=str, encoding="utf-8", sep=sep)
    except UnicodeDecodeError:
        df = pd.read_csv(EXPORT_CSV_PATH, dtype=str, encoding="latin-1", sep=sep)
    return normalize_df(df)

# -------- Détection doublons (saisie 1 item) --------
def find_duplicates_for_entry(df: pd.DataFrame, row: dict, topn=10, threshold=0.82):
    """Retourne candidats doublons triés — 'item_name' identique + fuzzy + boost ref_root."""
    if len(df) == 0:
        return pd.DataFrame(columns=EXPECTED_COLS + ["score","match_rule"])

    # 1) Match exact item_name
    item_norm = clean_text(row.get("item_name",""))
    exact = df[df["_item_name_norm"] == item_norm].copy()
    exact["score"] = 1.0
    exact["match_rule"] = "item_name_identique"

    # 2) Fuzzy
    query = " ".join([
        row.get("item_name",""),
        row.get("french_name",""),
        row.get("reference",""),
        row.get("type_name",""),
        row.get("sub_category_name",""),
        row.get("category_name",""),
        row.get("uom_name","")
    ])
    choices = [clean_text(t) for t in df["search_text"].tolist()]
    q = clean_text(query)
    limit = min(topn * 3, len(choices))
    matches = process.extract(q, choices, scorer=fuzz.token_set_ratio, limit=limit, score_cutoff=int(threshold*100))

    fuzzy_df = pd.DataFrame()
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        idxs = [m[2] for m in matches[:topn]]
        scores = [m[1]/100 for m in matches[:topn]]
        fuzzy_df = df.iloc[idxs].copy()
        fuzzy_df["score"] = scores
        fuzzy_df["match_rule"] = "fuzzy"
        rr_new = ref_root(row.get("reference",""))
        if rr_new:
            same_ref = df.iloc[idxs]["_ref_root"] == rr_new
            fuzzy_df.loc[same_ref.values, "score"] = fuzzy_df.loc[same_ref.values, "score"].clip(upper=1.0)

    # 3) Fusionner + trier
    if len(exact) and len(fuzzy_df):
        out = pd.concat([exact, fuzzy_df], ignore_index=True)
    elif len(exact):
        out = exact
    else:
        out = fuzzy_df

    if len(out):
        out = out.sort_values(["score","match_rule"], ascending=[False, True])
        out = out.drop_duplicates(subset=["id","reference","item_name"], keep="first")
    return out

# -------- Détection doublons (batch) --------
def normalize_new_items_df(df_new: pd.DataFrame) -> pd.DataFrame:
    """Normalise un CSV de nouveaux items à vérifier (colonnes minimales)."""
    # Renommer colonnes si besoin (mêmes règles)
    to_rename = {k: v for k, v in RENAME_MAP.items() if k in df_new.columns and v not in df_new.columns}
    if to_rename:
        df_new = df_new.rename(columns=to_rename)

    # Colonnes minimales pour la vérif (on ne force pas les champs 'id', 'last_use', etc.)
    required = ["item_name","french_name","reference","uom_name","type_name","sub_category_name","category_name","company_name"]
    for c in required:
        if c not in df_new.columns:
            df_new[c] = ""

    # Normaliser en strings
    for c in df_new.columns:
        df_new[c] = df_new[c].astype(str).fillna("").str.strip()

    # auxiliaires
    df_new["_item_name_norm"] = df_new["item_name"].map(lambda x: clean_text(str(x)))
    df_new["_ref_root"]       = df_new["reference"].map(lambda x: ref_root(str(x)))
    return df_new

def check_batch_duplicates(df_base: pd.DataFrame, df_new: pd.DataFrame, topn=5, threshold=0.82):
    """Vérifie une liste d'items vs base. Retourne (summary_df, details_df)."""
    rows = []
    det_rows = []
    for idx, r in df_new.iterrows():
        row_dict = {
            "item_name": r.get("item_name",""),
            "french_name": r.get("french_name",""),
            "reference": r.get("reference",""),
            "uom_name": r.get("uom_name",""),
            "type_name": r.get("type_name",""),
            "sub_category_name": r.get("sub_category_name",""),
            "category_name": r.get("category_name",""),
            "company_name": r.get("company_name",""),
        }
        cand = find_duplicates_for_entry(df_base, row_dict, topn=topn, threshold=threshold)

        exists_exact = False
        rr_new = ref_root(r.get("reference",""))
        exists_ref = False
        best_score = 0.0
        best_ref = ""
        best_name = ""
        best_id = ""

        if len(cand) > 0:
            exists_exact = any(cand["match_rule"] == "item_name_identique")
            exists_ref = bool(rr_new) and any(cand["_ref_root"] == rr_new)
            best_idx = cand["score"].idxmax()
            best_score = float(cand.loc[best_idx, "score"])
            best_ref = cand.loc[best_idx, "reference"]
            best_name = cand.loc[best_idx, "item_name"]
            best_id   = cand.loc[best_idx, "id"] if "id" in cand.columns else ""

            # détail : tag 'batch_row'
            tmp = cand.copy()
            tmp.insert(0, "batch_row", idx)
            det_rows.append(tmp)

        rows.append({
            "batch_row": idx,
            **row_dict,
            "exists_exact_name": bool(exists_exact),
            "exists_same_ref_root": bool(exists_ref),
            "best_match_score": round(best_score, 4),
            "best_match_reference": best_ref,
            "best_match_item_name": best_name,
            "best_match_id": best_id
        })

    summary_df = pd.DataFrame(rows)
    details_df = pd.concat(det_rows, ignore_index=True) if det_rows else pd.DataFrame(columns=["batch_row"])
    return summary_df, details_df

# ========================= UI =========================

st.sidebar.markdown("### 📥 Étape 1 — Téléverser la base existante (CSV)")
uploaded = st.sidebar.file_uploader("Choisir un fichier (.csv)", type=["csv"], help="UTF‑8 ou Latin‑1 ; séparateur ; ou ,")

col_local = st.sidebar.container()
if col_local.button("📂 Charger data/export.csv (optionnel)"):
    try:
        df = read_local_export_csv()
        st.session_state["df"] = df
        st.sidebar.success(f"Chargé : {str(EXPORT_CSV_PATH)} • lignes: {len(df)}")
    except Exception as e:
        st.sidebar.error(f"Erreur de lecture export.csv : {e}")

if uploaded:
    try:
        df = read_uploaded_csv(uploaded)
        st.session_state["df"] = df
        st.sidebar.success(f"Upload OK • lignes: {len(df)}")
    except Exception as e:
        st.sidebar.error(f"Lecture CSV échouée : {e}")

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Vider caches"):
    st.cache_data.clear()
    st.session_state.pop("df", None)
    st.experimental_rerun()

st.title("🧠 Items — Upload → Doublons → Saisie")

# Si pas de data, on arrête ici
if "df" not in st.session_state or st.session_state["df"] is None or len(st.session_state["df"]) == 0:
    st.info("➡️ Téléverse un CSV dans la barre latérale pour continuer.")
    st.stop()

df_all = st.session_state["df"]
st.caption(f"📂 Données chargées • lignes: {len(df_all)}")

# -------- Filtres (optionnels) --------
with st.expander("🎚️ Filtres (optionnels)"):
    c1, c2, c3 = st.columns(3)
    f_company = c1.selectbox("Société", [""] + sorted([x for x in df_all["company_name"].unique() if x]), index=0)
    f_type    = c2.selectbox("Type", [""] + sorted([x for x in df_all["type_name"].unique() if x]), index=0)
    f_cat     = c3.selectbox("Catégorie", [""] + sorted([x for x in df_all["category_name"].unique() if x]), index=0)

mask = pd.Series([True]*len(df_all))
if f_company: mask &= (df_all["company_name"] == f_company)
if f_type:    mask &= (df_all["type_name"] == f_type)
if f_cat:     mask &= (df_all["category_name"] == f_cat)
df = df_all[mask].reset_index(drop=True)

tab1, tab2 = st.tabs(["🧹 Détection de doublons (global)", "📝 Saisie & doublons (avant enregistrement)"])

# ====== Tab 1: Scan global ======
with tab1:
    st.subheader("🧹 Scanner les doublons sur toute la base (filtrée)")
    st.caption("Astuce : utilisez des colonnes de blocage pour limiter les comparaisons.")
    options_blocks = [c for c in ["item_name","company_name","type_name","sub_category_name","category_name","uom_name"] if c in df.columns]
    default_blocks = [c for c in ["item_name","type_name","category_name","uom_name"] if c in options_blocks]
    block_cols = st.multiselect("Colonnes de blocage", options_blocks, default=default_blocks)
    threshold_g = st.slider("Seuil de similarité (global)", 0.60, 0.95, 0.82, 0.01)
    max_block = st.number_input("Taille max d'un bloc", 200, 5000, 2500, step=100)
    group_same_name = st.checkbox("Grouper automatiquement les items au **même nom**", value=True)

    if st.button("🔍 Détecter les doublons (global)"):
        groups_df, members_df = detect_duplicate_groups(df, block_cols, threshold=threshold_g, max_block_size=max_block, same_name_group=group_same_name)
        if len(groups_df) == 0:
            st.success("✅ Aucun groupe de doublons détecté avec ces paramètres.")
        else:
            st.markdown("### 🔎 Groupes détectés")
            st.dataframe(groups_df, use_container_width=True)
            st.markdown("### 👥 Membres des groupes")
            view_cols = [c for c in ["group_id","id","reference","item_name","french_name","uom_name","type_name","sub_category_name","category_name","company_name","last_price","rule"] if c in members_df.columns]
            st.dataframe(members_df[view_cols], use_container_width=True)

            b1 = BytesIO(); groups_df.to_csv(b1, index=False, encoding="utf-8"); b1.seek(0)
            b2 = BytesIO(); members_df[view_cols].to_csv(b2, index=False, encoding="utf-8"); b2.seek(0)
            st.download_button("⬇️ Export Groupes (CSV)", data=b1.getvalue(), file_name="dupes_groups.csv", mime="text/csv")
            st.download_button("⬇️ Export Membres (CSV)", data=b2.getvalue(), file_name="dupes_members.csv", mime="text/csv")

# ====== Tab 2: Saisie & doublons ======
with tab2:
    st.subheader("📝 Saisir un nouvel item et vérifier les doublons")
    c1, c2, c3 = st.columns(3)
    with c1:
        item_name = st.text_input("Libellé (EN)", "")
        french_name = st.text_input("Libellé (FR)", "")
        reference = st.text_input("Référence", "")
        uom_name = st.text_input("UoM (ex: each, m, kg)", "")
    with c2:
        type_name = st.text_input("Type", "")
        sub_category_name = st.text_input("Sous-catégorie", "")
        category_name = st.text_input("Catégorie", "")
    with c3:
        company_name = st.text_input("Société/Filiale", "")
        last_price = st.text_input("Dernier prix", "")
        last_use = st.text_input("Dernière utilisation (YYYY-MM-DD)", "")

    st.divider()
    colA, colB = st.columns([1, 1])
    with colA:
        topn = st.slider("Top candidats doublons (fuzzy)", 3, 30, 8)
    with colB:
        threshold = st.slider("Seuil de similarité (fuzzy)", 0.60, 0.95, 0.82, 0.01)

    if st.button("🔎 Vérifier doublons (item saisi)"):
        new_row = {
            "item_name": item_name, "french_name": french_name, "reference": reference,
            "uom_name": uom_name, "type_name": type_name, "sub_category_name": sub_category_name,
            "category_name": category_name, "company_name": company_name,
            "last_price": last_price, "last_use": last_use, "created_at": datetime.utcnow().strftime("%Y-%m-%d")
        }
        candidates = find_duplicates_for_entry(df, new_row, topn=topn, threshold=threshold)

        if len(candidates) == 0:
            st.success("✅ Aucun doublon évident trouvé.")
        else:
            exact = candidates[candidates["match_rule"] == "item_name_identique"]
            fuzzy = candidates[candidates["match_rule"] == "fuzzy"]
            if len(exact):
                st.warning("⚠️ Doublons **nom identique** détectés")
                view_cols = [c for c in ["item_name","french_name","reference","uom_name","type_name","sub_category_name","category_name","company_name","last_price","score"] if c in exact.columns]
                st.dataframe(exact[view_cols], use_container_width=True)
            if len(fuzzy):
                st.info("🔎 Candidats **fuzzy**")
                view_cols = [c for c in ["item_name","french_name","reference","uom_name","type_name","sub_category_name","category_name","company_name","last_price","score"] if c in fuzzy.columns]
                st.dataframe(fuzzy[view_cols], use_container_width=True)

            b = BytesIO(); candidates.drop(columns=["match_rule"], errors="ignore").to_csv(b, index=False, encoding="utf-8"); b.seek(0)
            st.download_button("⬇️ Exporter les candidats (CSV)", data=b.getvalue(), file_name="candidats_doublons.csv", mime="text/csv")

        st.session_state["pending_item"] = new_row

    # ---------- 📦 Vérification en lot (batch) ----------
    st.subheader("📦 Vérifier une **liste** d'items (batch)")
    st.caption("Téléverse un CSV avec au minimum : item_name, reference (autres colonnes optionnelles). Les colonnes seront normalisées automatiquement.")
    uploaded_batch = st.file_uploader("CSV des nouveaux items à vérifier", type=["csv"], key="batch_uploader")
    use_filtered_base = st.checkbox("Utiliser la **base filtrée** pour la vérification (sinon toute la base chargée)", value=True)

    if uploaded_batch:
        # Lire et normaliser la liste
        raw = uploaded_batch.getvalue()
        sep_b = auto_detect_sep(raw)
        try:
            df_new = pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8", sep=sep_b)
        except UnicodeDecodeError:
            df_new = pd.read_csv(BytesIO(raw), dtype=str, encoding="latin-1", sep=sep_b)
        df_new = normalize_new_items_df(df_new)

        # Choix de la base
        base = df if use_filtered_base else df_all

        # Vérifier
        summary_df, details_df = check_batch_duplicates(base, df_new, topn=topn, threshold=threshold)

        st.markdown("### 📋 Résumé (par item)")
        st.dataframe(summary_df[
            ["item_name","reference","company_name","type_name","category_name",
             "exists_exact_name","exists_same_ref_root","best_match_score",
             "best_match_reference","best_match_item_name","best_match_id"]
        ], use_container_width=True)

        # Exports
        buf_s = BytesIO(); summary_df.to_csv(buf_s, index=False, encoding="utf-8"); buf_s.seek(0)
        st.download_button("⬇️ Export Résumé (CSV)", data=buf_s.getvalue(), file_name="batch_summary.csv", mime="text/csv", key="dl_summary")

        st.markdown("### 🔎 Détails (tous les candidats par item)")
        view_cols = [c for c in ["batch_row","id","reference","item_name","french_name","uom_name",
                                 "type_name","sub_category_name","category_name","company_name","last_price",
                                 "score","match_rule"] if c in details_df.columns]
        st.dataframe(details_df[view_cols], use_container_width=True)

        buf_d = BytesIO(); details_df[view_cols].to_csv(buf_d, index=False, encoding="utf-8"); buf_d.seek(0)
        st.download_button("⬇️ Export Détails (CSV)", data=buf_d.getvalue(), file_name="batch_details.csv", mime="text/csv", key="dl_details")

    # ---------- Export du dataset courant ----------
    st.markdown("### ⬇️ Télécharger le CSV **base chargée** (après filtres)")
    buf = BytesIO()
    df[EXPECTED_COLS].to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    st.download_button("Télécharger le CSV filtré", data=buf.getvalue(), file_name="items_filtered.csv", mime="text/csv")
