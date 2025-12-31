
# ================================================================
# Items — Upload CSV → Détection de doublons → Saisie
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

    # Assurer la présence des colonnes
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""

    # Normaliser en strings
    for c in df.columns:
        df[c] = df[c].astype(str).fillna("").str.strip()

    # search_text robuste
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

# -------- Détection doublons (saisie) --------
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

# -------- Scan global des doublons --------
def detect_duplicate_groups(df: pd.DataFrame, block_cols: list, threshold=0.82, max_block_size=2500, same_name_group=True):
    """
    Renvoie groups_df, members_df
    - groupage fuzzy avec blocage
    - option "same_name_group" : regroupe directement tous les items ayant le même item_name normalisé
    """
    if len(df) == 0:
        return (pd.DataFrame(columns=["group_id","size","representative_reference","representative_name","rule"]),
                pd.DataFrame())

    work = df.copy().reset_index(drop=True)

    groups = []
    members = []
    gid = 1

    # A) Groupes "item_name identique"
    if same_name_group and "_item_name_norm" in work.columns:
        for name_norm, g in work.groupby("_item_name_norm", dropna=False):
            if not name_norm or len(g) <= 1:
                continue
            g2 = g.copy()
            g2["ref_len"] = g2["reference"].fillna("").str.len()
            rep = g2.loc[g2["ref_len"].idxmax()]
            groups.append({
                "group_id": gid,
                "size": len(g2),
                "representative_reference": rep.get("reference",""),
                "representative_name": rep.get("item_name",""),
                "rule": "item_name_identique"
            })
            g2.insert(0, "group_id", gid)
            members.append(g2.drop(columns=["ref_len"]))
            gid += 1

    # B) Fuzzy + blocage
    work["dupe_text"] = work.apply(lambda r: clean_text(" ".join([
        r.get('item_name',''), r.get('french_name',''), r.get('reference',''),
        r.get('uom_name',''), r.get('type_name',''), r.get('sub_category_name',''), r.get('category_name','')
    ])), axis=1)

    available = [c for c in block_cols if c in work.columns]
    if not available:
        work["_block"] = "ALL"
        available = ["_block"]

    for _, block in work.groupby(available, dropna=False):
        block = block.reset_index(drop=True)
        if len(block) <= 1:
            continue

        # Sous-bloc si trop grand
        if len(block) > max_block_size:
            block["_hash"] = block["dupe_text"].apply(lambda s: hash(s) % 10)
            subgroups = [g for _, g in block.groupby("_hash")]
        else:
            subgroups = [block]

        for sub in subgroups:
            if len(sub) <= 1:
                continue
            texts = sub["dupe_text"].tolist()
            n = len(sub)
            pair_scores = []
            for i in range(n):
                res_i = process.extract(texts[i], texts, scorer=fuzz.token_set_ratio, limit=min(50, n))
                for candidate in res_i:
                    j = candidate[2]
                    if j <= i:
                        continue
                    s = candidate[1]/100
                    # Boost si même ref root
                    ri = ref_root(sub.iloc[i]["reference"])
                    rj = ref_root(sub.iloc[j]["reference"])
                    if ri and rj and ri == rj:
                        s = max(s, 0.95)
                    if s >= threshold:
                        pair_scores.append((i, j, s))

            if not pair_scores:
                continue

            # DSU simple
            parent = list(range(n)); rank = [0]*n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb: return
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1

            for i, j, s in pair_scores:
                union(i, j)

            comps = {}
            for i in range(n):
                r = find(i)
                comps.setdefault(r, []).append(i)

            for members_idx in comps.values():
                if len(members_idx) <= 1:
                    continue
                g2 = sub.iloc[members_idx].copy()
                g2["ref_len"] = g2["reference"].fillna("").str.len()
                rep = g2.loc[g2["ref_len"].idxmax()]
                groups.append({
                    "group_id": gid,
                    "size": len(g2),
                    "representative_reference": rep.get("reference",""),
                    "representative_name": rep.get("item_name",""),
                    "rule": "fuzzy_blocked"
                })
                g2.insert(0, "group_id", gid)
                members.append(g2.drop(columns=["ref_len"]))
                gid += 1

    groups_df = pd.DataFrame(groups).sort_values(["rule","size"], ascending=[True, False]) if groups else pd.DataFrame(columns=["group_id","size","representative_reference","representative_name","rule"])
    members_df = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return groups_df, members_df

# ========================= UI =========================

st.sidebar.markdown("### 📥 Étape 1 — Téléverser le CSV")
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
if "df" not in st.session_state or len(st.session_state["df"]) == 0:
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

    # Enregistre seulement dans la session (et propose un download)
    if st.session_state.get("pending_item"):
        st.info("Un item est prêt à être ajouté au CSV (fichier téléchargé).")
        colY, colZ = st.columns([1, 1])
        if colY.button("💾 Ajouter l’item au dataset (mémoire)"):
            row = st.session_state["pending_item"].copy()
            # Générer un id simple
            if "id" in df_all.columns and df_all["id"].str.isnumeric().any():
                try:
                    next_id = str(int(df_all["id"].dropna().astype(int).max()) + 1)
                except:
                    next_id = str(len(df_all) + 1)
            else:
                next_id = str(len(df_all) + 1)
            row["id"] = next_id
            # Assurer les colonnes
            for c in EXPECTED_COLS:
                row.setdefault(c, "")
            # Append
            df_updated = pd.concat([df_all[[c for c in df_all.columns if c in EXPECTED_COLS]], pd.DataFrame([row])], ignore_index=True)
            # Re-normaliser (search_text / auxiliaires)
            df_updated = normalize_df(df_updated)
            st.session_state["df"] = df_updated
            st.success(f"✅ Item ajouté (id={next_id}) au dataset en mémoire.")
        if colZ.button("🗑️ Annuler"):
            st.session_state["pending_item"] = None
            st.info("Saisie annulée.")

    # Exporter le dataset courant (après ajout éventuel)
    st.markdown("### ⬇️ Télécharger le CSV mis à jour")
    buf = BytesIO()
    st.session_state["df"][EXPECTED_COLS].to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    st.download_button("Télécharger le CSV", data=buf.getvalue(), file_name="items_updated.csv", mime="text/csv")
