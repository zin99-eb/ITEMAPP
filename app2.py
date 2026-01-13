
# app.py
# -------------------------------------------------------------------
# Streamlit : Comparaison Stock (Total livré) vs Quantité Commandée (PO)
# - Normalisation CSV POs (séparateurs ; , \t |)
# - Sélection de feuille Excel (PO & Stock)
# - Matching SAP Name (Stock) ~ Description (PO)
# - Orphelins + meilleur match (affiche PO description + PO Item Code)
# - Background image + overlay + ALERTES ROUGES + DÉTAILS (journal anomalies)
# -------------------------------------------------------------------

import io
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
import base64
from pathlib import Path

st.set_page_config(page_title="Comparaison Stock vs Commandes", layout="wide")

# =========================
# >>> THEME / BACKGROUND <<<
# =========================
def apply_background(image_path: str, overlay_rgba="rgba(255,255,255,0.85)"):
    """
    Applique une image de fond + un overlay translucide pour lire le contenu.
    Place simplement l'image (PNG/JPG) dans le même dossier que app.py.
    """
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            st.warning(f"Image de background introuvable : {image_path}. Vérifie le nom/fichier.")
            return
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        st.markdown(
            f"""
            <style>
            /* Fond principal de l'app */
            .stApp {{
                background: url("data:image/{img_path.suffix[1:]};base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            /* Overlay lisible sur le contenu principal */
            .stApp .block-container {{
                background: {overlay_rgba};
                border-radius: 14px;
                padding: 1.2rem 1.6rem;
            }}
            /* Sidebar lisible */
            [data-testid="stSidebar"] {{
                background: {overlay_rgba};
            }}
            /* Petits badges style chips */
            .badge {{
                display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-weight:600; font-size:0.85rem;
            }}
            .badge-red  {{ background:#ffe5e5; color:#b00020; border:1px solid #ffb3b3; }}
            .badge-green{{ background:#e7f8ed; color:#0b6b2a; border:1px solid #b3e6c5; }}
            .badge-amber{{ background:#fff5e6; color:#8a4b00; border:1px solid #ffd9a6; }}

            /* Table: rendre la colonne Écart plus visible */
            td[data-column="Écart = Stock - Commandé"] {{
                font-weight: 700;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Impossible d'appliquer le background: {e}")

# 👉 Mets ici le nom exact de ton image
apply_background("image_supply_chain_1.png", overlay_rgba="rgba(255,255,255,0.88)")

# =========================
# Helpers généraux (chargement/normalisation)
# =========================
def normalize_colnames(cols):
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]

def detect_delimiter(sample_text: str):
    candidates = [';', ',', '\t', '|']
    counts = {sep: sample_text.count(sep) for sep in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None

def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """Lit un CSV en essayant encodages + séparateurs; normalise si une seule colonne avec ';'."""
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    # encodages courants
    text = None
    encoding = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    sep = detect_delimiter(text)
    df = None
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, encoding=encoding, engine="python")
    except Exception:
        for s in (None, ';', ',', '\t', '|'):
            try:
                df = pd.read_csv(io.StringIO(text), sep=s, encoding=encoding, engine="python")
                break
            except Exception:
                continue

    # Si une seule colonne et présence de ';' -> re-split
    if df is not None and df.shape[1] == 1:
        col = df.columns[0]
        if (';' in col) or df.iloc[:, 0].astype(str).str.contains(';').any():
            split_df = df.iloc[:, 0].astype(str).str.split(';', expand=True)
            first_row = split_df.iloc[0]
            looks_like_header = any(h.lower() in ["item code", "qty", "created_at", "description"]
                                    for h in first_row.astype(str).str.lower())
            if looks_like_header:
                split_df.columns = [re.sub(r"\s+", " ", str(x)).strip() for x in first_row]
                split_df = split_df.iloc[1:].reset_index(drop=True)
            df = split_df

    if df is None:
        return None

    df.columns = normalize_colnames(df.columns)
    # garder Item Code en texte
    for c in df.columns:
        if "item" in c.lower() and "code" in c.lower():
            df[c] = df[c].astype(str).str.strip()
    return df

def read_excel_with_sheet_selector(uploaded_file, key_prefix: str):
    """Propose la sélection de feuille dans la sidebar et retourne le DF de la feuille choisie."""
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    sheet_names = xls.sheet_names
    st.sidebar.caption(f"📑 Feuilles détectées ({key_prefix})")
    sheet = st.sidebar.selectbox(
        f"Sélectionne la feuille pour {key_prefix}",
        options=sheet_names,
        index=0,
        key=f"{key_prefix}_sheet_select"
    )
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, engine="openpyxl")
    df.columns = normalize_colnames(df.columns)
    for c in df.columns:
        if "item" in c.lower() and "code" in c.lower():
            df[c] = df[c].astype(str).str.strip()
    return df

def load_any_table(uploaded_file, key_prefix: str):
    """Charge CSV ou Excel + sélection de feuille pour Excel, normalisation CSV."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return read_csv_safely(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return read_excel_with_sheet_selector(uploaded_file, key_prefix=key_prefix)
    else:
        try:
            return read_csv_safely(uploaded_file)
        except Exception:
            return None

def guess_column(cols, candidates):
    """Devine une colonne via mots-clés ou regex."""
    cols_norm = [c.lower() for c in cols]
    for cand in candidates:
        cand_low = cand.lower()
        for c in cols_norm:
            if cand_low in c:
                return cols[cols_norm.index(c)]
    for cand in candidates:
        if cand.startswith("^") or cand.endswith("$"):
            pat = re.compile(cand, re.I)
            for i, c in enumerate(cols):
                if re.search(pat, c):
                    return cols[i]
    return None

def coerce_numeric(series):
    """Convertit vers numérique en tolérant %, espaces, virgules FR."""
    s = series.astype(str).str.replace("%", "", regex=False)
    s = s.str.replace("\u202f", "", regex=False)  # espace fine
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)      # décimales FR
    return pd.to_numeric(s, errors="coerce")

def strip_accents(text: str) -> str:
    """Supprime les accents pour une comparaison robuste."""
    try:
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('utf-8')
        return text
    except Exception:
        return text

def clean_text(t: str) -> str:
    """Nettoie/normalise texte pour matching."""
    if t is None:
        return ""
    t = str(t).lower()
    t = strip_accents(t)
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def token_set(text: str):
    return set(clean_text(text).split())

from difflib import SequenceMatcher
def similarity(a: str, b: str):
    """Similarité combinée: difflib + Jaccard (0.6/0.4)."""
    a_clean, b_clean = clean_text(a), clean_text(b)
    if not a_clean and not b_clean:
        return 0.0
    seq = SequenceMatcher(None, a_clean, b_clean).ratio()
    ta, tb = token_set(a_clean), token_set(b_clean)
    inter = len(ta & tb)
    union = len(ta | tb) if (ta or tb) else 1
    jacc = inter / union
    return 0.6 * seq + 0.4 * jacc

def find_best_match(query_text: str, candidates: pd.Series, top_n=3):
    scores = candidates.fillna("").astype(str).apply(lambda x: similarity(query_text, x))
    top_idx = scores.sort_values(ascending=False).head(top_n).index
    return pd.DataFrame({
        "po_index": top_idx,
        "po_description": candidates.loc[top_idx].values,
        "score": scores.loc[top_idx].values
    })

def to_excel_bytes(df_dict: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in df_dict.items():
            pd.DataFrame(df).to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()

# =========================
# Sidebar: chargement
# =========================
st.sidebar.title("⚙️ Paramètres")

st.sidebar.header("1) Charger les fichiers")
po_file    = st.sidebar.file_uploader("Fichier POs (CSV/XLSX)",   type=["csv", "xlsx", "xls"], key="po_file")
stock_file = st.sidebar.file_uploader("Fichier Stock (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="stock_file")

po_df    = load_any_table(po_file,    key_prefix="PO")
stock_df = load_any_table(stock_file, key_prefix="STOCK")

# =========================
# Titre
# =========================
st.title("📦 Comparaison Stock (Total livré) vs Quantité Commandée (PO)")

# Aperçus
col1, col2 = st.columns(2)
with col1:
    if po_df is not None:
        st.subheader("Aperçu POs")
        st.dataframe(po_df.head(25), use_container_width=True)
with col2:
    if stock_df is not None:
        st.subheader("Aperçu Stock")
        st.dataframe(stock_df.head(25), use_container_width=True)

if (po_df is None) or (stock_df is None):
    st.info("➡️ Merci d’uploader **les deux fichiers**. Cette version normalise les CSV et permet la sélection de feuille Excel.")
    st.stop()

# =========================
# Mapping des colonnes
# =========================
st.sidebar.header("2) Mapper les colonnes")

# POs
st.sidebar.caption("🟦 Colonnes POs")
po_item_col = st.sidebar.selectbox(
    "Colonne Item Code (PO)",
    options=po_df.columns,
    index=(po_df.columns.tolist().index(guess_column(po_df.columns, ["item code", "item", "code"]))
           if guess_column(po_df.columns, ["item code", "item", "code"]) in po_df.columns else 0)
)
po_qty_col = st.sidebar.selectbox(
    "Colonne Quantité (PO)",
    options=po_df.columns,
    index=(po_df.columns.tolist().index(guess_column(po_df.columns, ["qty", "quantity", "^qte$", "qte"]))
           if guess_column(po_df.columns, ["qty", "quantity", "^qte$", "qte"]) in po_df.columns else 0)
)
po_date_col = st.sidebar.selectbox(
    "Colonne Date PO (facultatif)",
    options=["(aucune)"] + po_df.columns.tolist(),
    index=(po_df.columns.tolist().index(guess_column(po_df.columns, ["created_at", "po date", "date"])) + 1
           if guess_column(po_df.columns, ["created_at", "po date", "date"]) in po_df.columns else 0)
)
po_desc_col = st.sidebar.selectbox(
    "Colonne Description (POs - pour matching)",
    options=["(aucune)"] + po_df.columns.tolist(),
    index=(po_df.columns.tolist().index(guess_column(po_df.columns, ["description", "desc"])) + 1
           if guess_column(po_df.columns, ["description", "desc"]) in po_df.columns else 0)
)

# Stock
st.sidebar.caption("🟩 Colonnes Stock")
stock_item_col = st.sidebar.selectbox(
    "Colonne Item Code (Stock)",
    options=stock_df.columns,
    index=(stock_df.columns.tolist().index(guess_column(stock_df.columns, ["item code", "item", "code"]))
           if guess_column(stock_df.columns, ["item code", "item", "code"]) in stock_df.columns else 0)
)
stock_qty_col = st.sidebar.selectbox(
    "Colonne Quantité Stock (Total livré / Stock Qty)",
    options=stock_df.columns,
    index=(stock_df.columns.tolist().index(guess_column(stock_df.columns, ["total livré", "stock qty", "total stock", "qty", "quantity"]))
           if guess_column(stock_df.columns, ["total livré", "stock qty", "total stock", "qty", "quantity"]) in stock_df.columns else 0)
)
stock_sapname_col = st.sidebar.selectbox(
    "Colonne SAP Name (Stock - pour matching)",
    options=["(aucune)"] + stock_df.columns.tolist(),
    index=(stock_df.columns.tolist().index(guess_column(stock_df.columns, ["sap name", "sap", "designation", "item name"])) + 1
           if guess_column(stock_df.columns, ["sap name", "sap", "designation", "item name"]) in stock_df.columns else 0)
)

# =========================
# Préparation & filtres
# =========================
po_df["_item"] = po_df[po_item_col].astype(str).str.strip()
stock_df["_item"] = stock_df[stock_item_col].astype(str).str.strip()

po_df["_qty"] = coerce_numeric(po_df[po_qty_col])
stock_df["_stock_qty"] = coerce_numeric(stock_df[stock_qty_col])

if po_date_col and po_date_col != "(aucune)":
    po_df["_date"] = pd.to_datetime(po_df[po_date_col], errors="coerce", dayfirst=True)
else:
    po_df["_date"] = pd.NaT

st.sidebar.header("3) Filtres")
alert_threshold = st.sidebar.slider("Seuil Alerte % écarts négatifs", 0, 100, 20, help="Au-delà, alerte rouge globale")
if po_df["_date"].notna().any():
    dmin = pd.to_datetime(po_df["_date"].min())
    dmax = pd.to_datetime(po_df["_date"].max())
    date_range = st.sidebar.date_input("Filtrer par date PO", (dmin.date(), dmax.date()))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        po_df = po_df[(po_df["_date"] >= start) & (po_df["_date"] <= end)]
else:
    st.sidebar.caption("Aucune date valide détectée dans les POs.")

item_filter = st.sidebar.text_input("Filtrer par Item Code (contient…)", "")
if item_filter.strip():
    po_df = po_df[po_df["_item"].str.contains(item_filter.strip(), case=False, na=False)]

# =========================
# Agrégations principales
# =========================
agg_po = (po_df
          .groupby("_item", dropna=False)
          .agg(
              total_commande=("_qty", "sum"),
              nb_pos=("_qty", "count"),
              derniere_date=("_date", "max"),
              description=(po_desc_col, "last") if (po_desc_col and po_desc_col != "(aucune)") else ("_item", "first")
          )
          .reset_index()
         )

agg_stock = (stock_df
             .groupby("_item", dropna=False)
             .agg(
                 stock_total=("_stock_qty", "sum"),
                 sap_name=(stock_sapname_col, "last") if (stock_sapname_col and stock_sapname_col != "(aucune)") else ("_item", "first")
             )
             .reset_index()
            )

res = agg_po.merge(agg_stock, on="_item", how="outer")
res["total_commande"] = res["total_commande"].fillna(0)
res["stock_total"] = res["stock_total"].fillna(0)

res["ecart_stock_moins_commande"] = res["stock_total"] - res["total_commande"]
res["taux_couverture"] = np.where(res["total_commande"] > 0,
                                  res["stock_total"] / res["total_commande"],
                                  np.nan)

final_cols = {
    "_item": "Item Code",
    "description": "Description (PO)",
    "sap_name": "SAP Name (Stock)",
    "nb_pos": "Nb POs",
    "derniere_date": "Dernière date PO",
    "total_commande": "Total commandé",
    "stock_total": "Stock (Total livré)",
    "ecart_stock_moins_commande": "Écart = Stock - Commandé",
    "taux_couverture": "Taux de couverture"
}
res = res.rename(columns=final_cols)
res = res[list(final_cols.values())]

# =========================
# KPIs + ALERTES ROUGES
# =========================
all_items_mask = (res["Total commandé"].fillna(0) + res["Stock (Total livré)"].fillna(0)) > 0
nb_total_items = int(all_items_mask.sum())
nb_neg = int((res["Écart = Stock - Commandé"] < 0).sum())
nb_pos = int((res["Écart = Stock - Commandé"] > 0).sum())
pct_neg = (nb_neg / nb_total_items * 100) if nb_total_items else 0
pct_pos = (nb_pos / nb_total_items * 100) if nb_total_items else 0

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Total commandé (tous items)", f"{res['Total commandé'].sum():,.0f}")
with colB:
    st.metric("Stock total (tous items)", f"{res['Stock (Total livré)'].sum():,.0f}")
with colC:
    st.metric("Écart global (Stock - Commandé)",
              f"{(res['Stock (Total livré)'].sum() - res['Total commandé'].sum()):,.0f}")
with colD:
    st.metric("Écarts (+ / -)", f"{pct_pos:.1f}% / {pct_neg:.1f}%")

def show_alerts():
    messages = []
    if pct_neg >= alert_threshold:
        messages.append(f"⚠️ {pct_neg:.1f}% des items ont un **écart négatif** (stock < commandé).")
    # Orphelins
    _orph = res[(res["Stock (Total livré)"] > 0) & (res["Total commandé"] == 0)]
    if len(_orph) > 0:
        messages.append(f"📦 **{len(_orph)} item(s) en stock sans PO** (orphelins).")
    # Items inexistants (ni stock ni PO)
    _void = res[(res["Stock (Total livré)"] == 0) & (res["Total commandé"] == 0)]
    if len(_void) > 0:
        messages.append(f"ℹ️ {len(_void)} référence(s) sans activité (ni stock ni commande) dans la plage filtrée.")
    if messages:
        st.error(" / ".join(messages))
    else:
        st.success("Tout est OK ✅ : aucun signal critique sur la période.")

show_alerts()

# =========================
# Détails (table principale)
# =========================
st.markdown("### 🧾 Détails par Item")
def color_ecart(val):
    try:
        v = float(val)
        if v < 0:  # négatif -> rouge
            return "background-color:#ffefef;color:#b00020;font-weight:700;"
        elif v > 0:  # positif -> vert clair
            return "background-color:#eaffea;color:#0b6b2a;font-weight:700;"
    except:
        pass
    return ""
styled = (res
          .style
          .format({
              "Total commandé": "{:,.0f}",
              "Stock (Total livré)": "{:,.0f}",
              "Écart = Stock - Commandé": "{:,.0f}",
              "Taux de couverture": "{:.2%}"
          })
          .applymap(color_ecart, subset=["Écart = Stock - Commandé"])
         )
st.dataframe(styled, use_container_width=True, hide_index=True)

# =========================
# Analyse avancée : Orphelins & matching
# =========================
st.markdown("## 🔍 Analyse avancée : Items en Stock sans PO & correspondances par similarité")

st.sidebar.header("4) Paramètres de matching")
similarity_threshold = st.sidebar.slider("Seuil de similarité (0–1)", 0.0, 1.0, 0.65, 0.05)
top_n_matches = st.sidebar.slider("Top-N correspondances par item", 1, 5, 3)

# Orphans = items avec stock>0 ET total_commande==0
orphans = res[(res["Stock (Total livré)"] > 0) & (res["Total commandé"] == 0)].copy()
st.write(f"**Items orphelins détectés (stock > 0, aucune commande)** : {len(orphans)}")

# Séries POs pour matching
if po_desc_col and po_desc_col != "(aucune)" and po_desc_col in po_df.columns:
    po_desc_clean = po_df[po_desc_col].fillna("").astype(str)
else:
    po_desc_clean = pd.Series(dtype=str)

po_item_series = po_df["_item"].fillna("").astype(str)
po_qty_series  = po_df["_qty"].reset_index(drop=True)

matches_rows = []
if not orphans.empty and not po_desc_clean.empty and (stock_sapname_col and stock_sapname_col != "(aucune)"):
    po_desc_indexed = po_desc_clean.reset_index(drop=False)
    po_desc_indexed.columns = ["po_row_index", "po_description"]
    po_desc_indexed["po_item_code"] = po_item_series.reset_index(drop=True)

    for _, r in orphans.iterrows():
        stock_item   = r["Item Code"]
        stock_name   = r["SAP Name (Stock)"]
        stock_qty    = r["Stock (Total livré)"]

        top_df = find_best_match(stock_name, po_desc_indexed["po_description"], top_n=top_n_matches)

        top_df["po_item_code"] = top_df["po_index"].apply(
            lambda i: po_desc_indexed.loc[i, "po_item_code"] if i in po_desc_indexed.index else np.nan
        )
        top_df["po_qty_line"] = top_df["po_index"].apply(
            lambda i: po_qty_series.iloc[i] if i < len(po_qty_series) else np.nan
        )

        top_df["stock_item_code"] = stock_item
        top_df["stock_sap_name"]  = stock_name
        top_df["stock_qty"]       = stock_qty

        top_df = top_df[top_df["score"] >= similarity_threshold]
        if not top_df.empty:
            matches_rows.append(top_df)

if matches_rows:
    matches_df = pd.concat(matches_rows, ignore_index=True)
    matches_df["po_qty_line"] = pd.to_numeric(matches_df["po_qty_line"], errors="coerce")
    agg_matches = (matches_df
                   .groupby(["stock_item_code", "stock_sap_name", "po_description", "po_item_code"], dropna=False)
                   .agg(
                       similarity=("score", "max"),
                       stock_qty=("stock_qty", "first"),
                       total_po_qty_assoc=("po_qty_line", "sum")
                   ).reset_index())
else:
    agg_matches = pd.DataFrame(columns=["stock_item_code", "stock_sap_name", "po_description", "po_item_code", "similarity", "stock_qty", "total_po_qty_assoc"])

st.markdown("### 🧩 Correspondances proposées (SAP Name ~ Description PO)")
if agg_matches.empty:
    st.info("Aucune correspondance proposée selon le seuil actuel. Essaie d’abaisser le seuil de similarité.")
else:
    st.dataframe(
        agg_matches.sort_values(["similarity", "stock_qty"], ascending=[False, False]),
        use_container_width=True
    )

# ---- Tableau demandé : les orphelins + meilleur match (desc + item code PO) ----
if not agg_matches.empty:
    best_matches = (agg_matches.sort_values(["stock_item_code", "similarity"], ascending=[True, False])
                    .groupby("stock_item_code", as_index=False)
                    .first())
    orphans_summary = (orphans[["Item Code", "SAP Name (Stock)", "Stock (Total livré)"]]
                       .merge(best_matches.rename(columns={
                           "stock_item_code": "Item Code",
                           "stock_sap_name": "SAP Name (Stock)",
                           "po_description": "PO description correspondante",
                           "po_item_code": "PO Item Code",
                           "similarity": "Similarité",
                           "total_po_qty_assoc": "Total Qty PO (assoc.)"
                       }),
                              on=["Item Code", "SAP Name (Stock)"],
                              how="left"))
    orphans_summary["Match trouvé ?"] = np.where(orphans_summary["PO description correspondante"].notna(), "Oui", "Non")
else:
    orphans_summary = orphans[["Item Code", "SAP Name (Stock)", "Stock (Total livré)"]].copy()
    orphans_summary["PO description correspondante"] = np.nan
    orphans_summary["PO Item Code"] = np.nan
    orphans_summary["Similarité"] = np.nan
    orphans_summary["Total Qty PO (assoc.)"] = np.nan
    orphans_summary["Match trouvé ?"] = "Non"

st.markdown("### 📋 Orphelins — meilleur match (description & Item Code PO)")
st.dataframe(
    orphans_summary.sort_values(["Match trouvé ?", "Similarité"], ascending=[True, False]),
    use_container_width=True
)

# =========================
# DÉTAILS & Journal anomalies
# =========================
with st.expander("ℹ️ Détails & contrôles qualité (clique pour ouvrir)"):
    st.markdown("""
    **Contrôles effectués par l'application :**
    - Normalisation POs (CSV à séparateur ; , \\t |) et protection **Item Code** en texte.
    - Sélection de **feuille Excel** pour Stock/POs.
    - Agrégation par **Item Code** : *Total commandé*, *Stock (Total livré)*.
    - Calcul **Écart = Stock - Commandé** et **Taux de couverture**.
    - Détection **Orphelins** (stock > 0, aucune commande).
    - Matching par similarité **SAP Name** (Stock) ↔ **Description** (POs) + récupération **PO Item Code**.
    - Export Excel multi-feuilles : Comparaison, Orphans, Correspondances détaillées, Orphelins + meilleur match, Synthèse KPI.
    """)

    st.markdown("**Journal d’anomalies :**")
    anomalies = {
        "Écarts négatifs (stock < commandé)": res[res["Écart = Stock - Commandé"] < 0][["Item Code","Description (PO)","SAP Name (Stock)","Total commandé","Stock (Total livré)","Écart = Stock - Commandé"]],
        "Orphelins (stock > 0, aucune commande)": orphans[["Item Code","SAP Name (Stock)","Stock (Total livré)"]],
        "Références sans activité (0 stock & 0 PO)": res[(res["Stock (Total livré)"] == 0) & (res["Total commandé"] == 0)][["Item Code","Description (PO)","SAP Name (Stock)"]],
    }
    for title, df_ in anomalies.items():
        st.markdown(f"- <span class='badge badge-red'>{title} — {len(df_)}</span>", unsafe_allow_html=True)
        if len(df_) > 0:
            st.dataframe(df_, use_container_width=True)

# =========================
# Visualisation (Top écarts)
# =========================
st.markdown("### 🔎 Top 20 des écarts (absolus)")
top = res.assign(abs_ecart=res["Écart = Stock - Commandé"].abs()).sort_values("abs_ecart", ascending=False).head(20)
chart_data = top[["Item Code", "Écart = Stock - Commandé"]].set_index("Item Code")
st.bar_chart(chart_data)

# =========================
# Export Excel multi-feuilles
# =========================
st.markdown("### 📥 Export")
summary_df = pd.DataFrame({
    "KPI": ["Total commandé", "Stock total", "Écart global", "% Écarts positifs", "% Écarts négatifs",
            "Orphans (Stock sans PO)", "Orphans avec meilleur match"],
    "Valeur": [
        res["Total commandé"].sum(),
        res["Stock (Total livré)"].sum(),
        res["Stock (Total livré)"].sum() - res["Total commandé"].sum(),
        pct_pos,
        pct_neg,
        len(orphans),
        int(orphans_summary["Match trouvé ?"].eq("Oui").sum())
    ]
})

excel_bytes = to_excel_bytes({
    "Comparaison": res,
    "Orphans": orphans,
    "Correspondances_détaillées": agg_matches,
    "Orphans_meilleur_match": orphans_summary,
    "Synthese_KPI": summary_df
})
st.download_button(
    label="Télécharger l’analyse (Excel)",
    data=excel_bytes,
    file_name=f"analyse_stock_vs_pos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Astuce : ajuste le **seuil de similarité** et le **seuil d’alerte** pour ton contexte.")
