# ---------------------------------------------------------------
# app.py — version complète avec centralisation Inventaire,
# analyse Stock vs PO, matching enrichi & analyse anomalies/fraude
# ---------------------------------------------------------------

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

# ================================================================
# BACKGROUND (image + overlay)
# ================================================================
def apply_background(image_path: str, overlay_rgba="rgba(255,255,255,0.88)"):
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            st.warning(f"Image de background introuvable : {image_path}")
            return
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/{img_path.suffix[1:]};base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            .stApp .block-container {{
                background: {overlay_rgba};
                border-radius: 14px;
                padding: 1.2rem 1.6rem;
            }}
            [data-testid="stSidebar"] {{
                background: {overlay_rgba};
            }}
            .badge {{
                display:inline-block; padding:0.15rem 0.5rem;
                border-radius:999px; font-weight:600; font-size:0.85rem;
            }}
            .badge-red  {{ background:#ffe5e5; color:#b00020; border:1px solid #ffb3b3; }}
            .badge-green{{ background:#e7f8ed; color:#0b6b2a; border:1px solid #b3e6c5; }}
            .badge-amber{{ background:#fff5e6; color:#8a4b00; border:1px solid #ffd9a6; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Impossible d'appliquer le background: {e}")

# 👉 mets le bon nom d’image dans ton dossier
apply_background("image_supply_chain_1.png")

# ================================================================
# HELPERS GÉNÉRAUX
# ================================================================
def normalize_colnames(cols):
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]

def detect_delimiter(text: str):
    # ordre de préférence ; \t , |
    candidates = [';', '\t', ',', '|']
    counts = {sep: text.count(sep) for sep in candidates}
    # si égalité, on garde notre préférence
    best = max(candidates, key=lambda s: (counts[s], -candidates.index(s)))
    return best if counts[best] > 0 else None

def read_csv_safely(uploaded_file):
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()

    decoded = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            decoded = raw.decode(enc)
            break
        except Exception:
            continue
    if decoded is None:
        decoded = raw.decode("utf-8", errors="ignore")

    sep = detect_delimiter(decoded)
    df = None
    try:
        df = pd.read_csv(io.StringIO(decoded), sep=sep, engine="python")
    except Exception:
        for s in (None, ';', ',', '\t', '|'):
            try:
                df = pd.read_csv(io.StringIO(decoded), sep=s, engine="python")
                break
            except Exception:
                continue

    # CSV collé en une seule colonne → re-split
    if df is not None and df.shape[1] == 1:
        col = df.columns[0]
        if (";" in col) or df.iloc[:, 0].astype(str).str.contains(";").any():
            split_df = df.iloc[:, 0].astype(str).str.split(";", expand=True)
            header = split_df.iloc[0]
            # Heuristique : la 1ère ligne ressemble à un header ?
            if any(h.lower() in ["item code", "qty", "created_at", "description"] for h in header.astype(str).str.lower()):
                split_df.columns = [re.sub(r"\s+"," ", str(x)).strip() for x in header]
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
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    sheet_names = xls.sheet_names
    st.sidebar.caption(f"📑 Feuilles détectées ({key_prefix})")
    sheet = st.sidebar.selectbox(
        f"Sélectionne la feuille pour {key_prefix}",
        options=sheet_names, index=0, key=f"{key_prefix}_sheet_select"
    )
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, engine="openpyxl")
    df.columns = normalize_colnames(df.columns)
    for c in df.columns:
        if "item" in c.lower() and "code" in c.lower():
            df[c] = df[c].astype(str).str.strip()
    return df

def load_any_table(uploaded_file, key_prefix: str):
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
    cols_norm = [c.lower() for c in cols]
    for cand in candidates:
        cand_low = cand.lower()
        for i, c in enumerate(cols_norm):
            if cand_low in c:
                return cols[i]
    for cand in candidates:
        if cand.startswith("^") or cand.endswith("$"):
            pat = re.compile(cand, re.I)
            for i, c in enumerate(cols):
                if re.search(pat, c):
                    return cols[i]
    return None

def coerce_numeric(series):
    s = series.astype(str).str.replace("%", "", regex=False)
    s = s.str.replace("\u202f", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def strip_accents(text: str) -> str:
    try:
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('utf-8')
        return text
    except Exception:
        return text

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = str(t).lower()
    t = strip_accents(t)
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def token_set(text: str):
    return set(clean_text(text).split())

def similarity(a: str, b: str):
    """Similarité combinée difflib + Jaccard (0.6 / 0.4)."""
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

# ================================================================
# Générateur Excel multi-feuilles
# ================================================================
def to_excel_bytes(df_dict: dict):
    """Crée un fichier Excel multi-feuilles à partir d'un dict {sheet: df}."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in df_dict.items():
            pd.DataFrame(df).to_excel(writer, index=False, sheet_name=str(sheet)[:31])
    return output.getvalue()
# ================================================================
# 0) CENTRALISATION INVENTAIRE MULTI-FICHIERS
# ================================================================
st.sidebar.title("⚙️ Paramètres")
st.sidebar.header("0) Inventaire – Centralisation multi-fichiers")

inventory_files = st.sidebar.file_uploader(
    "Uploader plusieurs fichiers Inventory (Excel)",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    key="inventory_multi"
)

def load_inventory_file(uploaded_file):
    """Charge un fichier inventaire, détecte la bonne feuille, normalise."""
    raw = uploaded_file.getvalue()
    xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    # Feuille la plus probable
    sheet_name = None
    for sn in xls.sheet_names:
        if "inventory" in sn.lower():
            sheet_name = sn
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
    df.columns = normalize_colnames(df.columns)
    df["_source_file"] = uploaded_file.name
    return df

inventory_global = None

if inventory_files:
    st.markdown("## 📦 Inventaire consolidé — aperçu")
    inv_list = [load_inventory_file(f) for f in inventory_files]
    inventory_global = pd.concat(inv_list, ignore_index=True)

    # Normalisations clé
    col_item = guess_column(inventory_global.columns, ["item code", "sap", "code"])
    col_qty  = guess_column(inventory_global.columns, ["quantity", "qty"])
    col_ucost= guess_column(inventory_global.columns, ["unit cost", "unitcost", "cost"])
    col_in   = guess_column(inventory_global.columns, ["stock entrée", "entry", "receive", "date in", "stock in", "date de reception"])
    col_out  = guess_column(inventory_global.columns, ["last output", "sortie", "date out", "dernier mouvement", "last issue"])

    if col_qty:
        inventory_global[col_qty] = coerce_numeric(inventory_global[col_qty])
    if col_ucost:
        inventory_global[col_ucost] = coerce_numeric(inventory_global[col_ucost])
    if col_in:
        inventory_global[col_in] = pd.to_datetime(inventory_global[col_in], errors="coerce", dayfirst=True)
    if col_out:
        inventory_global[col_out] = pd.to_datetime(inventory_global[col_out], errors="coerce", dayfirst=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total QTY consolidé", f"{(inventory_global[col_qty].sum() if col_qty else 0):,.0f}")
    with c2:
        st.metric("Nb références uniques", inventory_global[col_item].nunique() if col_item else 0)
    with c3:
        total_cost = (inventory_global[col_qty] * inventory_global[col_ucost]).sum() if (col_qty and col_ucost) else 0
        st.metric("Total coût théorique", f"{total_cost:,.0f}")

    st.dataframe(inventory_global.head(60), use_container_width=True)

    inv_bytes = to_excel_bytes({"Inventaire_consolidé": inventory_global})
    st.download_button(
        "Télécharger inventaire consolidé (Excel)",
        data=inv_bytes,
        file_name=f"inventory_consolidated_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Option : utiliser inventaire consolidé comme STOCK
use_inventory_as_stock = False
if inventory_global is not None:
    use_inventory_as_stock = st.sidebar.checkbox("➡️ Utiliser inventaire consolidé comme fichier STOCK", value=False)
# ================================================================
# 1) CHARGEMENT FICHIERS PO & STOCK (ou inventaire consolidé)
# ================================================================
st.sidebar.header("1) Charger les fichiers")
po_file = st.sidebar.file_uploader("Fichier POs (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="po_file")
stock_file = None if use_inventory_as_stock else st.sidebar.file_uploader(
    "Fichier Stock (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="stock_file"
)

po_df = load_any_table(po_file, key_prefix="PO")
stock_df = inventory_global.copy() if use_inventory_as_stock else load_any_table(stock_file, key_prefix="STOCK")

# TITRE + Aperçus
st.title("📦 Comparaison Stock (Total livré) vs Quantité Commandée (PO)")

c1, c2 = st.columns(2)
with c1:
    if po_df is not None:
        st.subheader("Aperçu POs")
        st.dataframe(po_df.head(25), use_container_width=True)
with c2:
    if stock_df is not None:
        st.subheader("Aperçu Stock")
        st.dataframe(stock_df.head(25), use_container_width=True)

if (po_df is None) or (stock_df is None):
    st.info("➡️ Merci d’uploader les deux fichiers (ou activer l’inventaire consolidé).")
    st.stop()

# ================================================================
# 2) MAPPING DES COLONNES
# ================================================================
st.sidebar.header("2) Mapper les colonnes")

# POs
st.sidebar.caption("🟦 Colonnes POs")
po_item_guess = guess_column(po_df.columns, ["item code", "item", "code"])
po_qty_guess  = guess_column(po_df.columns, ["qty", "quantity", "^qte$", "qte"])
po_date_guess = guess_column(po_df.columns, ["created_at", "po date", "date"])
po_desc_guess = guess_column(po_df.columns, ["description", "desc"])

po_item_col = st.sidebar.selectbox("Colonne Item Code (PO)", options=po_df.columns,
                                   index=(po_df.columns.get_loc(po_item_guess) if po_item_guess in po_df.columns else 0))
po_qty_col = st.sidebar.selectbox("Colonne Quantité (PO)", options=po_df.columns,
                                  index=(po_df.columns.get_loc(po_qty_guess) if po_qty_guess in po_df.columns else 0))
po_date_col = st.sidebar.selectbox("Colonne Date PO (facultatif)", options=["(aucune)"] + po_df.columns.tolist(),
                                   index=((po_df.columns.get_loc(po_date_guess) + 1) if po_date_guess in po_df.columns else 0))
po_desc_col = st.sidebar.selectbox("Colonne Description (POs - pour matching)", options=["(aucune)"] + po_df.columns.tolist(),
                                   index=((po_df.columns.get_loc(po_desc_guess) + 1) if po_desc_guess in po_df.columns else 0))

# Stock
st.sidebar.caption("🟩 Colonnes Stock")
stock_item_guess = guess_column(stock_df.columns, ["item code", "item", "code"])
stock_qty_guess  = guess_column(stock_df.columns, ["total livré", "stock qty", "total stock", "qty", "quantity", "qte", "stock"])
stock_sap_guess  = guess_column(stock_df.columns, ["sap name", "sap", "designation", "item name"])

stock_item_col = st.sidebar.selectbox("Colonne Item Code (Stock)", options=stock_df.columns,
                                      index=(stock_df.columns.get_loc(stock_item_guess) if stock_item_guess in stock_df.columns else 0))
stock_qty_col = st.sidebar.selectbox("Colonne Quantité Stock (Total livré / Stock Qty)", options=stock_df.columns,
                                     index=(stock_df.columns.get_loc(stock_qty_guess) if stock_qty_guess in stock_df.columns else 0))
stock_sapname_col = st.sidebar.selectbox("Colonne SAP Name (Stock - pour matching)", options=["(aucune)"] + stock_df.columns.tolist(),
                                         index=((stock_df.columns.get_loc(stock_sap_guess) + 1) if stock_sap_guess in stock_df.columns else 0))

# ================================================================
# 3) PRÉPARATION & FILTRES
# ================================================================
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

# ================================================================
# 4) AGRÉGATIONS PRINCIPALES
# ================================================================
agg_po = (po_df
          .groupby("_item", dropna=False)
          .agg(
              total_commande=("_qty", "sum"),
              nb_pos=("_qty", "count"),
              derniere_date=("_date", "max"),
              description=(po_desc_col, "last") if (po_desc_col and po_desc_col != "(aucune)") else ("_item", "first")
          )
          .reset_index())

agg_stock = (stock_df
             .groupby("_item", dropna=False)
             .agg(
                 stock_total=("_stock_qty", "sum"),
                 sap_name=(stock_sapname_col, "last") if (stock_sapname_col and stock_sapname_col != "(aucune)") else ("_item", "first")
             )
             .reset_index())

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

# ================================================================
# 5) KPIs + ALERTES
# ================================================================
all_items_mask = (res["Total commandé"].fillna(0) + res["Stock (Total livré)"].fillna(0)) > 0
nb_total_items = int(all_items_mask.sum())
nb_neg = int((res["Écart = Stock - Commandé"] < 0).sum())
nb_pos = int((res["Écart = Stock - Commandé"] > 0).sum())
pct_neg = (nb_neg / nb_total_items * 100) if nb_total_items else 0
pct_pos = (nb_pos / nb_total_items * 100) if nb_total_items else 0

cA, cB, cC, cD = st.columns(4)
cA.metric("Total commandé (tous items)", f"{res['Total commandé'].sum():,.0f}")
cB.metric("Stock total (tous items)", f"{res['Stock (Total livré)'].sum():,.0f}")
cC.metric("Écart global (Stock - Commandé)",
          f"{(res['Stock (Total livré)'].sum() - res['Total commandé'].sum()):,.0f}")
cD.metric("Écarts (+ / -)", f"{pct_pos:.1f}% / {pct_neg:.1f}%")

def show_alerts():
    messages = []
    if pct_neg >= alert_threshold:
        messages.append(f"⚠️ {pct_neg:.1f}% des items ont un **écart négatif** (stock < commandé).")
    _orph = res[(res["Stock (Total livré)"] > 0) & (res["Total commandé"] == 0)]
    if len(_orph) > 0:
        messages.append(f"📦 **{len(_orph)} item(s) en stock sans PO** (orphelins).")
    _void = res[(res["Stock (Total livré)"] == 0) & (res["Total commandé"] == 0)]
    if len(_void) > 0:
        messages.append(f"ℹ️ {len(_void)} référence(s) sans activité dans la plage filtrée.")
    if messages:
        st.error(" / ".join(messages))
    else:
        st.success("Tout est OK ✅ : aucun signal critique sur la période.")

show_alerts()

# ================================================================
# 6) DÉTAILS (table principale)
# ================================================================
st.markdown("### 🧾 Détails par Item")
st.dataframe(
    res,
    use_container_width=True
)

# ================================================================
# 7) ANALYSE AVANCÉE : Orphelins & Matching enrichi
# ================================================================
st.markdown("## 🔍 Analyse avancée : Items en Stock sans PO & correspondances par similarité")
st.sidebar.header("4) Paramètres de matching")
similarity_threshold = st.sidebar.slider("Seuil de similarité (0–1)", 0.0, 1.0, 0.65, 0.05)
top_n_matches = st.sidebar.slider("Top-N correspondances par item", 1, 5, 3)

orphans = res[(res["Stock (Total livré)"] > 0) & (res["Total commandé"] == 0)].copy()
st.write(f"**Items orphelins détectés** : {len(orphans)}")

# Prépare séries POs pour matching
if po_desc_col and po_desc_col != "(aucune)" and po_desc_col in po_df.columns:
    po_desc_clean = po_df[po_desc_col].fillna("").astype(str)
else:
    po_desc_clean = pd.Series(dtype=str)

po_item_series = po_df["_item"].fillna("").astype(str).reset_index(drop=True)
po_qty_series  = po_df["_qty"].reset_index(drop=True)

# PO number auto
po_number_col_guess = guess_column(
    po_df.columns,
    ["po number", "po no", "po_n", "po id", "order number", "numéro po", "numero po", "reference", "ref po", "^po$"]
)
if po_number_col_guess and po_number_col_guess in po_df.columns:
    po_number_series = po_df[po_number_col_guess].astype(str).fillna("").reset_index(drop=True)
else:
    po_number_series = pd.Series([np.nan] * len(po_df), dtype=object)

# Date PO
po_date_series = (po_df["_date"].reset_index(drop=True)
                  if "_date" in po_df.columns else pd.Series([pd.NaT] * len(po_df)))

matches_rows = []
if not orphans.empty and not po_desc_clean.empty and (stock_sapname_col and stock_sapname_col != "(aucune)"):
    po_desc_indexed = po_desc_clean.reset_index(drop=False)
    po_desc_indexed.columns = ["po_row_index", "po_description"]
    po_desc_indexed["po_item_code"] = po_item_series
    po_desc_indexed["po_number"]    = po_number_series
    po_desc_indexed["po_date"]      = po_date_series

    for _, r in orphans.iterrows():
        stock_item   = r["Item Code"]
        stock_name   = r["SAP Name (Stock)"]
        stock_qty    = r["Stock (Total livré)"]

        # Réduction d'espace de recherche simple (tokens communs)
        tokens = token_set(stock_name)
        if len(tokens) >= 2:
            mask = po_desc_indexed["po_description"].astype(str).apply(lambda x: len(token_set(x) & tokens) >= 1)
            cands = po_desc_indexed.loc[mask, "po_description"]
            if cands.empty:
                cands = po_desc_indexed["po_description"]
        else:
            cands = po_desc_indexed["po_description"]

        top_df = find_best_match(stock_name, cands, top_n=top_n_matches)

        # Récup champs PO via l'index
        top_df["po_item_code"] = top_df["po_index"].apply(
            lambda i: po_desc_indexed.loc[i, "po_item_code"] if i in po_desc_indexed.index else np.nan
        )
        top_df["po_qty_line"] = top_df["po_index"].apply(
            lambda i: po_qty_series.iloc[i] if i < len(po_qty_series) else np.nan
        )
        top_df["po_number"] = top_df["po_index"].apply(
            lambda i: po_desc_indexed.loc[i, "po_number"] if i in po_desc_indexed.index else np.nan
        )
        top_df["po_date"] = top_df["po_index"].apply(
            lambda i: po_desc_indexed.loc[i, "po_date"] if i in po_desc_indexed.index else pd.NaT
        )

        # Contexte Stock
        top_df["stock_item_code"] = stock_item
        top_df["stock_sap_name"]  = stock_name
        top_df["stock_qty"]       = stock_qty

        # Filtre de similarité
        top_df = top_df[top_df["score"] >= similarity_threshold]
        if not top_df.empty:
            matches_rows.append(top_df)

if matches_rows:
    matches_df = pd.concat(matches_rows, ignore_index=True)
    matches_df["po_qty_line"] = pd.to_numeric(matches_df["po_qty_line"], errors="coerce")

    # Agrégation enrichie
    agg_matches = (matches_df
                   .groupby(["stock_item_code", "stock_sap_name", "po_description", "po_item_code"], dropna=False)
                   .agg(
                       similarity=("score", "max"),
                       stock_qty=("stock_qty", "first"),
                       total_po_qty_assoc=("po_qty_line", "sum"),
                       po_number=("po_number", "first"),
                       po_date=("po_date", "max")
                   ).reset_index())

    agg_matches["po_code_mismatch"] = np.where(
        agg_matches["stock_item_code"].astype(str) != agg_matches["po_item_code"].astype(str),
        "Oui", "Non"
    )
    agg_matches["po_date"] = pd.to_datetime(agg_matches["po_date"], errors="coerce").dt.date
else:
    agg_matches = pd.DataFrame(
        columns=["stock_item_code", "stock_sap_name", "po_description", "po_item_code",
                 "similarity", "stock_qty", "total_po_qty_assoc", "po_number", "po_date", "po_code_mismatch"]
    )

st.markdown("### 🧩 Correspondances proposées (SAP Name ~ Description PO)")
if agg_matches.empty:
    st.info("Aucune correspondance proposée selon le seuil actuel. Essaie d’abaisser le seuil de similarité.")
else:
    display_cols = [
        "stock_item_code", "stock_sap_name",
        "po_description", "po_item_code", "po_number", "po_date",
        "similarity", "stock_qty", "total_po_qty_assoc", "po_code_mismatch"
    ]
    st.dataframe(
        agg_matches.sort_values(["similarity", "stock_qty"], ascending=[False, False])[display_cols],
        use_container_width=True
    )

# Best match par orphelin
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
                           "total_po_qty_assoc": "Total Qty PO (assoc.)",
                           "po_number": "PO Number",
                           "po_date": "PO Date",
                           "po_code_mismatch": "PO ≠ Stock ?"
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
    orphans_summary["PO Number"] = np.nan
    orphans_summary["PO Date"] = np.nan
    orphans_summary["PO ≠ Stock ?"] = np.nan
    orphans_summary["Match trouvé ?"] = "Non"

st.markdown("### 📋 Orphelins — meilleur match (description & Item Code PO)")
st.dataframe(
    orphans_summary.sort_values(["Match trouvé ?", "Similarité"], ascending=[True, False]),
    use_container_width=True
)

# ================================================================
# 8) DÉTAILS & Journal anomalies (expander)
# ================================================================
with st.expander("ℹ️ Détails & contrôles qualité (clique pour ouvrir)"):
    st.markdown("""
    **Contrôles effectués :**
    - Normalisation POs (CSV à séparateur ; , \\t |) et protection **Item Code** en texte.
    - Sélection de **feuille Excel** pour Stock/POs.
    - Agrégation par **Item Code** : *Total commandé*, *Stock (Total livré)*.
    - Calcul **Écart = Stock - Commandé** et **Taux de couverture**.
    - Détection **Orphelins** (stock > 0, aucune commande).
    - Matching par similarité **SAP Name** (Stock) ↔ **Description** (POs) + récupération **PO Item Code**.
    - Export Excel multi-feuilles enrichi (comparaison, orphelins, correspondances, synthèse KPI).
    - ✨ Colonnes enrichies : **PO Number**, **PO Date**, **PO ≠ Stock ?**.
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

# ================================================================
# 9) Visualisation (Top 20 des écarts absolus)
# ================================================================
st.markdown("### 🔎 Top 20 des écarts (absolus)")
top = res.assign(abs_ecart=res["Écart = Stock - Commandé"].abs()).sort_values("abs_ecart", ascending=False).head(20)
chart_data = top[["Item Code", "Écart = Stock - Commandé"]].set_index("Item Code")
st.bar_chart(chart_data)
# =============================================================================
# 🔥 ANALYSE AVANCÉE — anomalies, incohérences & score de risque
# =============================================================================
st.markdown("## 🔥 Analyse avancée des anomalies & risques")

# A. ECARTS
ecarts_negatifs = res[res["Écart = Stock - Commandé"] < 0]
ecarts_pos_forts = res[res["Écart = Stock - Commandé"] > (res["Total commandé"] * 0.5)]

st.markdown("### 🔵 Écarts négatifs (Stock < Commandé)")
st.dataframe(ecarts_negatifs, use_container_width=True)

st.markdown("### 🟠 Écarts positifs extrêmes (Stock >> Commandé)")
st.dataframe(ecarts_pos_forts, use_container_width=True)

# B. ORPHELINS
st.markdown("### 🟣 Stock > 0 sans PO (orphelins)")
st.dataframe(orphans, use_container_width=True)

# C. ANOMALIES POs
po_anomalies_qty0 = po_df[po_df["_qty"] == 0]
po_desc_missing = (po_df[po_desc_col].isna().sum() if (po_desc_col and po_desc_col != "(aucune)" and po_desc_col in po_df.columns) else 0)

st.markdown("### 🟠 Anomalies POs")
st.write(f"- POs avec quantité = 0 : **{len(po_anomalies_qty0)}**")
if po_desc_col and po_desc_col != "(aucune)":
    st.write(f"- Descriptions PO manquantes : **{po_desc_missing}**")

# D. ANOMALIES STOCK
stock_negatif = stock_df[stock_df["_stock_qty"] < 0] if "_stock_qty" in stock_df.columns else pd.DataFrame()
st.markdown("### 🟡 Anomalies Stock")
st.write(f"- Lignes avec stock négatif : **{len(stock_negatif)}**")

# E. DATES (si dispo côté inventaire/stock)
stock_entry_guess = guess_column(stock_df.columns, ["stock entrée", "entry", "receive", "date in", "stock in", "date de reception"])
last_output_guess = guess_column(stock_df.columns, ["last output", "sortie", "date out", "dernier mouvement", "last issue"])

if stock_entry_guess and po_date_col and po_date_col != "(aucune)":
    # rapprochement dates par item (min date PO vs date entrée stock)
    po_min_date = po_df.groupby("_item")["_date"].min().rename("min_po_date")
    st_dates = stock_df[["_item", stock_entry_guess]].rename(columns={stock_entry_guess: "stock_entry_date"})
    date_check = st_dates.merge(po_min_date, left_on="_item", right_index=True, how="left")
    suspicious_early = date_check[(date_check["stock_entry_date"].notna()) & (date_check["min_po_date"].notna()) &
                                  (date_check["stock_entry_date"] < (date_check["min_po_date"] - pd.Timedelta(days=3)))]
    st.markdown("### 🔴 Réceptions avant les dates PO (suspect)")
    st.dataframe(suspicious_early.head(200), use_container_width=True)

# F. PRIX (si Unit Cost dans stock consolidé)
unit_cost_guess = guess_column(stock_df.columns, ["unit cost", "unitcost", "cost unitaire", "pu"])
if unit_cost_guess:
    stock_df["_unit_cost_norm"] = coerce_numeric(stock_df[unit_cost_guess])
    uc = stock_df["_unit_cost_norm"].dropna()
    if not uc.empty:
        threshold_hi = uc.mean() + 3 * uc.std()  # seuil très haut
        anomalie_prix = stock_df[stock_df["_unit_cost_norm"] > threshold_hi]
        st.markdown("### 🔴 Anomalies de prix (au‑delà de 3σ)")
        st.dataframe(anomalie_prix.head(200), use_container_width=True)

# G. SCORE DE RISQUE SIMPLE
def score_risque_row(row):
    score = 0
    # Écart négatif → 30
    if pd.notna(row["Écart = Stock - Commandé"]) and row["Écart = Stock - Commandé"] < 0:
        score += 30
    # Surstock > 50% du commandé → 20
    if pd.notna(row["Écart = Stock - Commandé"]) and pd.notna(row["Total commandé"]) and row["Écart = Stock - Commandé"] > row["Total commandé"] * 0.5:
        score += 20
    # Orphelin → 40
    if pd.notna(row["Total commandé"]) and pd.notna(row["Stock (Total livré)"]) and row["Total commandé"] == 0 and row["Stock (Total livré)"] > 0:
        score += 40
    # Description absente → 10
    if (("Description (PO)" not in row) or (pd.isna(row["Description (PO)"]) or str(row["Description (PO)"]).strip() == "")):
        score += 10
    return min(score, 100)

res["Risque (0-100)"] = res.apply(score_risque_row, axis=1)
st.markdown("### 🔥 Classement des items par risque")
st.dataframe(res.sort_values("Risque (0-100)", ascending=False).head(200), use_container_width=True)

# =============================================================================
# 10) EXPORT EXCEL MULTI-FEUILLES (complet)
# =============================================================================
st.markdown("### 📥 Export Excel")
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
        int(orphans_summary["Match trouvé ?"].eq("Oui").sum()) if "Match trouvé ?" in orphans_summary.columns else 0
    ]
})

excel_bytes = to_excel_bytes({
    "Comparaison": res,
    "Orphans": orphans,
    "Correspondances_détaillées": agg_matches.sort_values(["similarity", "stock_qty"], ascending=[False, False]) if not agg_matches.empty else pd.DataFrame(),
    "Orphans_meilleur_match": orphans_summary,
    "Ecarts_négatifs": ecarts_negatifs,
    "Ecarts_pos_forts": ecarts_pos_forts,
    "Anomalies_PO_qty0": po_anomalies_qty0,
    "Anomalies_stock_négatif": stock_negatif if not stock_negatif.empty else pd.DataFrame(),
    "Synthese_KPI": summary_df
})
st.download_button(
    label="Télécharger l’analyse (Excel)",
    data=excel_bytes,
    file_name=f"analyse_stock_vs_pos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Astuce : ajuste les **seuils** (similarité, alertes) pour ton contexte. 🚀")
# =============================================================================
# 🔮 MODULE FORECAST 2026 — basé sur la consommation 2025
# =============================================================================

st.markdown("## 🔮 Prévision (Forecast) 2026")

# Vérifier si on a des dates de PO
if "_date" not in po_df.columns or po_df["_date"].isna().all():
    st.warning("Impossible de calculer la consommation mensuelle : aucune date PO valide.")
else:
    # extraire uniquement l'année 2025
    po_2025 = po_df[(po_df["_date"].dt.year == 2025)].copy()

    if po_2025.empty:
        st.info("Aucune consommation PO détectée pour 2025.")
    else:
        # consommation mensuelle 2025
        po_2025["month"] = po_2025["_date"].dt.to_period("M")
        cons_monthly = po_2025.groupby(["_item", "month"])["_qty"].sum().reset_index()

        st.markdown("### 📈 Consommation mensuelle (2025)")
        st.dataframe(cons_monthly, use_container_width=True)

        # Forecast basé sur la moyenne mobile
        forecast = cons_monthly.groupby("_item")["_qty"].mean().rename("forecast_mois_2026").reset_index()

        # Ajouter stock actuel
        stock_now = stock_df.groupby("_item")["_stock_qty"].sum().rename("stock_actuel").reset_index()
        forecast = forecast.merge(stock_now, on="_item", how="left")

        # Besoin annuel
        forecast["forecast_2026_annuel"] = forecast["forecast_mois_2026"] * 12

        # Date rupture
        forecast["mois_avant_rupture"] = forecast["stock_actuel"] / forecast["forecast_mois_2026"]
        forecast["date_rupture_estimée"] = pd.to_datetime("2026-01-01") + \
            forecast["mois_avant_rupture"].fillna(0).apply(lambda x: pd.DateOffset(months=int(x)))

        # Besoin d’achat
        forecast["besoin_achat_2026"] = forecast["forecast_2026_annuel"] - forecast["stock_actuel"]
        forecast["besoin_achat_2026"] = forecast["besoin_achat_2026"].clip(lower=0)

        st.markdown("### 🔮 Forecast 2026 (par item)")
        st.dataframe(forecast, use_container_width=True)

        # KPI
        st.markdown("### 📊 Synthèse Forecast")
        tot_need = forecast["besoin_achat_2026"].sum()
        tot_stock = forecast["stock_actuel"].sum()
        st.metric("Total besoin d'achat 2026", f"{tot_need:,.0f}")
        st.metric("Stock total actuel", f"{tot_stock:,.0f}")

        # Export
        st.session_state["forecast_2026"] = forecast