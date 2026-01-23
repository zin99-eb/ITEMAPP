
# app.py
# -------------------------
# Dead Stock Analyzer - Streamlit (Semaines 2025 + Tendance + Recos IA + Debug)
# Auteur : Zineb (augmenté par Copilot)
# -------------------------
import os
import re
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dead Stock Analyzer", layout="wide")

# ---------- Utils
TODAY = pd.to_datetime(date.today())

COUNTRY_LIST = [
    "Benin","Burkina Faso","Cabo Verde","Cameroon","Congo","Congo (Kinshasa)","Côte d'Ivoire","Cote d'Ivoire",
    "Ghana","Guinea","Kenya","Mali","Morocco","Niger","Nigeria","Rwanda","Senegal","Sierra Leone",
    "Tanzania","Togo","Uganda","Zambia","South Africa","Mozambique","Guinea-Bissau","Gabon","Chad",
    "Liberia","Gambia","Mauritania","Ethiopia","Angola","Somalia","Sudan","South Sudan","Algeria","Tunisia","Libya"
]
COUNTRY_NORMALIZE = {
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "rdc": "Congo (Kinshasa)",
    "drc": "Congo (Kinshasa)",
    "congo kinshasa": "Congo (Kinshasa)",
    "congo drc": "Congo (Kinshasa)",
    "tanzanie": "Tanzania",
    "maroc": "Morocco",
    "cameroun": "Cameroon",
    "algérie": "Algeria",
    "tunisie": "Tunisia",
    "royaume du maroc": "Morocco",
    "cabo-verde": "Cabo Verde",
}

def normalize_country(text):
    if pd.isna(text):
        return None
    t = str(text).strip()
    candidates = [w for w in t.replace("_"," ").replace("-"," ").split() if w.strip()]
    if candidates:
        for k in range(1, min(3, len(candidates)) + 1):
            tail = " ".join(candidates[-k:]).lower()
            if tail in COUNTRY_NORMALIZE:
                return COUNTRY_NORMALIZE[tail]
            for c in COUNTRY_LIST:
                if tail == c.lower():
                    return c
    low = t.lower()
    for c in COUNTRY_LIST:
        if c.lower() in low:
            return c
    for k,v in COUNTRY_NORMALIZE.items():
        if k in low:
            return v
    return None

# Détection automatique des colonnes (FR/EN)
CANDIDATES = {
    "affiliate": ["Affiliate Name","Affiliate","Filiale","Entité","Country","Pays"],
    "item_code": ["Item code (SAP)","Item code","Code article","SKU","Item","Article"],
    "item_desc": ["Designation","Désignation","Description","Libellé","Item Description"],
    "category": ["Category","Catégorie"],
    "uom": ["UOM","u.m","Unit","Unité","U.M","UM"],
    "quantity": ["Quantity","Qty","Qte","Quantité"],
    "unit_price": ["Unit Price","Prix Unitaire","PU","Price"],
    "currency": ["Local currency","Devise","Currency"],
    "total_cost_usd": ["Total Cost USD","Valeur USD","Total USD","Cost USD","Stock Value USD","Total Price USD"],
    "stock_in_date": ["Stock entrance date","Date d'entrée","Date entrée","Date d’entrée stock","Stock entry date"],
    "last_move_date": ["Last moving date","Dernier mouvement","Last movement date"],
    "aging_text": ["Stock Aging","Aging","Tranche d'âge","Ancienneté"],
    "status_date": ["Status date","Status data","Date statut","Status","Semaine","Week"],
}

def auto_map_columns(df: pd.DataFrame):
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for std, names in CANDIDATES.items():
        hit = None
        for n in names:
            if n in df.columns:
                hit = n; break
            if n.lower() in cols_lower:
                hit = cols_lower[n.lower()]; break
        mapping[std] = hit
    return mapping

def parse_dates_safe(series):
    if series is None:
        return None
    try:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
        # Excel serial fallback
        if pd.api.types.is_numeric_dtype(series) and s.isna().all():
            s = pd.to_datetime("1899-12-30") + pd.to_timedelta(series, unit="D")
        return s
    except Exception:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

def compute_value_usd(row, qty_col, unit_col, usd_col):
    if usd_col and pd.notna(row.get(usd_col)):
        return row[usd_col]
    if unit_col and qty_col and pd.notna(row.get(unit_col)) and pd.notna(row.get(qty_col)):
        return row[unit_col] * row[qty_col]
    return np.nan

def bucketize_days(d):
    if pd.isna(d):
        return "Inconnu"
    d = int(d)
    if d < 180: return "< 6 mois"
    if d < 365: return "6–12 mois"
    if d < 730: return "1–2 ans"
    if d < 1095: return "2–3 ans"
    if d < 1460: return "3–4 ans"
    if d < 1825: return "4–5 ans"
    return "> 5 ans"

def to_excel_bytes(dfs_dict: dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in dfs_dict.items():
            sheetname = (sheet or "Sheet1")[:31]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(writer, index=False, sheet_name=sheetname)
    bio.seek(0)
    return bio

# ---------- Parsing des semaines de "Status date"
# Autorise: "W09", "W9", "W09/2025", "W09 2025", "W09-2025", "Week 09 2025", "2025 W09", "W09-25", etc.
WEEK_REGEX = re.compile(r'(?:^|\b)w(?:eek)?\s*(\d{1,2})(?:\D+(\d{2,4}))?', re.IGNORECASE)

def parse_status_week(series, default_year=2025):
    def _one(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        m = WEEK_REGEX.search(s)
        year = default_year
        if not m:
            # Essai basique si juste "09/2025"
            m2 = re.search(r'\b(\d{1,2})\s*[\/\-]\s*(\d{2,4})\b', s)
            if m2:
                week = int(m2.group(1))
                y = int(m2.group(2))
                if y < 100: y = 2000 + y
                year = y
            else:
                # Dernière chance : chiffre seul = semaine (année par défaut)
                m3 = re.search(r'\b(\d{1,2})\b', s)
                if not m3:
                    return pd.NaT
                week = int(m3.group(1))
        else:
            week = int(m.group(1))
            if m.group(2):
                y = int(m.group(2))
                if y < 100: y = 2000 + y
                year = y
        try:
            # lundi ISO de la semaine
            return pd.Timestamp(pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u"))
        except Exception:
            # fallback simple
            return pd.to_datetime(f"{year}-01-01") + pd.to_timedelta((week-1)*7, unit="D")
    return series.apply(_one)

# ---------- Sidebar
st.sidebar.title("⚙️ Paramètres")
uploaded = st.sidebar.file_uploader("Charge ton fichier (Excel .xlsx/.xls ou CSV)", type=["xlsx","xls","csv"])

if not uploaded:
    st.title("📦 Dead Stock Analyzer")
    st.info("👉 Charge un fichier pour commencer. Si c’est un Excel avec plusieurs feuilles, tu pourras choisir celle à analyser (ex. 'overview stock -GE -SP W31').")
    st.stop()

# ---------- Sélecteur de feuille(s) si Excel
file_name = uploaded.name.lower()
is_excel = file_name.endswith(".xlsx") or file_name.endswith(".xls")

df_raw = None
uploaded_bytes = uploaded.getvalue()

if is_excel:
    engine = "openpyxl" if file_name.endswith(".xlsx") else "xlrd"
    try:
        xl = pd.ExcelFile(BytesIO(uploaded_bytes), engine=engine)
        sheet_names = xl.sheet_names
    except Exception:
        xl = pd.ExcelFile(BytesIO(uploaded_bytes), engine="openpyxl")
        sheet_names = xl.sheet_names

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗂️ Feuilles Excel")

    preferred_order = ["overview stock -ge -sp w31", "overview", "stock", "ge", "sp", "w31"]
    def pick_default_sheet(names):
        low = [s.lower() for s in names]
        for p in preferred_order:
            for i, s in enumerate(low):
                if p in s:
                    return names[i]
        return names[0] if names else None

    combine = st.sidebar.checkbox("Combiner plusieurs feuilles", value=False)
    if combine:
        default_sheet = pick_default_sheet(sheet_names)
        selected_sheets = st.sidebar.multiselect(
            "Sélectionne les feuilles à combiner",
            options=sheet_names,
            default=[default_sheet] if default_sheet else []
        )
        if not selected_sheets:
            st.warning("Sélectionne au moins une feuille.")
            st.stop()
        frames = []
        for sh in selected_sheets:
            try:
                frames.append(pd.read_excel(BytesIO(uploaded_bytes), sheet_name=sh, engine="openpyxl"))
            except Exception:
                frames.append(pd.read_excel(BytesIO(uploaded_bytes), sheet_name=sh))
        df_raw = pd.concat(frames, ignore_index=True, sort=False)
        st.success(f"Feuilles combinées : {', '.join(selected_sheets)} → {df_raw.shape[0]} lignes")
    else:
        default_sheet = pick_default_sheet(sheet_names)
        chosen = st.sidebar.selectbox(
            "Choisis la feuille à analyser",
            options=sheet_names,
            index=sheet_names.index(default_sheet) if default_sheet in sheet_names else 0
        )
        try:
            df_raw = pd.read_excel(BytesIO(uploaded_bytes), sheet_name=chosen, engine="openpyxl")
        except Exception:
            df_raw = pd.read_excel(BytesIO(uploaded_bytes), sheet_name=chosen)
        st.success(f"Feuille chargée : {chosen} ({df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes)")
else:
    try:
        df_raw = pd.read_csv(BytesIO(uploaded_bytes))
    except UnicodeDecodeError:
        df_raw = pd.read_csv(BytesIO(uploaded_bytes), encoding="latin-1")
    st.success(f"Fichier CSV chargé : {uploaded.name} ({df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes)")

# ---------- Paramètres dead stock & référence
st.sidebar.markdown("---")
dead_by = st.sidebar.radio("Définition du *dead stock* :", [
    "Jours sans mouvement ≥ seuil",
    "Jours depuis entrée ≥ seuil",
    "Utiliser la colonne 'Stock Aging'"
], index=0)
threshold = st.sidebar.number_input("Seuil (jours)", min_value=30, value=365, step=30)
st.sidebar.markdown("---")
st.sidebar.caption("Date de référence pour les calculs (aujourd’hui par défaut)")
ref_date = st.sidebar.date_input("Date de référence", value=date.today())
REF = pd.to_datetime(ref_date)
st.sidebar.markdown("---")
default_status_year = st.sidebar.number_input("Année par défaut pour 'Status date'", min_value=2000, max_value=2100, value=2025, step=1)
treat_nan_as_dead = st.sidebar.checkbox("🔧 Considérer les dates manquantes comme dead stock", value=True)

st.title("📦 Dead Stock Analyzer")
st.caption("Sélectionne la (les) feuille(s), ajuste la définition du dead stock, explore les KPI, carte, tendances par semaine et recommandations IA.")

# ---------- Mapping colonnes
auto_map = auto_map_columns(df_raw)
with st.expander("🧭 Vérifier/ajuster le mapping des colonnes détectées"):
    cols = {}
    for std_key, label in [
        ("affiliate", "Filiale / Pays"),
        ("item_code", "Code article"),
        ("item_desc", "Désignation"),
        ("category", "Catégorie"),
        ("uom", "Unité (UOM)"),
        ("quantity", "Quantité"),
        ("unit_price", "Prix unitaire"),
        ("currency", "Devise locale"),
        ("total_cost_usd", "Valeur USD (Total Cost USD)"),
        ("stock_in_date", "Date d'entrée stock"),
        ("last_move_date", "Date dernier mouvement"),
        ("aging_text", "Colonne 'Stock Aging' (texte)"),
        ("status_date", "Status date (ex: W31 2025)"),
    ]:
        cols[std_key] = st.selectbox(
            f"{label}",
            options=["--Aucune--"] + list(df_raw.columns),
            index=(list(df_raw.columns).index(auto_map[std_key]) + 1) if auto_map[std_key] in df_raw.columns else 0,
            key=f"map_{std_key}"
        )
    cols = {k:(None if v=="--Aucune--" else v) for k,v in cols.items()}

# --- DEBUG APERÇU DES DONNÉES -------------------------------------------------
with st.expander("🧪 Debug rapide – aperçu du fichier (après mapping)"):
    st.write("**Colonnes mappées** :", cols)
    st.write("**Nb lignes / colonnes** :", df_raw.shape)
    st.dataframe(df_raw.head(20), use_container_width=True)

# ---------- Préparation des données
df = df_raw.copy()

# Dates & dérivés
df["_stock_in"] = parse_dates_safe(df[cols["stock_in_date"]]) if cols["stock_in_date"] in df.columns else pd.NaT
df["_last_move"] = parse_dates_safe(df[cols["last_move_date"]]) if cols["last_move_date"] in df.columns else pd.NaT
df["_days_since_in"] = (REF - df["_stock_in"]).dt.days
df["_days_since_move"] = (REF - df["_last_move"]).dt.days

with st.expander("🧪 Debug calculs de dates"):
    ok_in = df["_stock_in"].notna().mean()*100
    ok_mv = df["_last_move"].notna().mean()*100
    st.write(f"✓ % lignes avec **Date d'entrée** parsée : {ok_in:.1f}%")
    st.write(f"✓ % lignes avec **Dernier mouvement** parsé : {ok_mv:.1f}%")
    st.write("Exemples (5) :")
    dbg_cols = [c for c in [cols.get("stock_in_date"), "_stock_in", cols.get("last_move_date"), "_last_move", "_days_since_in", "_days_since_move"] if c]
    st.dataframe(df[dbg_cols].head(5), use_container_width=True)

# Quantité / Valeur
qty_col = cols["quantity"] if cols["quantity"] in df.columns else None
unit_col = cols["unit_price"] if cols["unit_price"] in df.columns else None
usd_col = cols["total_cost_usd"] if cols["total_cost_usd"] in df.columns else None

df["_value_calc"] = df.apply(lambda r: compute_value_usd(r, qty_col, unit_col, usd_col), axis=1)
used_value_field = "Total Cost USD" if usd_col else "Unit Price × Quantity (monnaie locale)"
st.caption(f"💡 Champ valeur utilisé : **{used_value_field}**")

# Country (dérivé)
aff_col = cols["affiliate"] if cols["affiliate"] in df.columns else None
if aff_col:
    df["_country"] = df[aff_col].apply(normalize_country)
else:
    df["_country"] = None

# Buckets d’âge
df["_bucket_since_in"] = df["_days_since_in"].apply(bucketize_days)
df["_bucket_since_move"] = df["_days_since_move"].apply(bucketize_days)

# Parsing Status date (semaines)
status_col = cols["status_date"] if cols["status_date"] in df.columns else None
if status_col:
    df["_status_week_start"] = parse_status_week(df[status_col], default_year=default_status_year)
    def _label(ts):
        if pd.isna(ts): return None
        iso = ts.isocalendar()
        return f"W{int(iso.week):02d}-{int(iso.year)}"
    df["_status_label"] = df["_status_week_start"].apply(_label)
    df["_status_year"] = df["_status_week_start"].dt.isocalendar().year
else:
    df["_status_week_start"] = pd.NaT
    df["_status_label"] = None
    df["_status_year"] = np.nan

with st.expander("🧪 Debug Status date (semaines)"):
    if df["_status_week_start"].notna().any():
        st.write("Exemples de semaines détectées :",
                 df[["_status_label","_status_week_start"]].dropna().head(10))
        st.write("Nb semaines uniques détectées :", df["_status_label"].nunique())
    else:
        st.error("Aucune semaine détectée à partir de 'Status date'. Vérifie le format (ex: 'W31 2025').")
        st.caption("Formats acceptés: 'W09', 'W9', 'W09 2025', 'W09/25', '2025 W09', 'W09-25', etc.")

# ---------- Définition du dead stock
if dead_by == "Jours sans mouvement ≥ seuil":
    key = "_days_since_move"
elif dead_by == "Jours depuis entrée ≥ seuil":
    key = "_days_since_in"
else:
    key = None

if key:
    s = df[key]
    if treat_nan_as_dead:
        s = s.fillna(10**9)  # NaN = considéré dead
    df_dead = df[s >= threshold].copy()
else:
    aging_col = cols["aging_text"]
    if aging_col and aging_col in df.columns:
        text = df[aging_col].astype(str).str.lower()
        mask_txt = text.str.contains("year|an|ans|mois|month", regex=True, na=False)
        s = df["_days_since_move"]
        if treat_nan_as_dead:
            s = s.fillna(10**9)
        mask_days = s >= threshold
        df_dead = df[mask_txt | mask_days].copy()
    else:
        s = df["_days_since_move"]
        if treat_nan_as_dead:
            s = s.fillna(10**9)
        df_dead = df[s >= threshold].copy()

# Option secours: voir tout
see_all = st.sidebar.checkbox("👀 Voir tout le stock (ignorer le filtre dead)", value=False)
if see_all:
    df_dead = df.copy()

# Si vide, afficher diagnostic et sortir proprement
if df_dead.empty:
    st.warning("Aucune ligne ne répond à la définition du dead stock et au seuil actuel.")
    st.info("👉 Astuces : baisse le seuil, active l’option dates manquantes, et vérifie le mapping des dates.")
    with st.expander("Voir un aperçu du stock brut (non filtré)"):
        _preview_cols = [c for c in [cols.get("item_code"), cols.get("item_desc"), cols.get("category"),
                                     cols.get("affiliate"), cols.get("quantity"), cols.get("total_cost_usd"),
                                     "_days_since_in", "_days_since_move"] if c]
        st.dataframe(df[_preview_cols].head(50), use_container_width=True)
    st.stop()

# ---------- KPI
def nuniq(s):
    try: return s.nunique()
    except: return np.nan

sku_col = cols["item_code"] if cols["item_code"] in df.columns else None
cat_col = cols["category"] if cols["category"] in df.columns else None
desc_col = cols["item_desc"] if cols["item_desc"] in df.columns else None

qty_sum = df_dead[qty_col].sum() if qty_col else np.nan
val_sum = df_dead["_value_calc"].sum(min_count=1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Valeur dead stock", f"{val_sum:,.0f}")
c2.metric("Quantité dead stock", f"{qty_sum:,.0f}" if not pd.isna(qty_sum) else "N/A")
c3.metric("# SKU dead stock", f"{nuniq(df_dead[sku_col]):,}" if sku_col is not None else "N/A")
c4.metric("Âge moyen (depuis entrée)", f"{df_dead['_days_since_in'].mean():.0f} j")
c5.metric("Jours sans mouvement (moyen)", f"{df_dead['_days_since_move'].mean():.0f} j")

pct_over_2y = (df_dead["_days_since_move"] >= 730).mean()*100
st.metric("% lignes sans mouvement > 2 ans", f"{pct_over_2y:.1f}%")

# ---------- Répartition par âge
st.markdown("### 📊 Répartition par tranches d’âge (sans mouvement)")
age_dist = df_dead["_bucket_since_move"].value_counts(dropna=False).rename_axis("Tranche").reset_index(name="Lignes")
age_fig = px.bar(age_dist.sort_values("Tranche"), x="Tranche", y="Lignes", text="Lignes", title="Distribution des âges (jours sans mouvement)")
st.plotly_chart(age_fig, use_container_width=True)

# ---------- Carte par pays (AFRIQUE)
st.markdown("### 🗺️ Carte des KPI par pays")
grp = df_dead.groupby("_country", dropna=False)
agg_dict = {"_value_calc":"sum", "_days_since_move":"mean"}
if qty_col: agg_dict[qty_col] = "sum"
if sku_col: agg_dict[sku_col] = pd.Series.nunique
by_country = grp.agg(agg_dict).reset_index()

rename_map = {"_country":"Pays", "_value_calc":"valeur", "_days_since_move":"age_moy"}
if qty_col: rename_map[qty_col] = "quantite"
if sku_col: rename_map[sku_col] = "skus"
by_country = by_country.rename(columns=rename_map)
by_country["Pays"] = by_country["Pays"].fillna("Inconnu")

soc_col = aff_col
if soc_col:
    soc_lut = (df_dead[[soc_col, "_country"]]
               .dropna(subset=["_country"])
               .groupby("_country")[soc_col]
               .agg(lambda s: ", ".join(sorted(pd.unique([str(x).strip() for x in s if pd.notna(x)])))).rename("Sociétés"))
    by_country = by_country.merge(soc_lut, left_on="Pays", right_index=True, how="left")
else:
    by_country["Sociétés"] = None

map_df = by_country[by_country["Pays"]!="Inconnu"].copy()
if not map_df.empty:
    show_labels = st.checkbox("Afficher les noms des sociétés sur la carte", value=True)
    fig_map = px.choropleth(
        map_df,
        locations="Pays",
        locationmode="country names",
        color="valeur",
        hover_name="Pays",
        hover_data={
            "valeur":":,.0f",
            "quantite":":,.0f" if "quantite" in map_df.columns else True,
            "skus":True if "skus" in map_df.columns else False,
            "age_moy":":.0f",
            "Sociétés":True
        },
        color_continuous_scale="YlOrRd",
        title="Valeur dead stock par pays (Afrique)"
    )
    fig_map.update_geos(
        scope="africa",
        projection_type="natural earth",
        showcountries=True, countrycolor="#FFFFFF",
        showsubunits=True, showcoastlines=False,
        fitbounds="locations", lataxis_range=[-35, 38], lonaxis_range=[-25, 60]
    )
    fig_map.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=50, b=0),
                          coloraxis_colorbar=dict(title="Valeur"))
    if show_labels:
        label_df = map_df.copy()
        def _shorten(txt, max_chars=60):
            if not isinstance(txt, str) or not txt:
                return ""
            return (txt[:max_chars] + "…") if len(txt) > max_chars else txt
        label_df["__label"] = label_df.apply(
            lambda r: f"{r['Pays']}\n{_shorten(r['Sociétés'] or '', 60)}\n{r['valeur']:,.0f}", axis=1
        )
        fig_pts = px.scatter_geo(label_df, locations="Pays", locationmode="country names", text="__label")
        fig_pts.update_traces(mode="text", textposition="middle center",
                              textfont=dict(size=10, color="#1f2937"),
                              hoverinfo="skip", showlegend=False)
        for tr in fig_pts.data:
            fig_map.add_trace(tr)
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Aucun pays reconnu automatiquement.")

with st.expander("🔎 Détail par pays"):
    st.dataframe(by_country.sort_values("valeur", ascending=False), use_container_width=True)

# ---------- Tops & Pivots
st.markdown("### 🥇 Top catégories et Top items")
if cat_col:
    top_cat = (df_dead.groupby(cat_col, dropna=False)["_value_calc"]
               .sum().reset_index().sort_values("_value_calc", ascending=False).head(20))
    top_cat_fig = px.bar(top_cat, x=cat_col, y="_value_calc", title="Top catégories par valeur", text_auto=".0f")
    st.plotly_chart(top_cat_fig, use_container_width=True)
    with st.expander("Voir le tableau catégories"):
        st.dataframe(top_cat, use_container_width=True)

top_items = pd.DataFrame()
if sku_col:
    base_cols = [c for c in [sku_col, desc_col, cat_col, aff_col, qty_col, unit_col, usd_col] if c and c in df_dead.columns]
    top_items = (df_dead[base_cols + ["_value_calc","_days_since_move","_days_since_in"]]
                 .groupby([sku_col] + ([desc_col] if desc_col else []), dropna=False)
                 .agg({"_value_calc":"sum", "_days_since_move":"mean", "_days_since_in":"mean"})
                 .reset_index().sort_values("_value_calc", ascending=False).head(25))
    top_items = top_items.rename(columns={"_value_calc":"Valeur","_days_since_move":"J sans mouv (moyen)","_days_since_in":"J depuis entrée (moyen)"})
    st.dataframe(top_items, use_container_width=True)

# ---------- ÉVOLUTION HEBDOMADAIRE
st.markdown("## 📈 Évolution hebdomadaire (Status date)")
if status_col and df_dead["_status_week_start"].notna().any():
    only_2025 = st.checkbox("Filtrer sur 2025 uniquement", value=True)
    trend_base = df_dead.copy()
    if only_2025:
        trend_base = trend_base[trend_base["_status_year"] == 2025]

    trend = (trend_base
             .dropna(subset=["_status_week_start"])
             .groupby(["_status_week_start","_status_label"])
             .agg(valeur=("_value_calc","sum"),
                  quantite=(qty_col,"sum") if qty_col else ("_value_calc","size"),
                  skus=(sku_col, pd.Series.nunique) if sku_col else ("_value_calc","size"))
             .reset_index()
             .sort_values("_status_week_start"))

    if trend.empty:
        st.info("Pas assez de données de semaines pour tracer la tendance.")
    else:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend["_status_week_start"], y=trend["valeur"],
                                       mode="lines+markers", name="Valeur"))
        if qty_col:
            fig_trend.add_trace(go.Scatter(x=trend["_status_week_start"], y=trend["quantite"],
                                           mode="lines+markers", name="Quantité", yaxis="y2"))
            fig_trend.update_layout(
                title="Évolution hebdo — Valeur & Quantité",
                xaxis_title="Semaine",
                yaxis_title="Valeur",
                yaxis2=dict(title="Quantité", overlaying="y", side="right"),
                template="plotly_white"
            )
        else:
            fig_trend.update_layout(title="Évolution hebdo — Valeur",
                                    xaxis_title="Semaine", yaxis_title="Valeur",
                                    template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)

        # KPI Delta vs semaine précédente
        last_row = trend.iloc[-1]
        prev_row = trend.iloc[-2] if len(trend) >= 2 else None
        kc1, kc2, kc3 = st.columns(3)
        if prev_row is not None:
            dv = last_row["valeur"] - prev_row["valeur"]
            kc1.metric(f"Δ Valeur (vs {prev_row['_status_label']})", f"{dv:,.0f}",
                       delta=f"{(dv/prev_row['valeur']*100):+.1f}%" if prev_row["valeur"] else None)
            if qty_col:
                dq = last_row["quantite"] - prev_row["quantite"]
                kc2.metric(f"Δ Quantité (vs {prev_row['_status_label']})", f"{dq:,.0f}")
            if sku_col:
                ds = last_row["skus"] - prev_row["skus"]
                kc3.metric(f"Δ #SKU (vs {prev_row['_status_label']})", f"{ds:+,}")
        else:
            kc1.info("Une seule semaine disponible : pas de delta.")

        # Comparateur de semaines
        st.markdown("#### 🔁 Comparer deux semaines")
        weeks = list(trend["_status_label"])
        a = st.selectbox("Semaine A", options=weeks, index=max(0, len(weeks)-2))
        b = st.selectbox("Semaine B", options=weeks, index=len(weeks)-1)
        A = trend[trend["_status_label"]==a].iloc[0]
        B = trend[trend["_status_label"]==b].iloc[0]
        dv2 = B["valeur"] - A["valeur"]
        txt = "🟢 On a avancé (valeur en baisse)" if dv2 < 0 else ("🔴 On a régressé (valeur en hausse)" if dv2 > 0 else "⚪ Stable")
        st.markdown(f"**Résultat : {txt}** — Δ Valeur = {dv2:,.0f}")
        if qty_col:
            dq2 = B["quantite"] - A["quantite"]
            st.caption(f"Δ Quantité = {dq2:,.0f}")
        if sku_col:
            ds2 = B["skus"] - A["skus"]
            st.caption(f"Δ #SKU = {ds2:+,}")
else:
    st.info("Aucune information de semaine ('Status date') exploitable.")

# ---------- RECOMMANDATIONS IA
st.markdown("## 🤖 Recommandations IA (items à vendre en priorité)")
if df_dead.empty:
    st.info("Aucune ligne de dead stock selon tes critères.")
    recos = pd.DataFrame()
else:
    tmp = df_dead.copy()

    # Scores normalisés (robustes aux extrêmes)
    val = tmp["_value_calc"].fillna(0)
    v99 = max(val.quantile(0.99), 1) if len(val) else 1
    tmp["_val_score"] = (val / v99).clip(0,1)

    if qty_col:
        q = tmp[qty_col].fillna(0)
        q90 = max(q.quantile(0.90), 1) if len(q) else 1
        tmp["_qty_score"] = (q / q90).clip(0,1)
    else:
        tmp["_qty_score"] = 0

    age = tmp["_days_since_move"].fillna(0)
    tmp["_age_score"] = (age / 730).clip(0,1)  # cap à 2 ans

    tmp["_sell_score"] = 0.5*tmp["_age_score"] + 0.3*tmp["_val_score"] + 0.2*tmp["_qty_score"]

    def action_ia(r):
        if (r["_days_since_move"] or 0) >= 1825 and (r["_value_calc"] or 0) < (v99*0.05):
            return "Mise au rebut (obsolète, faible valeur)"
        if r["_sell_score"] >= 0.75:
            return "Vendre en priorité (clearance / remise)"
        if r["_sell_score"] >= 0.50:
            return "Transfert interne / bundle / promo"
        return "Suivi standard"

    tmp["Action IA"] = tmp.apply(action_ia, axis=1)
    tmp["Priorité"] = pd.cut(tmp["_sell_score"],
                             bins=[-1,0.5,0.75,1.01],
                             labels=["C - Standard","B - Moyen","A - Urgent"])

    cols_show = [c for c in [aff_col, sku_col, desc_col, cat_col, qty_col, unit_col, usd_col] if c]
    recos = (tmp[cols_show + ["_value_calc","_days_since_move","_days_since_in","Priorité","Action IA","_sell_score"]]
             .rename(columns={
                 "_value_calc":"Valeur (calc.)",
                 "_days_since_move":"J sans mouv.",
                 "_days_since_in":"J depuis entrée",
                 "_sell_score":"Score vente"
             })
             .sort_values(["Priorité","Score vente"], ascending=[True, False]))

    # Filtres sur les recos
    fc1, fc2 = st.columns(2)
    if aff_col:
        affiliates = ["(Toutes)"] + sorted([x for x in pd.unique(df_dead[aff_col].dropna())])
        sel_aff = fc1.selectbox("Filiale", options=affiliates, index=0)
        if sel_aff != "(Toutes)":
            recos = recos[recos[aff_col]==sel_aff]
    if cat_col:
        cats = ["(Toutes)"] + sorted([x for x in pd.unique(df_dead[cat_col].dropna())])
        sel_cat = fc2.selectbox("Catégorie", options=cats, index=0)
        if sel_cat != "(Toutes)":
            recos = recos[recos[cat_col]==sel_cat]

    st.dataframe(recos.head(200), use_container_width=True)

# ---------- Export Excel enrichi
st.markdown("### ⬇️ Télécharger les résultats")
df_dead_export = df_dead.copy()
df_dead_export["Pays (déduit)"] = df_dead_export["_country"]
df_dead_export["Jours depuis entrée"] = df_dead_export["_days_since_in"]
df_dead_export["Jours sans mouvement"] = df_dead_export["_days_since_move"]
df_dead_export["Valeur (calculée)"] = df_dead_export["_value_calc"]
df_dead_export["Tranche âge (entrée)"] = df_dead_export["_bucket_since_in"]
df_dead_export["Tranche âge (sans mouvement)"] = df_dead_export["_bucket_since_move"]
if status_col:
    df_dead_export["Semaine (label)"] = df_dead_export["_status_label"]
    df_dead_export["Semaine (lundi)"] = df_dead_export["_status_week_start"]

sheets = {
    "Overview": pd.DataFrame({
        "KPI": [
            "Valeur dead stock",
            "Quantité dead stock",
            "SKU dead stock",
            "Âge moyen (depuis entrée, j)",
            "Jours sans mouvement (moyen)",
            "% lignes > 2 ans sans mouvement"
        ],
        "Valeur": [
            val_sum,
            qty_sum,
            nuniq(df_dead[sku_col]) if sku_col is not None else np.nan,
            df_dead["_days_since_in"].mean(),
            df_dead["_days_since_move"].mean(),
            pct_over_2y
        ]
    }),
    "By Country": by_country,
    "By Category": (df_dead.groupby(cat_col, dropna=False)["_value_calc"].sum().reset_index().rename(columns={"_value_calc":"Valeur"})
                    if cat_col else pd.DataFrame()),
    "Top Items": top_items if not top_items.empty else pd.DataFrame(),
    "Dead Lines": df_dead_export
}

# Tendance export
if status_col and df_dead["_status_week_start"].notna().any():
    trend_export = (df_dead.dropna(subset=["_status_week_start"])
                    .groupby(["_status_week_start","_status_label"])
                    .agg(Valeur=("_value_calc","sum"),
                         Quantite=(qty_col,"sum") if qty_col else ("_value_calc","size"),
                         SKUs=(sku_col, pd.Series.nunique) if sku_col else ("_value_calc","size"))
                    .reset_index()
                    .sort_values("_status_week_start"))
    sheets["Trend by Week"] = trend_export

# Recos IA export
if 'recos' in locals() and isinstance(recos, pd.DataFrame) and not recos.empty:
    sheets["IA Actions"] = recos

excel_bytes = to_excel_bytes(sheets)
st.download_button(
    "Télécharger l’Excel enrichi",
    data=excel_bytes,
    file_name="dead_stock_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Astuce : coche 'Combiner plusieurs feuilles' pour agréger GE/SP/Wxx. Active aussi l’option dates manquantes si besoin.")
