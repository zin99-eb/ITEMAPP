import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# 0) CONFIG
# ============================================================
st.set_page_config(page_title="🔎 PO Duplicates Detector", layout="wide")
st.title("🔎 Détection des PO dupliqués / suspects")

# Fonction export Excel
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()

# ============================================================
# 1) UPLOAD
# ============================================================
uploaded = st.file_uploader("📎 Upload PO file (Excel ou CSV)", type=["xlsx", "xls", "csv"])

if uploaded is None:
    st.info("➡️ Charge ton fichier PO pour commencer.")
    st.stop()

# Lecture fichier sécurisé
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str)
    else:
        df = pd.read_excel(uploaded, dtype=str)
except Exception as e:
    st.error(f"Erreur lecture fichier : {e}")
    st.stop()

# ============================================================
# 2) DÉTECTION COLONNE MONTANT
# ============================================================
montant_col = None
for c in df.columns:
    if "Total_PO_AmountInUSD" in c:
        montant_col = c
    elif "PrixTotalUSDHT" in c:
        montant_col = c

if montant_col is None:
    st.error("❌ Aucune colonne montant trouvée ('PrixTotalUSDHT' ou 'Total_PO_AmountInUSD').")
    st.write("Colonnes présentes :", df.columns.tolist())
    st.stop()

df = df.rename(columns={
    "name": "Description",
    montant_col: "Montant"
})

required = ["created_at", "PO reference", "Description", "Supplier", "Company", "Montant"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"❌ Colonnes manquantes : {missing}")
    st.stop()

# ============================================================
# 3) NETTOYAGE
# ============================================================
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Nettoyage du montant (2 964,18 → 2964.18)
df["Montant"] = (
    df["Montant"]
    .astype(str)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce")

df = df.dropna(subset=["Montant"])

# Normalisation texte
df["desc_norm"] = (
    df["Description"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z0-9 ]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["supplier_norm"] = df["Supplier"].str.lower().str.strip()

# ============================================================
# 4) FILTRE PAR PAYS / FILIALE
# ============================================================
st.sidebar.header("Filtres")

liste_filiales = sorted(df["Company"].dropna().unique().tolist())

filiale_sel = st.sidebar.multiselect(
    "🌍 Filiale / Country",
    liste_filiales,
    default=liste_filiales
)

df = df[df["Company"].isin(filiale_sel)]

if df.empty:
    st.warning("Aucune donnée pour les pays sélectionnés.")
    st.stop()

# ============================================================
# 5) DÉTECTION PO DUPLIQUÉS
# ============================================================
duplicates = (
    df.groupby(["supplier_norm", "Montant", "desc_norm"])
    .agg(
        nb_po=("PO reference", "nunique"),
        po_list=("PO reference", lambda x: ", ".join(sorted(x.unique()))),
        companies=("Company", lambda x: ", ".join(sorted(x.unique()))),
        dates=("created_at", lambda x: ", ".join(sorted(pd.to_datetime(x).dt.strftime("%Y-%m-%d").unique()))),
        description=("Description", "first")
    )
    .reset_index()
)

suspicious = duplicates[duplicates["nb_po"] > 1].sort_values("nb_po", ascending=False)

# ============================================================
# 6) AFFICHAGE TABLEAU DES DOUBLONS
# ============================================================
st.header("🛑 POs potentiellement dupliqués")

if suspicious.empty:
    st.success("🎉 Aucun PO dupliqué détecté.")
else:
    st.warning("⚠️ Doublons détectés (même supplier + même montant + description similaire).")
    st.dataframe(suspicious, use_container_width=True)

    # Export CSV
    st.download_button(
        "⬇️ Télécharger (CSV)",
        suspicious.to_csv(index=False),
        "PO_duplicates.csv",
        mime="text/csv"
    )

    # Export Excel
    st.download_button(
        "⬇️ Télécharger (Excel .xlsx)",
        to_excel(suspicious),
        "PO_duplicates.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# 7) DÉTAIL LIGNES SOURCES
# ============================================================
st.markdown("---")
st.subheader("🔍 Détails des PO dupliqués")

if not suspicious.empty:
    merged = df.merge(
        suspicious[["supplier_norm", "Montant", "desc_norm"]],
        on=["supplier_norm", "Montant", "desc_norm"],
        how="inner"
    ).sort_values("created_at")

    st.dataframe(
        merged[["created_at", "PO reference", "Supplier", "Company", "Description", "Montant"]],
        use_container_width=True
    )

    # Export détails : CSV
    st.download_button(
        "⬇️ Télécharger détails (CSV)",
        merged.to_csv(index=False),
        "PO_duplicates_details.csv",
        mime="text/csv"
    )

    # Export détails : Excel
    st.download_button(
        "⬇️ Télécharger détails (Excel .xlsx)",
        to_excel(merged),
        "PO_duplicates_details.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )