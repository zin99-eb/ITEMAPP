# ============================================================
# app.py — Détection des POs en doublons V4.4
# Avec détection intelligente des POs récurrents vs vrais doublons
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from difflib import SequenceMatcher
from io import BytesIO, StringIO
import hashlib
import re

# ==============================
# CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="Détection POs en doublons", page_icon="🔍", layout="wide")
st.title("🎯 Détection des POs en doublons — V4.4 (POs récurrents vs Doublons)")

# ==============================
# FONCTIONS D'ANALYSE DES NOMS
# ==============================
def detect_recurrent_po(name1, name2):
    """
    Détecte si deux POs sont des récurrences (même service, mois différent)
    Retourne True si ce sont des POs récurrents, False si potentiel doublon
    """
    if not isinstance(name1, str) or not isinstance(name2, str):
        return False
    
    name1_lower = name1.lower()
    name2_lower = name2.lower()
    
    # Liste des motifs de mois (français et anglais)
    mois_fr = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
               'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    mois_en = ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december']
    
    # Chercher des mois dans les noms
    mois_trouves_1 = [mois for mois in mois_fr + mois_en if mois in name1_lower]
    mois_trouves_2 = [mois for mois in mois_fr + mois_en if mois in name2_lower]
    
    # Si les deux ont des mois différents, c'est probablement une récurrence
    if mois_trouves_1 and mois_trouves_2:
        if mois_trouves_1[0] != mois_trouves_2[0]:
            return True
    
    # Chercher des patterns de date (MOIS DE XXX)
    pattern_mois = r'(mois de|mois d\'|month of|for)\s+([a-zA-Zéèêëàâäôöûüç]+)'
    mois_match_1 = re.search(pattern_mois, name1_lower)
    mois_match_2 = re.search(pattern_mois, name2_lower)
    
    if mois_match_1 and mois_match_2:
        mois_1 = mois_match_1.group(2)
        mois_2 = mois_match_2.group(2)
        if mois_1 != mois_2:
            return True
    
    # Chercher des numéros de semaine ou période
    pattern_periode = r'(semaine|week|période|period)\s+(\d+)'
    periode_match_1 = re.search(pattern_periode, name1_lower)
    periode_match_2 = re.search(pattern_periode, name2_lower)
    
    if periode_match_1 and periode_match_2:
        periode_1 = periode_match_1.group(2)
        periode_2 = periode_match_2.group(2)
        if periode_1 != periode_2:
            return True
    
    return False

def extract_common_base(name1, name2):
    """Extrait la partie commune de deux noms (sans les différences de mois/date)"""
    if not isinstance(name1, str) or not isinstance(name2, str):
        return "", ""
    
    # Enlever les parties de date/mois
    patterns_a_enlever = [
        r'mois de [a-zA-Zéèêëàâäôöûüç]+',
        r'mois d\'[a-zA-Zéèêëàâäôöûüç]+',
        r'month of [a-zA-Z]+',
        r'for [a-zA-Z]+',
        r'semaine \d+',
        r'week \d+',
        r'période \d+',
        r'period \d+',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}',
        r'janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre',
        r'january|february|march|april|may|june|july|august|september|october|november|december'
    ]
    
    base1 = name1.lower()
    base2 = name2.lower()
    
    for pattern in patterns_a_enlever:
        base1 = re.sub(pattern, '', base1, flags=re.IGNORECASE)
        base2 = re.sub(pattern, '', base2, flags=re.IGNORECASE)
    
    # Nettoyer les espaces multiples
    base1 = ' '.join(base1.split())
    base2 = ' '.join(base2.split())
    
    return base1, base2

def calculate_smart_similarity(name1, name2, base1, base2):
    """Calcule une similarité intelligente qui ignore les différences de date"""
    if not base1 or not base2:
        return SequenceMatcher(None, str(name1).lower(), str(name2).lower()).ratio()
    
    # Similarité sur la base commune
    base_similarity = SequenceMatcher(None, base1, base2).ratio()
    
    # Bonus si les bases sont très similaires (ce sont les mêmes services)
    if base_similarity > 0.9:
        return min(1.0, base_similarity + 0.2)  # Bonus pour service identique
    else:
        return base_similarity

# ==============================
# FONCTIONS D'EXPORT EXCEL
# ==============================
def export_to_excel(dataframes_dict, sheet_names_dict):
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            if df is not None and not df.empty:
                display_name = sheet_names_dict.get(sheet_name, sheet_name)
                df.to_excel(writer, sheet_name=display_name[:31], index=False)
    
    return output.getvalue()

# ==============================
# NORMALISATION COLONNES
# ==============================
PRICE_MAP = {
    "prixtotalusd": "PrixTotalUSD",
    "prixtotalusdht": "PrixTotalUSD",
    "prix_total_usd": "PrixTotalUSD",
    "prix_total_usdht": "PrixTotalUSD",
    "prix total usd": "PrixTotalUSD",
    "prix total": "PrixTotalUSD",
    "montant": "PrixTotalUSD",
    "montanttotal": "PrixTotalUSD",
    "montant_total": "PrixTotalUSD",
    "total": "PrixTotalUSD",
    "amount": "PrixTotalUSD",
    "amountusd": "PrixTotalUSD"
}
SUPP_MAP = {
    "supplier": "Supplier",
    "fournisseur": "Supplier",
    "vendor": "Supplier",
    "vendor name": "Supplier"
}
NAME_MAP = {
    "name": "name",
    "nom": "name",
    "description": "name",
    "po_name": "name",
    "po description": "name"
}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        low = c.lower().strip()
        comp = low.replace(" ", "").replace("_", "")
        if comp in PRICE_MAP:
            mapping[c] = PRICE_MAP[comp]
        if low in SUPP_MAP:
            mapping[c] = SUPP_MAP[low]
        if low in NAME_MAP:
            mapping[c] = NAME_MAP[low]
    df = df.rename(columns=mapping)
    if "PrixTotalUSD" in df.columns:
        df["PrixTotalUSD"] = pd.to_numeric(df["PrixTotalUSD"], errors="coerce")
    return df

# ==============================
# UTILS RAPIDES
# ==============================
def clean_text_fast(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return ' '.join(s.lower().translate(str.maketrans('', '', '/.,;:-_()[]{}|+*\'\"')).split())

# ==============================
# LECTURE FICHIER
# ==============================
@st.cache_data(show_spinner=False)
def read_file(bytes_file: bytes, ext: str) -> pd.DataFrame:
    bio = BytesIO(bytes_file)
    if ext == "csv":
        return pd.read_csv(bio, low_memory=False)
    return pd.read_excel(bio, engine="openpyxl")

# ==============================
# DÉTECTION INTELLIGENTE
# ==============================
def detect_doublons_intelligent(df: pd.DataFrame,
                               date_window: int,
                               use_supplier: bool,
                               use_amount: bool,
                               detect_recurrent: bool,
                               strict_mode: bool) -> pd.DataFrame:
    """
    Détection intelligente qui distingue vrais doublons vs POs récurrents
    """
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work = work.dropna(subset=["created_at", "PO reference"])
    work = work.sort_values("created_at")
    
    if "name" in work.columns:
        work["name_clean"] = work["name"].apply(lambda x: clean_text_fast(str(x)))
    
    work = work[work["status"] == "Validated"]
    
    has_price = "PrixTotalUSD" in work.columns
    has_supp = "Supplier" in work.columns
    has_name = "name" in work.columns
    
    results = []
    
    # Grouper par company d'abord
    for company, company_data in work.groupby("Company"):
        if len(company_data) <= 1:
            continue
        
        company_data = company_data.sort_values("created_at").reset_index(drop=True)
        n = len(company_data)
        
        # Vérifier les POs avec exactement la même référence
        po_dict = {}
        for idx, row in company_data.iterrows():
            po = str(row["PO reference"]).strip()
            if po:
                po_dict.setdefault(po, []).append(idx)
        
        # Doublons exacts (même référence PO)
        for po, indices in po_dict.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        r1, r2 = company_data.iloc[idx1], company_data.iloc[idx2]
                        
                        if abs((r1["created_at"] - r2["created_at"]).days) <= date_window:
                            results.append({
                                "Company": company,
                                "PO1": po,
                                "PO2": po,
                                "Nom1": r1.get("name", ""),
                                "Nom2": r2.get("name", ""),
                                "Supplier1": r1.get("Supplier", ""),
                                "Supplier2": r2.get("Supplier", ""),
                                "Montant1": r1.get("PrixTotalUSD", np.nan),
                                "Montant2": r2.get("PrixTotalUSD", np.nan),
                                "Date1": r1["created_at"],
                                "Date2": r2["created_at"],
                                "Score": 1.0,
                                "Raisons": "Même référence PO exacte",
                                "Type": "DOUBLON EXACT",
                                "Categorie": "DOUBLON REEL"
                            })
        
        # Comparaison paire par paire dans la fenêtre temporelle
        for i in range(n):
            po1 = str(company_data.iloc[i]["PO reference"]).strip()
            if not po1:
                continue
                
            date1 = company_data.iloc[i]["created_at"]
            
            for j in range(i + 1, n):
                date2 = company_data.iloc[j]["created_at"]
                
                if (date2 - date1).days > date_window:
                    break
                
                po2 = str(company_data.iloc[j]["PO reference"]).strip()
                if not po2 or po1 == po2:
                    continue
                
                r1, r2 = company_data.iloc[i], company_data.iloc[j]
                
                # Vérifier si strict mode est activé
                if strict_mode:
                    # En mode strict, on ne détecte que les doublons parfaits
                    if not (has_supp and has_price):
                        continue
                    
                    supp1 = str(r1.get("Supplier", "")).strip()
                    supp2 = str(r2.get("Supplier", "")).strip()
                    
                    try:
                        amt1 = float(r1.get("PrixTotalUSD", 0))
                        amt2 = float(r2.get("PrixTotalUSD", 0))
                    except (ValueError, TypeError):
                        continue
                    
                    if supp1 == supp2 and abs(amt1 - amt2) < 0.01:
                        # C'est un doublon parfait
                        type_po = "DOUBLON PARFAIT"
                        categorie = "DOUBLON REEL"
                        score = 1.0
                        raisons = "Même fournisseur + même montant exact"
                    else:
                        continue  # Passer au suivant en mode strict
                else:
                    # Mode normal avec analyse intelligente
                    score, raisons = 0.0, []
                    type_po = "DOUBLON POTENTIEL"
                    categorie = "A ANALYSER"
                    
                    # Vérifier le même fournisseur
                    if has_supp and use_supplier:
                        supp1 = str(r1.get("Supplier", "")).strip()
                        supp2 = str(r2.get("Supplier", "")).strip()
                        if supp1 and supp2 and supp1 == supp2:
                            score += 0.7
                            raisons.append("Même fournisseur")
                    
                    # Vérifier le même montant
                    if has_price and use_amount:
                        try:
                            amt1 = float(r1.get("PrixTotalUSD", 0))
                            amt2 = float(r2.get("PrixTotalUSD", 0))
                            if abs(amt1 - amt2) < 0.01:
                                score += 0.6
                                raisons.append("Même montant")
                        except (ValueError, TypeError):
                            pass
                    
                    # Vérifier si ce sont des POs récurrents
                    if has_name and detect_recurrent:
                        name1 = r1.get("name", "")
                        name2 = r2.get("name", "")
                        
                        if detect_recurrent_po(name1, name2):
                            # Ce sont des POs récurrents, pas des doublons
                            base1, base2 = extract_common_base(name1, name2)
                            smart_sim = calculate_smart_similarity(name1, name2, base1, base2)
                            
                            if smart_sim > 0.8:
                                type_po = "PO RECURRENT"
                                categorie = "NORMAL"
                                score = 0.3  # Score bas car ce n'est pas un doublon
                                raisons.append(f"Service récurrent (similarité base: {smart_sim:.2f})")
                    
                    # Doublon parfait (fournisseur + montant)
                    if (has_supp and has_price and use_supplier and use_amount and
                        "Même fournisseur" in raisons and "Même montant" in raisons):
                        type_po = "DOUBLON PARFAIT"
                        categorie = "DOUBLON REEL"
                        score = 1.0
                        raisons = ["Même fournisseur + même montant exact (DOUBLON REEL)"]
                
                # Déterminer si on ajoute aux résultats
                ajouter = False
                
                if strict_mode:
                    ajouter = True  # En mode strict, on a déjà filtré
                elif type_po == "DOUBLON REEL":
                    ajouter = True
                elif type_po == "DOUBLON PARFAIT":
                    ajouter = True
                elif score >= 0.8 and type_po != "PO RECURRENT":
                    ajouter = True
                elif type_po == "PO RECURRENT" and st.session_state.get('show_recurrent', False):
                    ajouter = True  # Seulement si l'utilisateur veut voir les récurrents
                
                if ajouter:
                    results.append({
                        "Company": company,
                        "PO1": po1,
                        "PO2": po2,
                        "Nom1": r1.get("name", ""),
                        "Nom2": r2.get("name", ""),
                        "Supplier1": r1.get("Supplier", ""),
                        "Supplier2": r2.get("Supplier", ""),
                        "Montant1": r1.get("PrixTotalUSD", np.nan),
                        "Montant2": r2.get("PrixTotalUSD", np.nan),
                        "Date1": date1,
                        "Date2": date2,
                        "Score": round(score, 3),
                        "Raisons": ", ".join(raisons),
                        "Type": type_po,
                        "Categorie": categorie
                    })
    
    return pd.DataFrame(results)

# ==============================
# UI : UPLOAD
# ==============================
uploaded = st.file_uploader("📥 Import POs (CSV/XLSX)")
if not uploaded:
    st.info("Colonnes minimales : PO reference, Company, status, created_at (optionnel : name, Supplier, PrixTotalUSD).")
    st.stop()

ext = uploaded.name.split(".")[-1].lower()
df = read_file(uploaded.getvalue(), ext)
df = normalize_cols(df)

st.success(f"Fichier chargé ✔ ({len(df):,} lignes)")
with st.expander("Aperçu des données"):
    st.dataframe(df.head(), use_container_width=True)

required = ["PO reference", "Company", "status", "created_at"]
if any(c not in df.columns for c in required):
    st.error(f"Colonnes obligatoires manquantes. Requises: {required}")
    st.stop()

# ==============================
# Options
# ==============================
st.subheader("⚙️ Options de détection")

col1, col2, col3 = st.columns(3)
companies = sorted(df["Company"].astype(str).unique().tolist())
companies_sel = col1.multiselect("Filtrer Company", companies, default=companies[:min(5, len(companies))])
date_window = col2.slider("Fenêtre temporelle (jours)", 1, 90, 30)

detection_mode = col3.selectbox(
    "Mode de détection",
    ["Intelligent (recommandé)", "Strict (doublons parfaits seulement)", "Large (tous les suspects)"],
    index=0
)

st.markdown("**Critères de détection:**")
col4, col5, col6, col7 = st.columns(4)

use_supplier = col4.checkbox("Même fournisseur", value=True)
use_amount = col5.checkbox("Même montant", value=True)
detect_recurrent = col6.checkbox("Détecter POs récurrents", value=True)

# Initialiser session state pour l'option d'affichage
if 'show_recurrent' not in st.session_state:
    st.session_state.show_recurrent = False

show_recurrent = col7.checkbox(
    "Afficher POs récurrents", 
    value=st.session_state.show_recurrent,
    help="Afficher les POs qui sont des récurrences normales (même service, mois différent)"
)

# Mettre à jour le session state
st.session_state.show_recurrent = show_recurrent

# Mode strict
strict_mode = (detection_mode == "Strict (doublons parfaits seulement)")

# Filtrer par company
work = df[df["Company"].astype(str).isin(companies_sel)].copy()

# ==============================
# ANALYSE
# ==============================
st.subheader("🔍 Analyse des doublons")

if st.button("🚀 Démarrer l'analyse intelligente", type="primary"):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Analyse en cours..."):
        status_text.text("Préparation des données...")
        progress_bar.progress(20)
        
        doublons_df = detect_doublons_intelligent(
            work,
            date_window,
            use_supplier,
            use_amount,
            detect_recurrent,
            strict_mode
        )
        
        progress_bar.progress(80)
    
    # Filtrer selon l'option d'affichage des récurrents
    if not show_recurrent and not doublons_df.empty and "Categorie" in doublons_df.columns:
        doublons_df = doublons_df[doublons_df["Categorie"] != "NORMAL"]
    
    progress_bar.progress(100)
    status_text.text("✅ Analyse terminée !")
    
    # =======================
    # 📊 ANALYSE DES RÉSULTATS
    # =======================
    st.success("✅ Analyse terminée avec distinction des POs récurrents !")
    
    if not doublons_df.empty:
        # Statistiques par catégorie
        categories = doublons_df["Categorie"].value_counts()
        types = doublons_df["Type"].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_pairs = len(doublons_df)
        col1.metric("Total paires analysées", f"{total_pairs:,}")
        
        doublons_reels = categories.get("DOUBLON REEL", 0)
        col2.metric("🔴 Doublons réels", f"{doublons_reels:,}", 
                   delta_color="inverse" if doublons_reels > 0 else "off")
        
        a_analyser = categories.get("A ANALYSER", 0)
        col3.metric("🟡 À analyser", f"{a_analyser:,}")
        
        recurrents = categories.get("NORMAL", 0)
        col4.metric("🟢 POs récurrents", f"{recurrents:,}")
        
        # =======================
        # 📋 TABLEAUX PAR CATÉGORIE
        # =======================
        st.subheader("📋 Résultats par catégorie")
        
        # Onglets pour chaque catégorie
        tab_reel, tab_analyser, tab_recurrent = st.tabs([
            f"🔴 Doublons réels ({doublons_reels})",
            f"🟡 À analyser ({a_analyser})", 
            f"🟢 POs récurrents ({recurrents})"
        ])
        
        with tab_reel:
            if doublons_reels > 0:
                reel_df = doublons_df[doublons_df["Categorie"] == "DOUBLON REEL"]
                st.warning("""
                **⚠️ ATTENTION : DOUBLONS RÉELS DÉTECTÉS !**
                
                Ces POs ont soit la **même référence**, soit le **même fournisseur + même montant exact**.
                Il s'agit très probablement de véritables doublons qui nécessitent une action immédiate.
                """)
                
                # Afficher les colonnes importantes
                cols_to_show = ["Company", "PO1", "PO2", "Nom1", "Supplier1", "Montant1", "Date1", "Date2", "Raisons"]
                cols_to_show = [c for c in cols_to_show if c in reel_df.columns]
                
                st.dataframe(reel_df[cols_to_show].sort_values("Date1"), use_container_width=True)
                
                # Bouton d'export
                st.download_button(
                    label="📥 Exporter les doublons réels (CSV)",
                    data=reel_df.to_csv(index=False, sep=';').encode('utf-8'),
                    file_name="doublons_reels.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ Aucun doublon réel détecté !")
        
        with tab_analyser:
            if a_analyser > 0:
                analyser_df = doublons_df[doublons_df["Categorie"] == "A ANALYSER"]
                st.info("""
                **📝 POs À ANALYSER MANUELLEMENT**
                
                Ces POs présentent des similarités mais nécessitent une vérification humaine.
                Ils pourraient être :
                - De vrais doublons avec des différences mineures
                - Des commandes similaires mais légitimes
                - Des erreurs de saisie
                """)
                
                # Ajouter une colonne d'action recommandée
                analyser_df["Action recommandée"] = analyser_df.apply(
                    lambda row: "Vérifier avec le demandeur" if "Même fournisseur" in row["Raisons"] 
                    else "Contacter le service achats",
                    axis=1
                )
                
                cols_to_show = ["Company", "PO1", "PO2", "Nom1", "Supplier1", "Montant1", "Score", "Raisons", "Action recommandée"]
                cols_to_show = [c for c in cols_to_show if c in analyser_df.columns]
                
                st.dataframe(analyser_df[cols_to_show].sort_values("Score", ascending=False), 
                           use_container_width=True)
                
                # Bouton d'export
                st.download_button(
                    label="📥 Exporter les POs à analyser (CSV)",
                    data=analyser_df.to_csv(index=False, sep=';').encode('utf-8'),
                    file_name="pos_a_analyser.csv",
                    mime="text/csv"
                )
            else:
                st.info("✅ Aucun PO nécessitant une analyse manuelle.")
        
        with tab_recurrent:
            if recurrents > 0:
                recurrent_df = doublons_df[doublons_df["Categorie"] == "NORMAL"]
                st.success("""
                **🔄 POs RÉCURRENTS (NORMALS)**
                
                Ces POs sont pour le même service mais à des périodes différentes (ex: janvier, février).
                Il s'agit de commandes régulières normales, PAS de doublons.
                
                **Exemples typiques :**
                - Gardiennage mensuel
                - Abonnements
                - Services récurrents
                - Maintenance régulière
                """)
                
                # Analyser les patterns récurrents
                if "Nom1" in recurrent_df.columns:
                    st.markdown("**🔍 Patterns récurrents détectés :**")
                    
                    # Extraire les bases communes
                    patterns = []
                    for _, row in recurrent_df.iterrows():
                        base1, base2 = extract_common_base(row["Nom1"], row["Nom2"])
                        if base1 and base1 not in patterns:
                            patterns.append(base1)
                    
                    for pattern in patterns[:5]:  # Limiter à 5 patterns
                        st.markdown(f"- `{pattern[:80]}...`")
                    
                    if len(patterns) > 5:
                        st.info(f"Et {len(patterns) - 5} autres patterns...")
                
                cols_to_show = ["Company", "PO1", "PO2", "Nom1", "Nom2", "Supplier1", "Montant1", "Date1", "Date2", "Raisons"]
                cols_to_show = [c for c in cols_to_show if c in recurrent_df.columns]
                
                st.dataframe(recurrent_df[cols_to_show].sort_values("Date1"), 
                           use_container_width=True)
                
                # Bouton d'export
                st.download_button(
                    label="📥 Exporter les POs récurrents (CSV)",
                    data=recurrent_df.to_csv(index=False, sep=';').encode('utf-8'),
                    file_name="pos_recurrents.csv",
                    mime="text/csv"
                )
            else:
                st.info("✅ Aucun PO récurrent détecté (ou masqué selon les options).")
        
        # =======================
        # 📈 STATISTIQUES DÉTAILLÉES
        # =======================
        st.subheader("📈 Statistiques détaillées")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("**📊 Par type de détection :**")
            for type_po, count in types.items():
                percentage = (count / total_pairs * 100)
                if type_po == "DOUBLON EXACT":
                    color = "🔴"
                elif type_po == "DOUBLON PARFAIT":
                    color = "🔴"
                elif type_po == "DOUBLON POTENTIEL":
                    color = "🟡"
                elif type_po == "PO RECURRENT":
                    color = "🟢"
                else:
                    color = "⚪"
                
                st.markdown(f"{color} **{type_po}** : {count} ({percentage:.1f}%)")
        
        with col_stat2:
            st.markdown("**🏢 Par filiale :**")
            company_stats = doublons_df.groupby("Company").agg({
                "PO1": "count",
                "Type": lambda x: (x == "DOUBLON REEL").sum() if "DOUBLON REEL" in x.values else 0
            }).rename(columns={"PO1": "Total", "Type": "Doublons réels"})
            
            for company, stats in company_stats.iterrows():
                st.markdown(f"**{company}** : {stats['Total']} paires ({stats['Doublons réels']} réels)")
        
        # =======================
        # 📥 EXPORT COMPLET
        # =======================
        st.subheader("📥 Export complet")
        
        # Préparer les dataframes pour Excel
        dataframes_excel = {}
        
        if doublons_reels > 0:
            reel_df = doublons_df[doublons_df["Categorie"] == "DOUBLON REEL"]
            dataframes_excel["doublons_reels"] = reel_df
        
        if a_analyser > 0:
            analyser_df = doublons_df[doublons_df["Categorie"] == "A ANALYSER"]
            dataframes_excel["a_analyser"] = analyser_df
        
        if recurrents > 0 and show_recurrent:
            recurrent_df = doublons_df[doublons_df["Categorie"] == "NORMAL"]
            dataframes_excel["recurrents"] = recurrent_df
        
        # Ajouter un résumé
        summary_data = {
            "Catégorie": ["Doublons réels", "À analyser", "POs récurrents", "TOTAL"],
            "Nombre": [doublons_reels, a_analyser, recurrents, total_pairs],
            "Pourcentage": [
                f"{doublons_reels/total_pairs*100:.1f}%" if total_pairs > 0 else "0%",
                f"{a_analyser/total_pairs*100:.1f}%" if total_pairs > 0 else "0%",
                f"{recurrents/total_pairs*100:.1f}%" if total_pairs > 0 else "0%",
                "100%"
            ]
        }
        dataframes_excel["resume"] = pd.DataFrame(summary_data)
        
        # Bouton d'export Excel
        if dataframes_excel:
            excel_data = export_to_excel(
                dataframes_excel,
                {
                    "doublons_reels": "Doublons réels",
                    "a_analyser": "À analyser",
                    "recurrents": "POs récurrents",
                    "resume": "Résumé"
                }
            )
            
            st.download_button(
                label="📊 Télécharger rapport Excel complet",
                data=excel_data,
                file_name=f"rapport_doublons_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # =======================
        # 🎯 RECOMMANDATIONS
        # =======================
        st.subheader("🎯 Plan d'action recommandé")
        
        if doublons_reels > 0:
            st.error(f"""
            **URGENT : {doublons_reels} DOUBLON(S) RÉEL(S) DÉTECTÉ(S)**
            
            **Actions immédiates :**
            1. **Contacter les équipes concernées** pour chaque doublon réel
            2. **Identifier la PO à conserver** (généralement la plus ancienne)
            3. **Annuler les doublons** dans le système
            4. **Documenter les actions** pour audit
            5. **Réviser les procédures** pour éviter les récidives
            
            **Délai recommandé :** 24-48 heures maximum
            """)
        
        if a_analyser > 0:
            st.warning(f"""
            **À TRAITER : {a_analyser} PO(S) À ANALYSER**
            
            **Actions à mener :**
            1. **Analyser manuellement** chaque cas
            2. **Contacter les demandeurs** pour clarification
            3. **Vérifier la légitimité** de chaque PO
            4. **Prendre une décision** (annuler ou valider)
            
            **Délai recommandé :** 1 semaine
            """)
        
        if recurrents > 0:
            st.info(f"""
            **INFORMATION : {recurrents} PO(S) RÉCURRENT(S) IDENTIFIÉ(S)**
            
            **Ces POs sont normaux :**
            - Services récurrents mensuels/trimestriels
            - Abonnements et contrats réguliers
            - Pas d'action nécessaire (sauf vérification des montants)
            
            **Recommandation :** Vérifier la cohérence des montants sur la période
            """)
    
    else:
        st.success("""
        ✅ **EXCELLENT ! AUCUN DOUBLON DÉTECTÉ**
        
        Votre processus d'achats semble bien maîtrisé.
        
        **Recommandations de maintien :**
        - Continuer les contrôles réguliers
        - Former les nouveaux utilisateurs
        - Maintenir les procédures actuelles
        """)

# ==============================
# SIDEBAR INFORMATIONS
# ==============================
with st.sidebar:
    st.markdown("## 🎯 Détection intelligente")
    st.markdown("""
    **Nouveautés V4.4 :**
    
    🔴 **DOUBLONS RÉELS**
    - Même référence PO
    - Même fournisseur + même montant exact
    - Nécessite action immédiate
    
    🟡 **À ANALYSER**
    - Similarités suspectes
    - Nécessite vérification manuelle
    - Décision humaine requise
    
    🟢 **POs RÉCURRENTS**
    - Même service, mois différent
    - Commandes régulières normales
    - Pas d'action nécessaire
    
    **Exemple de POs récurrents :**
    - "Gardiennage janvier" vs "Gardiennage février"
    - "Maintenance mars" vs "Maintenance avril"
    - "Abonnement logiciel Q1" vs "Abonnement logiciel Q2"
    """)
    
    st.markdown("---")
    st.markdown("## ⚙️ Comment ça marche ?")
    st.markdown("""
    1. **Analyse linguistique** des noms de PO
    2. **Détection des mois/périodes** différents
    3. **Extraction de la base commune** (sans dates)
    4. **Classification intelligente** en 3 catégories
    5. **Recommandations adaptées** pour chaque cas
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Paramètres conseillés")
    st.markdown("""
    **Pour audit strict :**
    - Mode : Intelligent
    - Fenêtre : 30 jours
    - Détecter récurrents : OUI
    - Afficher récurrents : NON
    
    **Pour analyse complète :**
    - Mode : Intelligent  
    - Fenêtre : 60 jours
    - Détecter récurrents : OUI
    - Afficher récurrents : OUI
    """)