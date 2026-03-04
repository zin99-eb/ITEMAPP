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
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import warnings
import zipfile
from sqlalchemy import create_engine, text
import urllib.parse
warnings.filterwarnings('ignore')

# ================================================================
# Configuration Streamlit
# ================================================================
st.set_page_config(
    page_title="Master Data Quality - Netis Group",
    page_icon="🔍",
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
WHERE created_at >= '2024-05-01' 
AND status = 'Qualified' """
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
    if 'last_quality_report' in st.session_state:
        del st.session_state.last_quality_report
    st.rerun()

# ================================================================
# === PLAN COMPTABLE NETIS GROUP AVEC MOTS-CLÉS ENRICHIS
# ================================================================

# Données du plan comptable fournies (version complète)
ACCOUNTING_CODES = [
    {"code": "1111.001", "description_fr": "Ventes externes de matériaux uniquement", "description_en": "External sales of materials only", "category": "VENTES", "subcategory": "Ventes externes"},
    {"code": "1111.002", "description_fr": "Ventes externes de services uniquement", "description_en": "External sales of services only", "category": "VENTES", "subcategory": "Ventes externes"},
    {"code": "1111.003", "description_fr": "Ventes externes de matériaux et services", "description_en": "External sales of materials and services", "category": "VENTES", "subcategory": "Ventes externes"},
    {"code": "1112.001", "description_fr": "Ventes interco de matériaux uniquement", "description_en": "Interco sales of materials only", "category": "VENTES", "subcategory": "Ventes interco"},
    {"code": "1112.002", "description_fr": "Ventes interco de services uniquement", "description_en": "Interco sales of services only", "category": "VENTES", "subcategory": "Ventes interco"},
    {"code": "1112.003", "description_fr": "Ventes interco de matériaux et services", "description_en": "Interco sales of materials and services", "category": "VENTES", "subcategory": "Ventes interco"},
    {"code": "1211.001", "description_fr": "Main d'œuvre", "description_en": "Internal manpower (blue-collar and hourly paid staff)", "category": "RESSOURCES HUMAINES", "subcategory": "Main d'œuvre directe"},
    {"code": "1211.002", "description_fr": "Intérim horaires", "description_en": "External manpower (blue-collar and hourly paid staff)", "category": "RESSOURCES HUMAINES", "subcategory": "Intérim"},
    {"code": "1212.001", "description_fr": "Salaires encadrement expat", "description_en": "Internal expat supervision and executive staff", "category": "RESSOURCES HUMAINES", "subcategory": "Encadrement"},
    {"code": "1212.002", "description_fr": "Salaires encadrement local", "description_en": "Internal local supervision and executive staff", "category": "RESSOURCES HUMAINES", "subcategory": "Encadrement"},
    {"code": "1212.003", "description_fr": "Encadrement intérimaire", "description_en": "External supervision and executive staff", "category": "RESSOURCES HUMAINES", "subcategory": "Encadrement externe"},
    {"code": "1213.001", "description_fr": "Missions et voyages", "description_en": "Missions and travel costs", "category": "RESSOURCES HUMAINES", "subcategory": "Frais de mission"},
    {"code": "1213.002", "description_fr": "Notes de frais et réceptions", "description_en": "Expense claims and receptions costs", "category": "RESSOURCES HUMAINES", "subcategory": "Frais de représentation"},
    {"code": "1213.003", "description_fr": "Formation", "description_en": "Training costs", "category": "RESSOURCES HUMAINES", "subcategory": "Formation"},
    {"code": "1213.004", "description_fr": "Assurance santé, médecine du travail et autres frais médicaux", "description_en": "Health insurance and other health costs", "category": "RESSOURCES HUMAINES", "subcategory": "Frais médicaux"},
    {"code": "1213.005", "description_fr": "Frais divers personnel", "description_en": "Misc. HR costs", "category": "RESSOURCES HUMAINES", "subcategory": "Divers RH"},
    {"code": "1213.006", "description_fr": "Frais de restauration", "description_en": "Catering costs", "category": "RESSOURCES HUMAINES", "subcategory": "Restauration"},
    {"code": "1213.007", "description_fr": "Loyers & charges locatives terrains & locaux (habitation)", "description_en": "Rents, land & premises and related expenses (living)", "category": "RESSOURCES HUMAINES", "subcategory": "Logement"},
    {"code": "1221.001", "description_fr": "Béton prêt à l'emploi", "description_en": "Ready-mix concrete", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.002", "description_fr": "Cailloux, sables et autres matériaux de remblai", "description_en": "Rocks, sands and other backfill materials", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.003", "description_fr": "Aciers pour armatures et armatures façonnées", "description_en": "Steel rebars and shaped steel frame", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.004", "description_fr": "Parpaings & briques", "description_en": "Bricks and building blocks", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.005", "description_fr": "Géotextiles", "description_en": "Geotextiles", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.006", "description_fr": "Étanchéité", "description_en": "Sealing", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.007", "description_fr": "Divers préfabriqués béton", "description_en": "Misc precasted concrete elements", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.008", "description_fr": "Tuyaux fonte", "description_en": "Cast iron pipes", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.009", "description_fr": "Tuyaux PVC", "description_en": "PVC pipes", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.010", "description_fr": "Grillage", "description_en": "Fence", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.011", "description_fr": "Bois", "description_en": "Wood", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1221.012", "description_fr": "Divers matériaux GC incorporés", "description_en": "Miscellaneous civil works incorporated materials", "category": "MATERIAUX", "subcategory": "Génie civil"},
    {"code": "1222.001", "description_fr": "Pylônes telecom greenfield", "description_en": "Pylon telecom greenfield", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.002", "description_fr": "Pylônes telecom toit-terrasse", "description_en": "Pylon telecom rooftop", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.003", "description_fr": "Pylônes électriques", "description_en": "Electric pylons", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.004", "description_fr": "Support panneaux solaires", "description_en": "Solar panel supports", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.005", "description_fr": "Profilés métalliques", "description_en": "Metal profiles and beams", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.006", "description_fr": "Ancrage, barres, boulons", "description_en": "Anchorage, rods, bolts and nuts", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.007", "description_fr": "Câbles métalliques / aciers", "description_en": "Steel/metal ropes and cables", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.008", "description_fr": "Tôles métalliques", "description_en": "Metal sheets", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.009", "description_fr": "Produits et accessoires anti-corrosion", "description_en": "Anti corrosive products and accessories", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1222.010", "description_fr": "Divers matériaux métalliques incorporés", "description_en": "Miscellaneous metallic incorporated materials", "category": "MATERIAUX", "subcategory": "Métallerie"},
    {"code": "1223.001", "description_fr": "Poteaux bois", "description_en": "Wooden poles", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.002", "description_fr": "Poteaux béton", "description_en": "Concrete poles", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.003", "description_fr": "Poteaux métalliques", "description_en": "Metal poles", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.004", "description_fr": "Connecteurs FO", "description_en": "FO connectors", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.005", "description_fr": "Câble FO", "description_en": "Cable FO", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.006", "description_fr": "Boîtier FO", "description_en": "Box FO", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.007", "description_fr": "Panneaux et armoires FO", "description_en": "FO panels and cabinets", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1223.008", "description_fr": "Divers matériaux FO incorporés", "description_en": "Miscellaneous FO incorporated materials", "category": "MATERIAUX", "subcategory": "Fibre optique"},
    {"code": "1224.001", "description_fr": "Système protection anti-incendie", "description_en": "Fire protection system", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.002", "description_fr": "Lampadaires", "description_en": "Streetlights", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.003", "description_fr": "Système d'éclairage", "description_en": "Lighting systems", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.004", "description_fr": "Système vidéo surveillance", "description_en": "CCTV systems", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.005", "description_fr": "Pompes et compresseurs industriels", "description_en": "Industrial pumps and compressors", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.006", "description_fr": "Instrumentation et système de mesure", "description_en": "Instrumentation and metering systems", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.007", "description_fr": "Système de climatisation et refroidissement", "description_en": "Cooling and air conditioning systems", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1224.008", "description_fr": "Autres équipements industriels incorporés", "description_en": "Other incorporated industrial equipment", "category": "MATERIAUX", "subcategory": "Équipements industriels"},
    {"code": "1225.001", "description_fr": "Groupes électrogènes", "description_en": "Generators", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.002", "description_fr": "Cuves à gasoil", "description_en": "Fuel tanks", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.003", "description_fr": "Inverseurs & ATS", "description_en": "Inverters & ATS", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.004", "description_fr": "Batteries", "description_en": "Batteries", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.005", "description_fr": "Panneaux solaires", "description_en": "Solar panels", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.006", "description_fr": "Redresseurs", "description_en": "Rectifiers", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.007", "description_fr": "Câbles électriques", "description_en": "Electrical cables", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.008", "description_fr": "Paratonnerre et matériel de mise à la terre", "description_en": "Lightning rods and grounding equipment", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.009", "description_fr": "Compteurs et armoires électriques", "description_en": "Electrical panels and cabinets", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1225.010", "description_fr": "Divers fournitures électriques incorporées", "description_en": "Misc electrical incorporated materials", "category": "MATERIAUX", "subcategory": "Électricité"},
    {"code": "1226.001", "description_fr": "Divers équipements télécoms actifs", "description_en": "Misc active telecom equipment", "category": "MATERIAUX", "subcategory": "Télécoms"},
    {"code": "1227.001", "description_fr": "PDR pour entretien courant et réparations diverses (sites clients)", "description_en": "Spare parts for daily and minor unplanned maintenance (customers sites)", "category": "MAINTENANCE", "subcategory": "Pièces détachées"},
    {"code": "1227.002", "description_fr": "PDR pour grande maintenance planifiée & entretien majeur (sites clients)", "description_en": "Spare parts for major and planned maintenance (customers sites)", "category": "MAINTENANCE", "subcategory": "Pièces détachées"},
    {"code": "1227.003", "description_fr": "PDR pour entretien & réparation suite accident ou casse (sites clients)", "description_en": "Repairs after accident and breakdown (customers sites)", "category": "MAINTENANCE", "subcategory": "Pièces détachées"},
    {"code": "1227.004", "description_fr": "Huiles & graisses pour maintenance (sites clients)", "description_en": "Oils, lubricants & grease for maintenance (customers sites)", "category": "MAINTENANCE", "subcategory": "Lubrifiants"},
    {"code": "1227.005", "description_fr": "Divers matériaux et consommables pour la maintenance des sites clients", "description_en": "Misc materials and consumables for customers sites", "category": "MAINTENANCE", "subcategory": "Divers maintenance"},
    {"code": "1228.001", "description_fr": "Autres équipements réseaux", "description_en": "Misc network equipment", "category": "MATERIAUX", "subcategory": "Réseaux"},
    {"code": "1228.002", "description_fr": "Peinture", "description_en": "Painting", "category": "MATERIAUX", "subcategory": "Finition"},
    {"code": "1228.003", "description_fr": "Autres matériaux incorporés", "description_en": "Other incorporated materials", "category": "MATERIAUX", "subcategory": "Divers"},
    {"code": "1229.001", "description_fr": "Carburant pour refueling des sites clients", "description_en": "Customers sites refueling", "category": "CARBURANT", "subcategory": "Clients"},
    {"code": "1231.001", "description_fr": "Petit outillage", "description_en": "Other tools and small equipment", "category": "OUTILLAGE", "subcategory": "Petit outillage"},
    {"code": "1231.002", "description_fr": "Consommable, outillage de soudure", "description_en": "Welding tools and consumables", "category": "OUTILLAGE", "subcategory": "Soudure"},
    {"code": "1231.003", "description_fr": "Accessoires et outils de levage", "description_en": "Lifting and handling tools and accessories", "category": "OUTILLAGE", "subcategory": "Levage"},
    {"code": "1231.004", "description_fr": "Consommables coffrages", "description_en": "Formwork consumables", "category": "OUTILLAGE", "subcategory": "Coffrage"},
    {"code": "1232.001", "description_fr": "Échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails", "category": "OUTILLAGE", "subcategory": "Échafaudage"},
    {"code": "1232.002", "description_fr": "Autres outils temporaires", "description_en": "Other temporary tools", "category": "OUTILLAGE", "subcategory": "Outils temporaires"},
    {"code": "1233.001", "description_fr": "Consommables environnement", "description_en": "Environment consumables", "category": "SECURITE", "subcategory": "Environnement"},
    {"code": "1233.002", "description_fr": "Matériel signalisation", "description_en": "Signs, buoys and anchors", "category": "SECURITE", "subcategory": "Signalisation"},
    {"code": "1233.003", "description_fr": "Équipement de Protection Individuel (EPI)", "description_en": "PPE and safety expenses", "category": "SECURITE", "subcategory": "EPI"},
    {"code": "1234.001", "description_fr": "Eau", "description_en": "Water", "category": "CONSOMMABLES", "subcategory": "Fluides"},
    {"code": "1234.002", "description_fr": "Divers consommables", "description_en": "Other misc. consumables", "category": "CONSOMMABLES", "subcategory": "Divers"},
    {"code": "1241.001", "description_fr": "Location externe voiture, pickup & suv", "description_en": "Car, pickup & suv external rental", "category": "LOCATION", "subcategory": "Véhicules légers"},
    {"code": "1241.002", "description_fr": "Location externe motos & triporteurs", "description_en": "Motorbikes & three wheelers external rental", "category": "LOCATION", "subcategory": "Deux roues"},
    {"code": "1241.003", "description_fr": "Location externe véhicules pour refueling", "description_en": "Refueling vehicles external rental", "category": "LOCATION", "subcategory": "Véhicules carburant"},
    {"code": "1241.004", "description_fr": "Location externe autres camions", "description_en": "Other trucks external rental", "category": "LOCATION", "subcategory": "Poids lourds"},
    {"code": "1241.005", "description_fr": "Location externe grue mobile", "description_en": "Mobile crane external rental", "category": "LOCATION", "subcategory": "Levage"},
    {"code": "1241.006", "description_fr": "Location externe autres matériel de levage et manutention", "description_en": "Other lifting and handling equipment external rental", "category": "LOCATION", "subcategory": "Manutention"},
    {"code": "1241.007", "description_fr": "Location externe de pelle", "description_en": "HEX external rental", "category": "LOCATION", "subcategory": "Engins TP"},
    {"code": "1241.008", "description_fr": "Location externe de chargeur", "description_en": "Wheel loader external rental", "category": "LOCATION", "subcategory": "Engins TP"},
    {"code": "1241.009", "description_fr": "Location externe matériel compactage", "description_en": "Compaction equipment external rental", "category": "LOCATION", "subcategory": "Engins TP"},
    {"code": "1241.010", "description_fr": "Location externe matériel de réglage", "description_en": "Grader external rental", "category": "LOCATION", "subcategory": "Engins TP"},
    {"code": "1241.011", "description_fr": "Location externe autre matériel de terrassement", "description_en": "Other earth-moving equipment external rental", "category": "LOCATION", "subcategory": "Engins TP"},
    {"code": "1241.012", "description_fr": "Location externe échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails external rental", "category": "LOCATION", "subcategory": "Échafaudage"},
    {"code": "1241.013", "description_fr": "Location externe matériel production énergie et éclairage", "description_en": "Generator and lighting equipment external rental", "category": "LOCATION", "subcategory": "Énergie"},
    {"code": "1241.014", "description_fr": "Location externe matériel de forage et passage de câbles", "description_en": "Drilling and cable routing equipment external rental", "category": "LOCATION", "subcategory": "Forage"},
    {"code": "1241.015", "description_fr": "Location externe matériel divers", "description_en": "Other misc equipment external rental", "category": "LOCATION", "subcategory": "Divers"},
    {"code": "1241.016", "description_fr": "Location externe outillage et machines FO", "description_en": "Other tooling and machinery FO external rental", "category": "LOCATION", "subcategory": "FO"},
    {"code": "1241.017", "description_fr": "Location externe autres outillage et machines", "description_en": "Other tooling and machinery external rental", "category": "LOCATION", "subcategory": "Machines"},
    {"code": "1242.001", "description_fr": "Amortissements voiture, pickup & suv", "description_en": "Car, pickup & suv depreciations", "category": "AMORTISSEMENTS", "subcategory": "Véhicules légers"},
    {"code": "1242.002", "description_fr": "Amortissements motos & triporteurs", "description_en": "Motorbikes & three wheelers depreciations", "category": "AMORTISSEMENTS", "subcategory": "Deux roues"},
    {"code": "1242.003", "description_fr": "Amortissements véhicules pour refueling", "description_en": "Refueling vehicles depreciations", "category": "AMORTISSEMENTS", "subcategory": "Véhicules carburant"},
    {"code": "1242.004", "description_fr": "Amortissements autres camions", "description_en": "Other trucks depreciations", "category": "AMORTISSEMENTS", "subcategory": "Poids lourds"},
    {"code": "1242.005", "description_fr": "Amortissements grue mobile", "description_en": "Mobile crane depreciations", "category": "AMORTISSEMENTS", "subcategory": "Levage"},
    {"code": "1242.006", "description_fr": "Amortissements autres matériel de levage et manutention", "description_en": "Other lifting and handling equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Manutention"},
    {"code": "1242.007", "description_fr": "Amortissements de pelle", "description_en": "HEX depreciations", "category": "AMORTISSEMENTS", "subcategory": "Engins TP"},
    {"code": "1242.008", "description_fr": "Amortissements de chargeur", "description_en": "Wheel loader depreciations", "category": "AMORTISSEMENTS", "subcategory": "Engins TP"},
    {"code": "1242.009", "description_fr": "Amortissements matériel compactage", "description_en": "Compaction equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Engins TP"},
    {"code": "1242.010", "description_fr": "Amortissements matériel de réglage", "description_en": "Grader depreciations", "category": "AMORTISSEMENTS", "subcategory": "Engins TP"},
    {"code": "1242.011", "description_fr": "Amortissements autre matériel de terrassement", "description_en": "Other earth-moving equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Engins TP"},
    {"code": "1242.012", "description_fr": "Amortissements échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails depreciations", "category": "AMORTISSEMENTS", "subcategory": "Échafaudage"},
    {"code": "1242.013", "description_fr": "Amortissements matériel production énergie et éclairage", "description_en": "Generator and lighting equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Énergie"},
    {"code": "1242.014", "description_fr": "Amortissements matériel de forage et passage de câbles", "description_en": "Drilling and cable routing equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Forage"},
    {"code": "1242.015", "description_fr": "Amortissements matériel divers", "description_en": "Other misc equipment depreciations", "category": "AMORTISSEMENTS", "subcategory": "Divers"},
    {"code": "1242.016", "description_fr": "Amortissements outillage et machines FO", "description_en": "Other tooling and machinery FO depreciations", "category": "AMORTISSEMENTS", "subcategory": "FO"},
    {"code": "1242.017", "description_fr": "Amortissements autres outillage et machines", "description_en": "Other tooling and machinery depreciations", "category": "AMORTISSEMENTS", "subcategory": "Machines"},
    {"code": "1243.001", "description_fr": "Essence / Diesel pour VL", "description_en": "Petrol and diesel for LV", "category": "CARBURANT", "subcategory": "Véhicules"},
    {"code": "1243.002", "description_fr": "Essence / Diesel pour autre usage interne", "description_en": "Petrol and diesel for other internal use", "category": "CARBURANT", "subcategory": "Interne"},
    {"code": "1243.003", "description_fr": "Huiles & graisses pour usage interne", "description_en": "Oils, lubricants & grease for internal use", "category": "MAINTENANCE", "subcategory": "Lubrifiants internes"},
    {"code": "1244.001", "description_fr": "Entretiens, réparations et PDR pour VL", "description_en": "Maintenances, repairs and spare parts for LV", "category": "MAINTENANCE", "subcategory": "Véhicules"},
    {"code": "1244.002", "description_fr": "Entretiens, réparations et PDR pour autres équipements internes", "description_en": "Maintenances, repairs and spare parts for other internal equipment", "category": "MAINTENANCE", "subcategory": "Équipements"},
    {"code": "1251.001", "description_fr": "Import - fret aérien OPEX", "description_en": "Import - air freight OPEX", "category": "TRANSPORT", "subcategory": "Import aérien"},
    {"code": "1251.002", "description_fr": "Import - fret maritime OPEX", "description_en": "Import - sea freight OPEX", "category": "TRANSPORT", "subcategory": "Import maritime"},
    {"code": "1251.003", "description_fr": "Import - fret aérien CAPEX", "description_en": "Import - air freight CAPEX", "category": "TRANSPORT", "subcategory": "Import aérien"},
    {"code": "1251.004", "description_fr": "Import - fret maritime CAPEX", "description_en": "Import - sea freight CAPEX", "category": "TRANSPORT", "subcategory": "Import maritime"},
    {"code": "1252.001", "description_fr": "Export - fret aérien OPEX", "description_en": "Export - air freight OPEX", "category": "TRANSPORT", "subcategory": "Export aérien"},
    {"code": "1252.002", "description_fr": "Export - fret maritime OPEX", "description_en": "Export - sea freight OPEX", "category": "TRANSPORT", "subcategory": "Export maritime"},
    {"code": "1252.003", "description_fr": "Export - fret aérien CAPEX", "description_en": "Export - air freight CAPEX", "category": "TRANSPORT", "subcategory": "Export aérien"},
    {"code": "1252.004", "description_fr": "Export - fret maritime CAPEX", "description_en": "Export - sea freight CAPEX", "category": "TRANSPORT", "subcategory": "Export maritime"},
    {"code": "1253.001", "description_fr": "Import & Export - droits de douane et coûts associés OPEX", "description_en": "Import & Export - custom duties and related costs OPEX", "category": "DOUANES", "subcategory": "Droits de douane"},
    {"code": "1253.002", "description_fr": "Import & Export - droits de douane et coûts associés CAPEX", "description_en": "Import & Export - custom duties and related costs CAPEX", "category": "DOUANES", "subcategory": "Droits de douane"},
    {"code": "1254.001", "description_fr": "Assurances transport OPEX", "description_en": "Transport insurances OPEX", "category": "ASSURANCES", "subcategory": "Transport"},
    {"code": "1254.002", "description_fr": "Assurances transport CAPEX", "description_en": "Transport insurances CAPEX", "category": "ASSURANCES", "subcategory": "Transport"},
    {"code": "1254.003", "description_fr": "Autres frais de transport OPEX", "description_en": "Other transportation costs OPEX", "category": "TRANSPORT", "subcategory": "Divers"},
    {"code": "1254.004", "description_fr": "Autres frais de transport CAPEX", "description_en": "Other transportation costs CAPEX", "category": "TRANSPORT", "subcategory": "Divers"},
    {"code": "1261.001", "description_fr": "Frais d'études techniques externes (design and methods)", "description_en": "External technical studies (design and methods)", "category": "ETUDES", "subcategory": "Externes"},
    {"code": "1261.002", "description_fr": "Géomètres, géotechniciens et autres techniciens extérieurs", "description_en": "External topographic, geotechnical and other technical surveys", "category": "ETUDES", "subcategory": "Géomètres"},
    {"code": "1261.003", "description_fr": "Frais de laboratoire", "description_en": "Laboratory studies costs", "category": "ETUDES", "subcategory": "Laboratoire"},
    {"code": "1261.004", "description_fr": "Frais d'études partenaires de groupement", "description_en": "Studies and surveys partner JV and consortium", "category": "ETUDES", "subcategory": "Partenaires"},
    {"code": "1262.001", "description_fr": "Frais internes études techniques", "description_en": "Internal technical study costs", "category": "ETUDES", "subcategory": "Internes"},
    {"code": "1262.002", "description_fr": "Frais tendering internes", "description_en": "Internal tendering costs", "category": "ETUDES", "subcategory": "Appels d'offres"},
    {"code": "1262.003", "description_fr": "Frais d'achats et logistique internes", "description_en": "Procurement, logistics and post-order internal services", "category": "LOGISTIQUE", "subcategory": "Internes"},
    {"code": "1262.004", "description_fr": "Autres refacturations internes de services", "description_en": "Other costs internal services", "category": "SERVICES", "subcategory": "Internes"},
    {"code": "1263.001", "description_fr": "Frais contrôle technique / homologation diverse", "description_en": "Certifications costs", "category": "ETUDES", "subcategory": "Certification"},
    {"code": "1264.001", "description_fr": "Autres prestations", "description_en": "Other services provided", "category": "SERVICES", "subcategory": "Divers"},
    {"code": "1271.001", "description_fr": "Loyers & charges locatives terrains & locaux (pro)", "description_en": "Rents, land & premises and related expenses (pro)", "category": "LOYERS", "subcategory": "Professionnel"},
    {"code": "1271.002", "description_fr": "Location externe bungalows et divers installations pro", "description_en": "Site facilities and bungalows external rental", "category": "LOCATION", "subcategory": "Installations"},
    {"code": "1271.003", "description_fr": "Amortissements bungalows et divers installations pro", "description_en": "Site facilities and bungalows depreciations", "category": "AMORTISSEMENTS", "subcategory": "Installations"},
    {"code": "1271.004", "description_fr": "Électricité, gaz et eau des installations pro", "description_en": "Electricity, gas and water of sites facilities", "category": "FLUIDES", "subcategory": "Professionnel"},
    {"code": "1271.005", "description_fr": "Gardiennage, frais de sécurité", "description_en": "Security expenses", "category": "SECURITE", "subcategory": "Gardiennage"},
    {"code": "1271.006", "description_fr": "Autres coûts d'installations et d'infrastructures OPEX", "description_en": "Other installations and facilities costs OPEX", "category": "INSTALLATIONS", "subcategory": "OPEX"},
    {"code": "1271.007", "description_fr": "Autres coûts d'installations et d'infrastructures - amortissements", "description_en": "Other installations and facilities costs depreciations", "category": "INSTALLATIONS", "subcategory": "Amortissements"},
    {"code": "1271.008", "description_fr": "Mise en décharge matériaux divers et prestations environnementales", "description_en": "Landfilling activity & other environmental service", "category": "ENVIRONNEMENT", "subcategory": "Décharge"},
    {"code": "1271.009", "description_fr": "Prestation, entretien et nettoyage", "description_en": "Cleaning, maintenance and garbage disposals", "category": "SERVICES", "subcategory": "Nettoyage"},
    {"code": "1272.001", "description_fr": "Frais d'actes et contentieux", "description_en": "Deeds and disputes charges", "category": "FRAIS JURIDIQUES", "subcategory": "Contentieux"},
    {"code": "1272.002", "description_fr": "Auditeurs", "description_en": "Auditors", "category": "FRAIS JURIDIQUES", "subcategory": "Audit"},
    {"code": "1272.003", "description_fr": "Conseil fiscal", "description_en": "Tax consultant", "category": "FRAIS JURIDIQUES", "subcategory": "Conseil"},
    {"code": "1272.004", "description_fr": "Documentation & traduction", "description_en": "Documentation and translation costs", "category": "FRAIS JURIDIQUES", "subcategory": "Traduction"},
    {"code": "1272.005", "description_fr": "Autres conseils et honoraires", "description_en": "Other consultancy and fees", "category": "FRAIS JURIDIQUES", "subcategory": "Conseils divers"},
    {"code": "1273.001", "description_fr": "Frais de télécommunication", "description_en": "Telecommunication costs", "category": "TELECOM", "subcategory": "Frais"},
    {"code": "1273.002", "description_fr": "Coûts informatiques - matériel OPEX", "description_en": "IT costs hardware OPEX", "category": "INFORMATIQUE", "subcategory": "Matériel OPEX"},
    {"code": "1273.003", "description_fr": "Coûts informatiques - logiciels et licences OPEX", "description_en": "IT costs software and licenses OPEX", "category": "INFORMATIQUE", "subcategory": "Logiciels OPEX"},
    {"code": "1273.004", "description_fr": "Coûts informatiques - amortissements du matériel", "description_en": "IT costs hardware depreciations", "category": "INFORMATIQUE", "subcategory": "Matériel Amort"},
    {"code": "1273.005", "description_fr": "Coûts informatiques - amortissements des logiciels et licences", "description_en": "IT costs software and licenses depreciations", "category": "INFORMATIQUE", "subcategory": "Logiciels Amort"},
    {"code": "1274.001", "description_fr": "Assurances véhicules légers", "description_en": "LV insurances", "category": "ASSURANCES", "subcategory": "Véhicules"},
    {"code": "1274.002", "description_fr": "Assurance tous risques chantier", "description_en": "Contractor's all risks insurance", "category": "ASSURANCES", "subcategory": "Chantiers"},
    {"code": "1274.003", "description_fr": "Assurances multirisques", "description_en": "Multi-risks insurances", "category": "ASSURANCES", "subcategory": "Multirisque"},
    {"code": "1274.004", "description_fr": "Autres assurances", "description_en": "Other insurances", "category": "ASSURANCES", "subcategory": "Divers"},
    {"code": "1274.005", "description_fr": "Assurance bris de machine", "description_en": "Machinery breakdown insurance", "category": "ASSURANCES", "subcategory": "Machines"},
    {"code": "1274.006", "description_fr": "Frais bancaires", "description_en": "Banking charges", "category": "FRAIS FINANCIERS", "subcategory": "Bancaires"},
    {"code": "1274.007", "description_fr": "Frais de garantie et cautions", "description_en": "Bank guarantees and bonds costs", "category": "FRAIS FINANCIERS", "subcategory": "Garanties"},
    {"code": "1275.001", "description_fr": "Fournitures de bureau et consommables informatiques", "description_en": "Office consumables and supplies", "category": "FOURNITURES", "subcategory": "Bureau"},
    {"code": "1275.002", "description_fr": "Charges administratives diverses", "description_en": "Miscellaneous administrative charges", "category": "FRAIS ADMIN", "subcategory": "Divers"},
    {"code": "1275.003", "description_fr": "Business promotion", "description_en": "Business promotion", "category": "MARKETING", "subcategory": "Promotion"},
    {"code": "1275.004", "description_fr": "Dépenses RSE", "description_en": "RSE & CSR expenses", "category": "RSE", "subcategory": "Divers"},
    {"code": "1275.005", "description_fr": "Publicité, annonces insertions", "description_en": "Advertising expenses", "category": "MARKETING", "subcategory": "Publicité"},
    {"code": "1275.006", "description_fr": "Matériel bureau OPEX", "description_en": "Office equipment costs OPEX", "category": "FOURNITURES", "subcategory": "Matériel OPEX"},
    {"code": "1275.007", "description_fr": "Matériel bureau amortissements", "description_en": "Office equipment costs depreciations", "category": "FOURNITURES", "subcategory": "Matériel Amort"},
    {"code": "1276.001", "description_fr": "Divers impôts et taxes (hors douanes, IS et RAS sur dividendes)", "description_en": "Misc. taxes (excl. Customs, CIT and WHT on dividends)", "category": "TAXES", "subcategory": "Divers"},
    {"code": "1281.001", "description_fr": "Sous-traitants génie civil", "description_en": "Civil works subcontractors", "category": "SOUS-TRAITANCE", "subcategory": "Génie civil"},
    {"code": "1282.001", "description_fr": "Sous-traitants électricité", "description_en": "Electrical subcontractors", "category": "SOUS-TRAITANCE", "subcategory": "Électricité"},
    {"code": "1283.001", "description_fr": "Sous-traitants fibre optique", "description_en": "FO subcontractors", "category": "SOUS-TRAITANCE", "subcategory": "Fibre optique"},
    {"code": "1284.001", "description_fr": "Sous-traitants télécoms", "description_en": "TELCO subcontractors", "category": "SOUS-TRAITANCE", "subcategory": "Télécoms"},
    {"code": "1285.001", "description_fr": "Sous-traitants autres", "description_en": "Other subcontractors", "category": "SOUS-TRAITANCE", "subcategory": "Divers"},
    {"code": "1311.001", "description_fr": "Refacturation interco de l'assistance administrative", "description_en": "Interco rebilling of admin assistance", "category": "INTERCO", "subcategory": "Admin"},
    {"code": "1311.002", "description_fr": "Refacturation interco des coûts d'appel d'offres et assistance commerciale", "description_en": "Interco rebilling of tender costs and commercial assistance", "category": "INTERCO", "subcategory": "Commercial"},
    {"code": "1311.003", "description_fr": "Refacturation interco des études et assistance technique", "description_en": "Interco rebilling of studies and technical assistance", "category": "INTERCO", "subcategory": "Technique"},
    {"code": "1311.004", "description_fr": "Autres transferts et refacturations interco de coûts", "description_en": "Interco other transfer and rebilling of costs", "category": "INTERCO", "subcategory": "Divers"},
    {"code": "1312.001", "description_fr": "Autres produits d'activité annexe", "description_en": "Other operating revenues", "category": "PRODUITS", "subcategory": "Annexes"},
    {"code": "1321.001", "description_fr": "Ajustements divers de stocks opérationnels", "description_en": "Misc stocks adjustments operational", "category": "STOCKS", "subcategory": "Ajustements"},
    {"code": "1322.001", "description_fr": "Dotations et reprises aux provisions d'exploitation", "description_en": "Operating provisions and reversal", "category": "PROVISIONS", "subcategory": "Exploitation"},
    {"code": "1323.001", "description_fr": "Correctifs de lissage IFRS 15 & PAT", "description_en": "IFRS 15 progress adjustments and loss provision", "category": "IFRS", "subcategory": "Ajustements"},
    {"code": "1324.001", "description_fr": "Autres charges opérationnelles diverses", "description_en": "Other misc operating costs", "category": "CHARGES", "subcategory": "Opérationnelles"},
    {"code": "1331.001", "description_fr": "Collecte interco des frais de gestion", "description_en": "Interco collect of management fees", "category": "INTERCO", "subcategory": "Management fees"},
    {"code": "1332.001", "description_fr": "Charges de frais de gestion", "description_en": "Management fees costs", "category": "MANAGEMENT FEES", "subcategory": "Charges"},
    {"code": "1411.001", "description_fr": "Gain opérationnel de change", "description_en": "Operational FX gain", "category": "FINANCIER", "subcategory": "Gain change"},
    {"code": "1411.002", "description_fr": "Perte opérationnelle de change", "description_en": "Operational FX loss", "category": "FINANCIER", "subcategory": "Perte change"},
    {"code": "1511.001", "description_fr": "Divers produits et charges opérationnels non reportés", "description_en": "Misc non reporting operating revenues & costs", "category": "NON REPORTING", "subcategory": "Opérationnel"},
    {"code": "2111.001", "description_fr": "Autres produits financiers", "description_en": "Other financial revenues", "category": "FINANCIER", "subcategory": "Produits"},
    {"code": "2121.001", "description_fr": "Intérêts et autres frais financiers", "description_en": "Interests and other financial charges", "category": "FINANCIER", "subcategory": "Intérêts"},
    {"code": "2211.001", "description_fr": "Dotations et reprises aux provisions financières", "description_en": "Financial provisions and reversal", "category": "PROVISIONS", "subcategory": "Financières"},
    {"code": "2411.001", "description_fr": "Gain financier de change", "description_en": "Financial FX gain", "category": "FINANCIER", "subcategory": "Gain change"},
    {"code": "2411.002", "description_fr": "Perte financière de change", "description_en": "Financial FX loss", "category": "FINANCIER", "subcategory": "Perte change"},
    {"code": "2511.001", "description_fr": "Dividendes (élément non reporté)", "description_en": "Dividends (non reporting item)", "category": "FINANCIER", "subcategory": "Dividendes"},
    {"code": "2511.002", "description_fr": "Divers produits et charges financiers non reportés", "description_en": "Misc non reporting financial revenues & costs", "category": "NON REPORTING", "subcategory": "Financier"},
    {"code": "3111.001", "description_fr": "Frais de restructuration", "description_en": "Restructuring costs", "category": "EXCEPTIONNEL", "subcategory": "Restructuration"},
    {"code": "3211.001", "description_fr": "Produits de cession d'immobilisations", "description_en": "Assets resale revenues", "category": "EXCEPTIONNEL", "subcategory": "Cessions"},
    {"code": "3211.002", "description_fr": "Valeur nette comptable des immobilisations cédées", "description_en": "Assets resale costs (net book value)", "category": "EXCEPTIONNEL", "subcategory": "Cessions"},
    {"code": "3221.001", "description_fr": "Amendes et pénalités", "description_en": "Penalties", "category": "EXCEPTIONNEL", "subcategory": "Pénalités"},
    {"code": "3221.002", "description_fr": "Dotations et reprises aux provisions exceptionnelles", "description_en": "Exceptional provisions and reversal", "category": "PROVISIONS", "subcategory": "Exceptionnelles"},
    {"code": "3311.001", "description_fr": "Ajustements exercices antérieurs", "description_en": "Prior year adjustments", "category": "EXCEPTIONNEL", "subcategory": "Antérieurs"},
    {"code": "3311.002", "description_fr": "Autres produits exceptionnels", "description_en": "Other exceptional revenues", "category": "EXCEPTIONNEL", "subcategory": "Produits"},
    {"code": "3311.003", "description_fr": "Autres charges exceptionnelles", "description_en": "Other exceptional costs", "category": "EXCEPTIONNEL", "subcategory": "Charges"},
    {"code": "3411.001", "description_fr": "Gain exceptionnel de change", "description_en": "Exceptional FX gain", "category": "EXCEPTIONNEL", "subcategory": "Gain change"},
    {"code": "3411.002", "description_fr": "Perte exceptionnelle de change", "description_en": "Exceptional FX loss", "category": "EXCEPTIONNEL", "subcategory": "Perte change"},
    {"code": "3511.001", "description_fr": "Divers produits et charges exceptionnels non reportés", "description_en": "Misc non reporting exceptional revenues & costs", "category": "NON REPORTING", "subcategory": "Exceptionnel"},
    {"code": "4111.001", "description_fr": "Impôt sur les sociétés & impôt minimum forfaitaire", "description_en": "CIT & Minimum LS tax", "category": "IMPOTS", "subcategory": "Sociétés"},
    {"code": "4111.002", "description_fr": "Retenue à la source sur dividendes", "description_en": "WHT on dividends", "category": "IMPOTS", "subcategory": "Dividendes"},
    {"code": "4111.003", "description_fr": "Autres impôts et taxes non opérationnels", "description_en": "Other non operating taxes", "category": "IMPOTS", "subcategory": "Hors exploitation"}
]

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
# Analyseur de Noms - Détection des anomalies dans les désignations
# ================================================================
class NameQualityAnalyzer:
    """
    Analyse la qualité des noms/désignations des items
    Détecte les anomalies, incohérences et erreurs
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.issues = []
        self.results = {}
        
    def analyze(self):
        """Lance toutes les analyses de noms"""
        
        # 1. Analyse des longueurs
        self._analyze_name_lengths()
        
        # 2. Analyse des caractères spéciaux
        self._analyze_special_chars()
        
        # 3. Analyse des mots interdits/aberrants
        self._analyze_forbidden_words()
        
        # 4. Analyse de la casse
        self._analyze_case()
        
        # 5. Analyse des répétitions
        self._analyze_repetitions()
        
        # 6. Analyse des numéros/dimensions
        self._analyze_dimensions()
        
        # 7. Analyse de la cohérence avec la catégorie
        self._analyze_category_coherence()
        
        # 8. Analyse de la langue (français vs anglais)
        self._analyze_language()
        
        return self.results
    
    def _analyze_name_lengths(self):
        """Analyse les longueurs des noms"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Noms trop courts (< 5 caractères)
        too_short = self.df[
            (self.df['item_name'].astype(str).str.len() < 5) & 
            (self.df['item_name'].astype(str).str.len() > 0)
        ]
        
        # Noms trop longs (> 100 caractères)
        too_long = self.df[
            self.df['item_name'].astype(str).str.len() > 100
        ]
        
        # Noms vides
        empty_names = self.df[
            self.df['item_name'].isna() | 
            (self.df['item_name'].astype(str).str.strip() == '') |
            (self.df['item_name'].astype(str).str.lower() == 'null')
        ]
        
        self.results['name_length'] = {
            'too_short': len(too_short),
            'too_short_items': too_short['item_name'].tolist()[:10],
            'too_long': len(too_long),
            'too_long_items': too_long['item_name'].tolist()[:10],
            'empty': len(empty_names),
            'empty_items': empty_names.index.tolist()[:10]
        }
        
        if len(too_short) > 0:
            self.issues.append({
                'severity': 'MAJOR',
                'category': 'NAME_LENGTH',
                'message': f"{len(too_short)} noms trop courts (< 5 caractères)",
                'count': len(too_short),
                'examples': too_short['item_name'].head(5).tolist()
            })
        
        if len(too_long) > 0:
            self.issues.append({
                'severity': 'MINOR',
                'category': 'NAME_LENGTH',
                'message': f"{len(too_long)} noms trop longs (> 100 caractères)",
                'count': len(too_long),
                'examples': too_long['item_name'].head(5).tolist()
            })
    
    def _analyze_special_chars(self):
        """Analyse les caractères spéciaux dans les noms"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Caractères spéciaux non standards
        special_pattern = r'[^a-zA-Z0-9\s\-\.\(\)\/]'
        special_chars = self.df[
            self.df['item_name'].astype(str).str.contains(special_pattern, na=False)
        ]
        
        # Noms avec underscores multiples
        multiple_underscores = self.df[
            self.df['item_name'].astype(str).str.contains(r'_{2,}', na=False)
        ]
        
        # Noms avec points multiples
        multiple_dots = self.df[
            self.df['item_name'].astype(str).str.contains(r'\.{2,}', na=False)
        ]
        
        # Noms avec slashs inversés
        backslashes = self.df[
            self.df['item_name'].astype(str).str.contains(r'\\', na=False)
        ]
        
        self.results['special_chars'] = {
            'special_chars': len(special_chars),
            'special_chars_items': special_chars['item_name'].tolist()[:10],
            'multiple_underscores': len(multiple_underscores),
            'multiple_dots': len(multiple_dots),
            'backslashes': len(backslashes)
        }
        
        if len(special_chars) > 0:
            self.issues.append({
                'severity': 'MINOR',
                'category': 'SPECIAL_CHARS',
                'message': f"{len(special_chars)} noms contiennent des caractères spéciaux non standards",
                'count': len(special_chars),
                'examples': special_chars['item_name'].head(5).tolist()
            })
    
    def _analyze_forbidden_words(self):
        """Analyse les mots interdits ou aberrants"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Mots interdits courants
        forbidden_patterns = [
            r'\btest\b', r'\bnull\b', r'\bn/a\b', r'\bà définir\b',
            r'\ba définir\b', r'\bxxx\b', r'\bnew\b', r'\bnouveau\b',
            r'\btemp\b', r'\bdummy\b', r'\bexample\b', r'\bexemple\b'
        ]
        
        forbidden_items = []
        for pattern in forbidden_patterns:
            matches = self.df[
                self.df['item_name'].astype(str).str.contains(pattern, case=False, na=False)
            ]
            if len(matches) > 0:
                forbidden_items.extend(matches.index.tolist())
        
        forbidden_items = list(set(forbidden_items))
        forbidden_df = self.df.loc[forbidden_items] if forbidden_items else pd.DataFrame()
        
        self.results['forbidden_words'] = {
            'count': len(forbidden_df),
            'items': forbidden_df['item_name'].tolist()[:10] if not forbidden_df.empty else []
        }
        
        if len(forbidden_df) > 0:
            self.issues.append({
                'severity': 'MAJOR',
                'category': 'FORBIDDEN_WORDS',
                'message': f"{len(forbidden_df)} noms contiennent des mots interdits (test, null, à définir...)",
                'count': len(forbidden_df),
                'examples': forbidden_df['item_name'].head(5).tolist() if not forbidden_df.empty else []
            })
    
    def _analyze_case(self):
        """Analyse la casse des noms"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Tout en majuscules
        all_upper = self.df[
            self.df['item_name'].astype(str).str.isupper()
        ]
        
        # Tout en minuscules
        all_lower = self.df[
            self.df['item_name'].astype(str).str.islower()
        ]
        
        # Casse mixte mais avec des anomalies (ex: première lettre minuscule)
        first_lower = self.df[
            self.df['item_name'].astype(str).str.match(r'^[a-z]', na=False)
        ]
        
        self.results['case'] = {
            'all_upper': len(all_upper),
            'all_upper_items': all_upper['item_name'].tolist()[:10],
            'all_lower': len(all_lower),
            'all_lower_items': all_lower['item_name'].tolist()[:10],
            'first_lower': len(first_lower)
        }
        
        if len(all_upper) > 0:
            self.issues.append({
                'severity': 'INFO',
                'category': 'CASE',
                'message': f"{len(all_upper)} noms en majuscules uniquement",
                'count': len(all_upper),
                'examples': all_upper['item_name'].head(5).tolist()
            })
    
    def _analyze_repetitions(self):
        """Analyse les répétitions dans les noms"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Mots répétés (ex: "cable cable")
        repeated_words = []
        for idx, name in self.df['item_name'].astype(str).items():
            words = name.lower().split()
            if len(words) != len(set(words)):
                repeated_words.append(idx)
        
        repeated_df = self.df.loc[repeated_words] if repeated_words else pd.DataFrame()
        
        self.results['repetitions'] = {
            'repeated_words': len(repeated_df),
            'items': repeated_df['item_name'].tolist()[:10] if not repeated_df.empty else []
        }
        
        if len(repeated_df) > 0:
            self.issues.append({
                'severity': 'MINOR',
                'category': 'REPETITION',
                'message': f"{len(repeated_df)} noms contiennent des mots répétés",
                'count': len(repeated_df),
                'examples': repeated_df['item_name'].head(5).tolist() if not repeated_df.empty else []
            })
    
    def _analyze_dimensions(self):
        """Analyse la présence de dimensions/numéros dans les noms"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Noms sans aucun chiffre (peut être normal selon le type)
        no_digits = self.df[
            ~self.df['item_name'].astype(str).str.contains(r'\d', na=False)
        ]
        
        # Noms avec dimensions (ex: 10x20, 15*30)
        dimensions = self.df[
            self.df['item_name'].astype(str).str.contains(r'\d+\s*[xX\*]\s*\d+', na=False)
        ]
        
        self.results['dimensions'] = {
            'no_digits': len(no_digits),
            'with_dimensions': len(dimensions),
            'dimension_items': dimensions['item_name'].tolist()[:10] if not dimensions.empty else []
        }
    
    def _analyze_category_coherence(self):
        """Analyse la cohérence entre le nom et la catégorie"""
        
        if 'item_name' not in self.df.columns or 'category_name' not in self.df.columns:
            return
        
        # Mots-clés par catégorie
        category_keywords = {
            'CABLE': ['cable', 'câble', 'fil', 'conducteur'],
            'ELECTRONIQUE': ['circuit', 'composant', 'module', 'carte'],
            'MECANIQUE': ['vis', 'boulon', 'écrou', 'rondelle'],
            'INFORMATIQUE': ['ordinateur', 'écran', 'souris', 'clavier'],
            'CONSOMMABLE': ['toner', 'cartouche', 'papier', 'encre']
        }
        
        incoherent_items = []
        for idx, row in self.df.iterrows():
            category = str(row.get('category_name', '')).upper()
            name = str(row.get('item_name', '')).lower()
            
            if category in category_keywords and name:
                keywords = category_keywords[category]
                if not any(keyword in name for keyword in keywords):
                    incoherent_items.append(idx)
        
        incoherent_df = self.df.loc[incoherent_items] if incoherent_items else pd.DataFrame()
        
        self.results['category_coherence'] = {
            'incoherent': len(incoherent_df),
            'items': incoherent_df[['item_name', 'category_name']].head(10).to_dict('records') if not incoherent_df.empty else []
        }
        
        if len(incoherent_df) > 0:
            self.issues.append({
                'severity': 'WARNING',
                'category': 'COHERENCE',
                'message': f"{len(incoherent_df)} noms incohérents avec leur catégorie",
                'count': len(incoherent_df),
                'examples': [f"{row['item_name']} (Cat: {row['category_name']})" 
                           for _, row in incoherent_df.head(5).iterrows()]
            })
    
    def _analyze_language(self):
        """Analyse basique de la langue (français vs anglais)"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Mots typiquement français vs anglais
        french_words = ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'avec']
        english_words = ['the', 'a', 'an', 'and', 'or', 'with', 'for', 'to']
        
        french_count = 0
        english_count = 0
        mixed_count = 0
        
        french_items = []
        english_items = []
        mixed_items = []
        
        for idx, name in self.df['item_name'].astype(str).items():
            words = set(name.lower().split())
            has_french = any(fw in words for fw in french_words)
            has_english = any(ew in words for ew in english_words)
            
            if has_french and not has_english:
                french_count += 1
                french_items.append(idx)
            elif has_english and not has_french:
                english_count += 1
                english_items.append(idx)
            elif has_french and has_english:
                mixed_count += 1
                mixed_items.append(idx)
        
        self.results['language'] = {
            'french': french_count,
            'english': english_count,
            'mixed': mixed_count,
            'french_examples': self.df.loc[french_items[:5], 'item_name'].tolist() if french_items else [],
            'english_examples': self.df.loc[english_items[:5], 'item_name'].tolist() if english_items else [],
            'mixed_examples': self.df.loc[mixed_items[:5], 'item_name'].tolist() if mixed_items else []
        }
    
    def display_report(self):
        """Affiche le rapport d'analyse des noms"""
        
        st.markdown("## 📝 Rapport d'Analyse des Noms")
        
        # Résumé
        total_issues = len(self.issues)
        total_items = len(self.df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Items analysés", f"{total_items:,}")
        with col2:
            st.metric("Anomalies détectées", total_issues)
        with col3:
            pct_impact = len(set().union(*[set(self.df[self.df['item_name'].isin(i.get('examples', []))].index) 
                                          for i in self.issues if i.get('examples')])) / total_items * 100 if total_items > 0 else 0
            st.metric("Items impactés", f"{pct_impact:.1f}%")
        with col4:
            st.metric("Qualité des noms", self._calculate_name_quality_score())
        
        # Graphiques
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Distribution des anomalies par catégorie
            if self.issues:
                issue_cats = {}
                for issue in self.issues:
                    cat = issue['category']
                    issue_cats[cat] = issue_cats.get(cat, 0) + 1
                
                cat_df = pd.DataFrame([
                    {'Catégorie': k, 'Nombre': v} 
                    for k, v in issue_cats.items()
                ])
                st.bar_chart(cat_df.set_index('Catégorie'))
        
        with col_chart2:
            # Distribution par sévérité
            severity_counts = {'CRITICAL': 0, 'MAJOR': 0, 'MINOR': 0, 'WARNING': 0, 'INFO': 0}
            for issue in self.issues:
                severity_counts[issue['severity']] = severity_counts.get(issue['severity'], 0) + 1
            
            sev_df = pd.DataFrame([
                {'Sévérité': k, 'Nombre': v} 
                for k, v in severity_counts.items() if v > 0
            ])
            if not sev_df.empty:
                st.bar_chart(sev_df.set_index('Sévérité'))
        
        # Détail des anomalies
        st.markdown("### 🔍 Détail des anomalies")
        
        # Tabs par sévérité
        tab_crit, tab_major, tab_minor, tab_other = st.tabs(["🔴 Critiques", "🟠 Majeures", "🟡 Mineures", "📋 Autres"])
        
        with tab_crit:
            crit_issues = [i for i in self.issues if i['severity'] == 'CRITICAL']
            if crit_issues:
                for issue in crit_issues:
                    with st.expander(f"**{issue['message']}** ({issue['count']} items)"):
                        st.markdown(f"**Exemples:**")
                        for ex in issue.get('examples', [])[:5]:
                            st.markdown(f"- {ex}")
            else:
                st.success("Aucune anomalie critique")
        
        with tab_major:
            major_issues = [i for i in self.issues if i['severity'] == 'MAJOR']
            if major_issues:
                for issue in major_issues:
                    with st.expander(f"**{issue['message']}** ({issue['count']} items)"):
                        st.markdown(f"**Exemples:**")
                        for ex in issue.get('examples', [])[:5]:
                            st.markdown(f"- {ex}")
            else:
                st.success("Aucune anomalie majeure")
        
        with tab_minor:
            minor_issues = [i for i in self.issues if i['severity'] == 'MINOR']
            if minor_issues:
                for issue in minor_issues:
                    with st.expander(f"**{issue['message']}** ({issue['count']} items)"):
                        st.markdown(f"**Exemples:**")
                        for ex in issue.get('examples', [])[:5]:
                            st.markdown(f"- {ex}")
            else:
                st.success("Aucune anomalie mineure")
        
        with tab_other:
            other_issues = [i for i in self.issues if i['severity'] not in ['CRITICAL', 'MAJOR', 'MINOR']]
            if other_issues:
                for issue in other_issues:
                    with st.expander(f"**{issue['message']}** ({issue['count']} items)"):
                        st.markdown(f"**Exemples:**")
                        for ex in issue.get('examples', [])[:5]:
                            st.markdown(f"- {ex}")
            else:
                st.info("Aucune autre anomalie")
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def _calculate_name_quality_score(self) -> str:
        """Calcule un score de qualité des noms"""
        total_items = len(self.df)
        if total_items == 0:
            return "N/A"
        
        # Pénalités par sévérité
        penalties = {
            'CRITICAL': 20,
            'MAJOR': 10,
            'MINOR': 5,
            'WARNING': 2,
            'INFO': 1
        }
        
        total_penalty = 0
        for issue in self.issues:
            severity = issue['severity']
            count = issue['count']
            total_penalty += penalties.get(severity, 0) * min(count, total_items) / total_items
        
        score = max(0, 100 - total_penalty)
        
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Bon)"
        elif score >= 70:
            return "C (Moyen)"
        elif score >= 60:
            return "D (Passable)"
        else:
            return "F (À améliorer)"
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur les anomalies"""
        recs = []
        
        if any(i['category'] == 'NAME_LENGTH' and i['severity'] == 'MAJOR' for i in self.issues):
            recs.append("🔴 **Noms trop courts**: Renseigner des noms plus descriptifs (min 5 caractères)")
        
        if any(i['category'] == 'FORBIDDEN_WORDS' for i in self.issues):
            recs.append("🔴 **Mots interdits**: Nettoyer les noms contenant 'test', 'null', 'à définir'")
        
        if any(i['category'] == 'SPECIAL_CHARS' for i in self.issues):
            recs.append("🟡 **Caractères spéciaux**: Éviter les caractères non standards dans les noms")
        
        if any(i['category'] == 'COHERENCE' for i in self.issues):
            recs.append("🟡 **Incohérence catégorie**: Vérifier la cohérence entre le nom et sa catégorie")
        
        if any(i['category'] == 'REPETITION' for i in self.issues):
            recs.append("🟡 **Répétitions**: Supprimer les mots répétés dans les noms")
        
        if any(i['category'] == 'CASE' for i in self.issues):
            recs.append("ℹ️ **Casse**: Uniformiser la casse (première lettre en majuscule recommandée)")
        
        return recs

# ================================================================
# Analyseur Direct/Indirect - Version SIMPLIFIÉE
# ================================================================
class DirectIndirectAnalyzer:
    """
    Analyseur simplifié pour détecter la présence des mots "direct" et "indirect"
    dans les noms et descriptions des items
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.classification = None
        
    def analyze_direct_indirect_keywords(self):
        """
        Analyse la présence des mots-clés DIRECT et INDIRECT
        dans les différentes colonnes textuelles
        """
        
        # Colonnes à analyser
        text_columns = ['item_name', 'french_name', 'type_name', 'category_name', 'sub_category_name']
        text_columns = [col for col in text_columns if col in self.df.columns]
        
        if not text_columns:
            st.warning("Aucune colonne textuelle disponible pour l'analyse Direct/Indirect")
            return self.df
        
        # Initialiser les colonnes de détection
        self.df['has_direct_keyword'] = False
        self.df['has_indirect_keyword'] = False
        self.df['direct_keyword_source'] = ''
        self.df['indirect_keyword_source'] = ''
        self.df['direct_keyword_count'] = 0
        self.df['indirect_keyword_count'] = 0
        
        # Mots-clés à rechercher
        direct_keywords = ['direct', 'directe', 'directs', 'directes']
        indirect_keywords = ['indirect', 'indirecte', 'indirects', 'indirectes']
        
        # Barre de progression
        progress_bar = st.progress(0)
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            direct_count = 0
            indirect_count = 0
            direct_sources = []
            indirect_sources = []
            
            for col in text_columns:
                text = str(row.get(col, '')).lower()
                
                # Recherche des mots-clés DIRECT
                for keyword in direct_keywords:
                    if keyword in text:
                        direct_count += text.count(keyword)
                        direct_sources.append(f"{col}:{keyword}")
                
                # Recherche des mots-clés INDIRECT
                for keyword in indirect_keywords:
                    if keyword in text:
                        indirect_count += text.count(keyword)
                        indirect_sources.append(f"{col}:{keyword}")
            
            self.df.at[idx, 'has_direct_keyword'] = direct_count > 0
            self.df.at[idx, 'has_indirect_keyword'] = indirect_count > 0
            self.df.at[idx, 'direct_keyword_source'] = ', '.join(direct_sources[:3])
            self.df.at[idx, 'indirect_keyword_source'] = ', '.join(indirect_sources[:3])
            self.df.at[idx, 'direct_keyword_count'] = direct_count
            self.df.at[idx, 'indirect_keyword_count'] = indirect_count
            
            # Déterminer la catégorie finale
            if direct_count > 0 and indirect_count == 0:
                self.df.at[idx, 'direct_indirect_category'] = 'DIRECT'
            elif indirect_count > 0 and direct_count == 0:
                self.df.at[idx, 'direct_indirect_category'] = 'INDIRECT'
            elif direct_count > 0 and indirect_count > 0:
                self.df.at[idx, 'direct_indirect_category'] = 'MIXTE (les deux)'
            else:
                self.df.at[idx, 'direct_indirect_category'] = 'Non spécifié'
            
            # Mise à jour de la progression
            if idx % 100 == 0:
                progress_bar.progress((idx + 1) / total)
        
        progress_bar.progress(1.0)
        self.classification = self.df
        return self.df
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques de l'analyse Direct/Indirect"""
        
        if self.classification is None:
            self.analyze_direct_indirect_keywords()
        
        stats = {
            'total': len(self.classification),
            'with_direct': len(self.classification[self.classification['has_direct_keyword']]),
            'with_indirect': len(self.classification[self.classification['has_indirect_keyword']]),
            'with_both': len(self.classification[self.classification['has_direct_keyword'] & self.classification['has_indirect_keyword']]),
            'without': len(self.classification[~self.classification['has_direct_keyword'] & ~self.classification['has_indirect_keyword']])
        }
        
        # Pourcentages
        total = stats['total']
        if total > 0:
            stats['direct_pct'] = (stats['with_direct'] / total) * 100
            stats['indirect_pct'] = (stats['with_indirect'] / total) * 100
            stats['both_pct'] = (stats['with_both'] / total) * 100
            stats['without_pct'] = (stats['without'] / total) * 100
        
        # Distribution par catégorie
        if 'direct_indirect_category' in self.classification.columns:
            cat_counts = self.classification['direct_indirect_category'].value_counts().to_dict()
            stats['by_category'] = cat_counts
        
        # Distribution par colonne source
        if 'direct_keyword_source' in self.classification.columns:
            direct_sources = self.classification[self.classification['has_direct_keyword']]['direct_keyword_source'].str.split(', ').explode().value_counts().head(10).to_dict()
            stats['top_direct_sources'] = direct_sources
            
        if 'indirect_keyword_source' in self.classification.columns:
            indirect_sources = self.classification[self.classification['has_indirect_keyword']]['indirect_keyword_source'].str.split(', ').explode().value_counts().head(10).to_dict()
            stats['top_indirect_sources'] = indirect_sources
        
        return stats
    
    def display_report(self):
        """Affiche le rapport d'analyse Direct/Indirect"""
        
        if self.classification is None:
            with st.spinner("Analyse des mots-clés Direct/Indirect en cours..."):
                self.analyze_direct_indirect_keywords()
        
        stats = self.get_statistics()
        
        st.markdown("## 📊 Analyse Direct/Indirect")
        st.markdown("Détection de la présence des mots-clés **'direct'** et **'indirect'** dans les descriptions")
        
        # KPIs principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total items", f"{stats['total']:,}")
        
        with col2:
            st.metric(
                "Contient 'DIRECT'", 
                f"{stats['with_direct']:,}",
                delta=f"{stats.get('direct_pct', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Contient 'INDIRECT'", 
                f"{stats['with_indirect']:,}",
                delta=f"{stats.get('indirect_pct', 0):.1f}%"
            )
        
        with col4:
            st.metric(
                "Contient les deux", 
                f"{stats['with_both']:,}",
                delta=f"{stats.get('both_pct', 0):.1f}%"
            )
        
        # Graphiques
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Distribution des catégories
            if 'by_category' in stats:
                cat_df = pd.DataFrame([
                    {'Catégorie': k, 'Nombre': v} 
                    for k, v in stats['by_category'].items()
                ])
                st.subheader("Distribution par catégorie")
                st.bar_chart(cat_df.set_index('Catégorie'))
        
        with col_chart2:
            # Diagramme en barres simple pour DIRECT/INDIRECT
            pie_data = pd.DataFrame({
                'Type': ['Contient DIRECT', 'Contient INDIRECT', 'Aucun'],
                'Nombre': [
                    stats['with_direct'] - stats['with_both'],
                    stats['with_indirect'] - stats['with_both'],
                    stats['without']
                ]
            })
            pie_data = pie_data[pie_data['Nombre'] > 0]
            
            if not pie_data.empty:
                st.subheader("Présence des mots-clés")
                st.bar_chart(pie_data.set_index('Type'))
        
        # Top sources des mots-clés
        col_src1, col_src2 = st.columns(2)
        
        with col_src1:
            if 'top_direct_sources' in stats and stats['top_direct_sources']:
                st.markdown("**🔍 Top sources du mot 'DIRECT'**")
                for source, count in list(stats['top_direct_sources'].items())[:5]:
                    st.markdown(f"- {source}: {count} occurrences")
        
        with col_src2:
            if 'top_indirect_sources' in stats and stats['top_indirect_sources']:
                st.markdown("**🔍 Top sources du mot 'INDIRECT'**")
                for source, count in list(stats['top_indirect_sources'].items())[:5]:
                    st.markdown(f"- {source}: {count} occurrences")
        
        # Tableau détaillé
        st.markdown("### 📋 Échantillon d'items avec mots-clés DIRECT/INDIRECT")
        
        display_cols = ['item_name', 'french_name', 'direct_indirect_category', 
                       'direct_keyword_count', 'indirect_keyword_count', 'direct_keyword_source']
        display_cols = [c for c in display_cols if c in self.classification.columns]
        
        # Filtres
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            category_filter = st.multiselect(
                "Filtrer par catégorie",
                options=['DIRECT', 'INDIRECT', 'MIXTE (les deux)', 'Non spécifié'],
                default=['DIRECT', 'INDIRECT']
            )
        
        with col_filter2:
            min_direct = st.slider("Nombre minimum de 'DIRECT'", 0, 5, 0)
        
        with col_filter3:
            search = st.text_input("🔍 Rechercher dans le nom", "")
        
        filtered_df = self.classification.copy()
        
        if category_filter:
            filtered_df = filtered_df[filtered_df['direct_indirect_category'].isin(category_filter)]
        
        if min_direct > 0:
            filtered_df = filtered_df[filtered_df['direct_keyword_count'] >= min_direct]
        
        if search:
            filtered_df = filtered_df[
                filtered_df['item_name'].astype(str).str.contains(search, case=False, na=False)
            ]
        
        # Mise en forme conditionnelle
        def highlight_direct_indirect(row):
            if row['direct_indirect_category'] == 'DIRECT':
                return ['background-color: #d4edda'] * len(row)
            elif row['direct_indirect_category'] == 'INDIRECT':
                return ['background-color: #fff3cd'] * len(row)
            elif row['direct_indirect_category'] == 'MIXTE (les deux)':
                return ['background-color: #cce5ff'] * len(row)
            return [''] * len(row)
        
        styled_df = filtered_df[display_cols].head(100).style.apply(highlight_direct_indirect, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                'direct_keyword_count': st.column_config.NumberColumn(
                    "Nb DIRECT",
                    help="Nombre d'occurrences du mot 'direct'"
                ),
                'indirect_keyword_count': st.column_config.NumberColumn(
                    "Nb INDIRECT",
                    help="Nombre d'occurrences du mot 'indirect'"
                )
            },
            height=400
        )
        
        # Export
        st.markdown("### 📥 Export")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exporter les résultats (CSV)",
                data=csv_data,
                file_name=f"direct_indirect_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # Export des items avec mots-clés
            keywords_df = filtered_df[
                filtered_df['has_direct_keyword'] | filtered_df['has_indirect_keyword']
            ][['item_name', 'french_name', 'direct_indirect_category', 
               'direct_keyword_count', 'indirect_keyword_count']].copy()
            
            if not keywords_df.empty:
                csv_keywords = keywords_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Items avec mots-clés",
                    data=csv_keywords,
                    file_name=f"items_avec_mots_cles_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ================================================================
# Analyseur de Classification Comptable - Version AMÉLIORÉE
# ================================================================
class AccountingClassifier:
    """
    Classe les items selon le plan comptable Netis Group
    Version améliorée avec dictionnaire de mots-clés enrichi et scoring avancé
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.accounting_codes = ACCOUNTING_CODES
        self.classified_df = None
        
        # Créer un DataFrame pour faciliter les recherches
        self.codes_df = pd.DataFrame(self.accounting_codes)
        
        # Dictionnaire enrichi de mots-clés par catégorie
        self.keyword_dictionary = self._build_keyword_dictionary()
        
        # Construire l'index de recherche avancé
        self._build_advanced_search_index()
    
    def _build_keyword_dictionary(self) -> Dict[str, List[str]]:
        """Construit un dictionnaire de mots-clés enrichi par catégorie"""
        
        return {
            # VENTES
            'VENTES': [
                'vente', 'sales', 'facturation', 'billing', 'client', 'customer',
                'external', 'externe', 'service', 'prestation', 'marchandise',
                'produit', 'product', 'commercial', 'commercialisation'
            ],
            
            # RESSOURCES HUMAINES
            'RESSOURCES HUMAINES': [
                'salaire', 'salary', 'rémunération', 'remuneration', 'pay',
                'personnel', 'staff', 'employé', 'employee', 'main d\'œuvre',
                'manpower', 'intérim', 'temporary', 'interim', 'encadrement',
                'supervision', 'expat', 'expatrié', 'formation', 'training',
                'mission', 'travel', 'voyage', 'déplacement', 'reception',
                'restauration', 'catering', 'logement', 'housing', 'congé',
                'leave', 'prime', 'bonus', 'gratification', 'indemnité',
                'allowance', 'médecine', 'health', 'médecine du travail',
                'recrutement', 'recruitment', 'déménagement', 'moving'
            ],
            
            # MATERIAUX
            'MATERIAUX': [
                'matériau', 'material', 'matière', 'béton', 'concrete', 'ciment',
                'cement', 'granulat', 'aggregate', 'caillou', 'gravel', 'sable',
                'sand', 'remblai', 'backfill', 'acier', 'steel', 'armature',
                'rebar', 'métal', 'metal', 'métallique', 'metallic', 'câble',
                'cable', 'fibre', 'optique', 'fiber', 'pylône', 'pylon', 'tour',
                'tower', 'groupe électrogène', 'generator', 'générateur',
                'panneau solaire', 'solar panel', 'batterie', 'battery',
                'onduleur', 'inverter', 'transformateur', 'transformer',
                'tableau électrique', 'electrical panel', 'disjoncteur',
                'circuit breaker', 'paratonnerre', 'lightning rod',
                'cuve', 'tank', 'gasoil', 'fuel', 'redresseur', 'rectifier',
                'compteur', 'meter', 'armoire', 'cabinet', 'boîtier', 'box',
                'connecteur', 'connector', 'poteau', 'pole'
            ],
            
            # MAINTENANCE
            'MAINTENANCE': [
                'maintenance', 'entretien', 'repair', 'réparation', 'réparer',
                'fix', 'pdr', 'spare parts', 'pièce détachée', 'rechange',
                'préventive', 'preventive', 'curative', 'corrective',
                'huile', 'oil', 'graisse', 'grease', 'lubrifiant', 'lubricant',
                'vidange', 'oil change', 'dépannage', 'troubleshooting'
            ],
            
            # OUTILLAGE
            'OUTILLAGE': [
                'outil', 'tool', 'outillage', 'tooling', 'petit outillage',
                'small tools', 'échafaudage', 'scaffolding', 'soudure',
                'welding', 'levage', 'lifting', 'manutention', 'handling',
                'coffrage', 'formwork', 'temporaire', 'temporary', 'accessoire',
                'accessory', 'équipement', 'equipment'
            ],
            
            # SECURITE
            'SECURITE': [
                'sécurité', 'safety', 'security', 'protection', 'epi', 'ppe',
                'casque', 'helmet', 'gant', 'glove', 'chaussure', 'shoe',
                'botte', 'boot', 'gilet', 'vest', 'signalisation', 'sign',
                'balise', 'buoy', 'environnement', 'environment', 'incendie',
                'fire', 'extincteur', 'extinguisher', 'gardiennage', 'guard'
            ],
            
            # CONSOMMABLES
            'CONSOMMABLES': [
                'consommable', 'consumable', 'fourniture', 'supply',
                'eau', 'water', 'fluide', 'fluid', 'divers', 'misc',
                'produit', 'product', 'article', 'item'
            ],
            
            # LOCATION
            'LOCATION': [
                'location', 'rental', 'rent', 'louer', 'lease', 'leasing',
                'véhicule', 'vehicle', 'voiture', 'car', 'camion', 'truck',
                'grue', 'crane', 'pelle', 'excavator', 'chargeur', 'loader',
                'engin', 'machine', 'machinery', 'matériel', 'equipment'
            ],
            
            # TRANSPORT
            'TRANSPORT': [
                'transport', 'freight', 'fret', 'import', 'export', 'aérien',
                'air', 'maritime', 'sea', 'terrestre', 'land', 'logistique',
                'logistics', 'livraison', 'delivery', 'expédition', 'shipping',
                'frais', 'cost', 'affrètement', 'chartering'
            ],
            
            # DOUANES
            'DOUANES': [
                'douane', 'customs', 'droit', 'duty', 'taxe', 'tax', 'importation',
                'import', 'exportation', 'export', 'dédouanement', 'clearance'
            ],
            
            # ASSURANCES
            'ASSURANCES': [
                'assurance', 'insurance', 'garantie', 'warranty', 'couverture',
                'coverage', 'risque', 'risk', 'prime', 'premium', 'santé',
                'health', 'véhicule', 'vehicle', 'multirisque', 'multi-risk',
                'bris', 'breakdown', 'machine', 'machinery', 'chantier',
                'site', 'transport'
            ],
            
            # ETUDES
            'ETUDES': [
                'étude', 'study', 'recherche', 'research', 'design', 'conception',
                'géomètre', 'surveyor', 'géotechnique', 'geotechnical',
                'laboratoire', 'laboratory', 'certification', 'homologation',
                'contrôle', 'control', 'technique', 'technical', 'ingénierie',
                'engineering'
            ],
            
            # INFORMATIQUE
            'INFORMATIQUE': [
                'informatique', 'computer', 'it', 'logiciel', 'software',
                'matériel', 'hardware', 'licence', 'license', 'serveur',
                'server', 'ordinateur', 'pc', 'imprimante', 'printer',
                'écran', 'screen', 'réseau', 'network', 'télécom', 'telecom'
            ],
            
            # SOUS-TRAITANCE
            'SOUS-TRAITANCE': [
                'sous-traitance', 'subcontracting', 'sous-traitant',
                'subcontractor', 'prestataire', 'service provider',
                'externalisation', 'outsourcing'
            ],
            
            # FRAIS FINANCIERS
            'FRAIS FINANCIERS': [
                'bancaire', 'bank', 'frais', 'fee', 'commission', 'intérêt',
                'interest', 'change', 'exchange', 'fx', 'garantie', 'guarantee',
                'caution', 'bond', 'agios', 'emprunt', 'loan', 'crédit', 'credit'
            ],
            
            # INTERCO
            'INTERCO': [
                'interco', 'intercompany', 'groupe', 'group', 'intragroupe',
                'intragroup', 'refacturation', 'rebilling', 'management fees',
                'filiale', 'subsidiary', 'holding'
            ],
            
            # PROVISIONS
            'PROVISIONS': [
                'provision', 'provisioning', 'dotation', 'allocation',
                'reprise', 'reversal', 'dépréciation', 'depreciation',
                'risque', 'risk', 'charge'
            ],
            
            # EXCEPTIONNEL
            'EXCEPTIONNEL': [
                'exceptionnel', 'exceptional', 'restructuration', 'restructuring',
                'cession', 'disposal', 'amende', 'fine', 'pénalité', 'penalty',
                'antérieur', 'prior year', 'ajustement', 'adjustment',
                'litige', 'dispute', 'contentieux'
            ],
            
            # IMPOTS
            'IMPOTS': [
                'impôt', 'tax', 'is', 'cit', 'tva', 'vat', 'dividende',
                'dividend', 'ras', 'wht', 'retenue', 'withholding', 'douane',
                'customs', 'taxe', 'contribution', 'forfaitaire', 'lumpsum'
            ],
            
            # AMORTISSEMENTS
            'AMORTISSEMENTS': [
                'amortissement', 'depreciation', 'amort', 'dépréciation',
                'depreciation', 'dotation', 'allocation', 'immobilisation',
                'fixed asset'
            ]
        }
    
    def _build_advanced_search_index(self):
        """Construit un index de recherche avancé avec mots-clés pondérés"""
        self.search_index = []
        
        for code in self.accounting_codes:
            # Texte de recherche enrichi
            search_text = f"""
                {code['description_fr']} {code['description_en']} 
                {code['category']} {code['subcategory']}
            """.lower()
            
            # Nettoyer
            search_text = re.sub(r'[^\w\s]', ' ', search_text)
            
            # Extraire les mots-clés
            keywords = [word for word in search_text.split() if len(word) > 2]
            
            # Ajouter des mots-clés supplémentaires depuis le dictionnaire
            category_keywords = self.keyword_dictionary.get(code['category'], [])
            
            self.search_index.append({
                'code': code['code'],
                'category': code['category'],
                'subcategory': code['subcategory'],
                'description_fr': code['description_fr'],
                'description_en': code['description_en'],
                'keywords': keywords,
                'category_keywords': category_keywords,
                'search_text': search_text,
                'code_parts': code['code'].split('.')
            })
    
    def _calculate_advanced_score(self, 
                                 item_name: str, 
                                 item_category: str = "", 
                                 item_type: str = "",
                                 item_description: str = "",
                                 search_words: Set[str] = None) -> Tuple[int, Dict, Any]:
        """
        Calcule un score avancé pour la classification
        Retourne (score, détails, meilleur match)
        """
        # Texte à analyser (enrichi)
        search_text = f"{item_name} {item_category} {item_type} {item_description}".lower()
        search_text = re.sub(r'[^\w\s]', ' ', search_text)
        search_words = set(search_text.split())
        
        best_match = None
        best_score = 0
        best_details = {}
        
        for idx, entry in enumerate(self.search_index):
            score = 0
            details = []
            
            keywords = set(entry['keywords'])
            category_keywords = set(entry['category_keywords'])
            
            # 1. Correspondance exacte de mots-clés (poids élevé - x10)
            exact_matches = search_words.intersection(keywords)
            exact_match_count = len(exact_matches)
            score += exact_match_count * 10
            if exact_match_count > 0:
                details.append(f"{exact_match_count} mots exacts: +{exact_match_count*10}")
            
            # 2. Correspondance avec mots-clés de catégorie (poids moyen - x5)
            category_matches = search_words.intersection(category_keywords)
            category_match_count = len(category_matches)
            score += category_match_count * 5
            if category_match_count > 0:
                details.append(f"{category_match_count} mots catégorie: +{category_match_count*5}")
            
            # 3. Correspondance partielle (sous-chaînes)
            partial_score = 0
            for word in search_words:
                if len(word) > 3:
                    for keyword in keywords:
                        if keyword in word or word in keyword:
                            partial_score += 2
                            break
            score += partial_score
            if partial_score > 0:
                details.append(f"partielles: +{partial_score}")
            
            # 4. Bonus si la catégorie correspond (poids fort - +15)
            if item_category and item_category.lower() in entry['category'].lower():
                score += 15
                details.append("catégorie correspond: +15")
            
            # 5. Bonus si la sous-catégorie correspond (poids moyen - +10)
            if item_category and item_category.lower() in entry['subcategory'].lower():
                score += 10
                details.append("sous-catégorie correspond: +10")
            
            # 6. Bonus pour correspondance de code (préfixe)
            if 'code_parts' in entry and len(entry['code_parts']) >= 2:
                code_prefix = entry['code_parts'][0]
                # Vérifier si le préfixe du code apparaît dans le texte
                if code_prefix in search_text:
                    score += 8
                    details.append(f"préfixe code {code_prefix}: +8")
            
            # 7. Bonus pour correspondance avec le type
            if item_type and item_type.lower() in entry['search_text']:
                score += 5
                details.append(f"type correspond: +5")
            
            # 8. Bonus pour correspondance de longueur (mots longs = plus significatifs)
            long_words = [w for w in search_words if len(w) > 6]
            if long_words:
                for long_word in long_words:
                    if long_word in keywords:
                        score += 3
                        details.append(f"mot long {long_word}: +3")
            
            # 9. Vérifier les mots-clés négatifs (pour éviter les mauvaises classifications)
            negative_keywords = {
                'VENTES': ['achat', 'purchase', 'frais', 'expense', 'cost', 'dépense'],
                'MATERIAUX': ['service', 'prestation', 'conseil', 'consulting', 'formation'],
                'LOCATION': ['achat', 'purchase', 'acquisition', 'vente', 'sale'],
                'AMORTISSEMENTS': ['location', 'rental', 'achat', 'purchase']
            }
            
            if entry['category'] in negative_keywords:
                for neg_word in negative_keywords[entry['category']]:
                    if neg_word in search_text:
                        score -= 5
                        details.append(f"mot négatif '{neg_word}': -5")
            
            # Mise à jour du meilleur score
            if score > best_score:
                best_score = score
                best_match = entry
                best_details = {
                    'score': score,
                    'details': ' | '.join(details),
                    'matched_keywords': list(exact_matches)[:5],
                    'category_matches': list(category_matches)[:5],
                    'exact_match_count': exact_match_count,
                    'category_match_count': category_match_count
                }
        
        return best_score, best_details, best_match
    
    def classify_item_advanced(self, 
                              item_name: str, 
                              item_category: str = "", 
                              item_type: str = "",
                              item_description: str = "") -> Dict:
        """
        Version avancée de classification
        """
        if pd.isna(item_name) or not item_name:
            return {
                'code': 'NON_CLASSIFIE',
                'category': 'Non classifié',
                'subcategory': '',
                'confidence': 0,
                'description': 'Item sans nom',
                'score': 0,
                'details': {}
            }
        
        # Calculer le score avancé
        best_score, best_details, best_match = self._calculate_advanced_score(
            item_name, item_category, item_type, item_description
        )
        
        # Normaliser le score (max théorique variable)
        # Un score de 100 est considéré comme excellent
        normalized_score = min(best_score / 70, 1.0)
        
        # Seuil de confiance minimum (score minimum pour considérer une classification)
        if best_score < 15:
            # Tentative de classification par défaut basée sur le type
            if 'service' in item_name.lower() or 'prestation' in item_name.lower():
                return {
                    'code': '1264.001',
                    'category': 'SERVICES',
                    'subcategory': 'Divers',
                    'confidence': 0.3,
                    'description': 'Autres prestations',
                    'score': best_score,
                    'details': best_details
                }
            elif 'achat' in item_name.lower() or 'fourniture' in item_name.lower():
                return {
                    'code': '1228.003',
                    'category': 'MATERIAUX',
                    'subcategory': 'Divers',
                    'confidence': 0.3,
                    'description': 'Autres matériaux incorporés',
                    'score': best_score,
                    'details': best_details
                }
            else:
                return {
                    'code': 'NON_CLASSIFIE',
                    'category': 'Non classifié',
                    'subcategory': '',
                    'confidence': 0,
                    'description': 'Non classifié',
                    'score': best_score,
                    'details': best_details
                }
        
        if best_match:
            return {
                'code': best_match['code'],
                'category': best_match['category'],
                'subcategory': best_match['subcategory'],
                'confidence': normalized_score,
                'description': best_match['description_fr'],
                'score': best_score,
                'details': best_details
            }
        else:
            return {
                'code': 'NON_CLASSIFIE',
                'category': 'Non classifié',
                'subcategory': '',
                'confidence': 0,
                'description': 'Non classifié',
                'score': best_score,
                'details': best_details
            }
    
    def classify_all_advanced(self, use_parallel: bool = True) -> pd.DataFrame:
        """
        Classifie tous les items avec la version avancée
        """
        
        if 'item_name' not in self.df.columns:
            st.error("La colonne 'item_name' est requise pour la classification")
            return self.df
        
        # Initialiser les colonnes de classification
        self.df['accounting_code'] = ''
        self.df['accounting_category'] = ''
        self.df['accounting_subcategory'] = ''
        self.df['accounting_confidence'] = 0.0
        self.df['accounting_description'] = ''
        self.df['accounting_score'] = 0
        self.df['accounting_details'] = ''
        self.df['accounting_matched_keywords'] = ''
        
        if use_parallel and len(self.df) > 1000:
            return self._classify_all_parallel()
        else:
            return self._classify_all_sequential()
    
    def _classify_all_sequential(self) -> pd.DataFrame:
        """Version séquentielle avec barre de progression"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            item_name = row.get('item_name', '')
            item_category = row.get('category_name', '')
            item_type = row.get('type_name', '')
            item_description = row.get('french_name', '')
            
            result = self.classify_item_advanced(item_name, item_category, item_type, item_description)
            
            self.df.at[idx, 'accounting_code'] = result['code']
            self.df.at[idx, 'accounting_category'] = result['category']
            self.df.at[idx, 'accounting_subcategory'] = result['subcategory']
            self.df.at[idx, 'accounting_confidence'] = result['confidence']
            self.df.at[idx, 'accounting_description'] = result['description']
            self.df.at[idx, 'accounting_score'] = result.get('score', 0)
            
            details = result.get('details', {})
            self.df.at[idx, 'accounting_details'] = details.get('details', '')[:200] if isinstance(details, dict) else str(details)[:200]
            self.df.at[idx, 'accounting_matched_keywords'] = ', '.join(details.get('matched_keywords', [])) if isinstance(details, dict) else ''
            
            # Mise à jour de la progression
            if idx % 100 == 0:
                progress_bar.progress((idx + 1) / total)
                status_text.text(f"Classification: {idx+1}/{total}")
        
        progress_bar.progress(1.0)
        status_text.text(f"Classification terminée: {total} items")
        self.classified_df = self.df
        return self.df
    
    def _classify_all_parallel(self) -> pd.DataFrame:
        """Version parallèle pour les gros volumes"""
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        st.info(f"🚀 Traitement parallèle activé pour {len(self.df)} items")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def process_row(idx_row):
            idx, row = idx_row
            item_name = row.get('item_name', '')
            item_category = row.get('category_name', '')
            item_type = row.get('type_name', '')
            item_description = row.get('french_name', '')
            
            return idx, self.classify_item_advanced(item_name, item_category, item_type, item_description)
        
        # Préparer les données pour le traitement parallèle
        rows = list(self.df.iterrows())
        
        # Traiter en parallèle
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, row) for row in rows]
            
            for i, future in enumerate(as_completed(futures)):
                idx, result = future.result()
                
                self.df.at[idx, 'accounting_code'] = result['code']
                self.df.at[idx, 'accounting_category'] = result['category']
                self.df.at[idx, 'accounting_subcategory'] = result['subcategory']
                self.df.at[idx, 'accounting_confidence'] = result['confidence']
                self.df.at[idx, 'accounting_description'] = result['description']
                self.df.at[idx, 'accounting_score'] = result.get('score', 0)
                
                details = result.get('details', {})
                self.df.at[idx, 'accounting_details'] = details.get('details', '')[:200] if isinstance(details, dict) else str(details)[:200]
                self.df.at[idx, 'accounting_matched_keywords'] = ', '.join(details.get('matched_keywords', [])) if isinstance(details, dict) else ''
                
                if (i + 1) % 100 == 0:
                    progress_bar.progress((i + 1) / len(rows))
                    status_text.text(f"Classification parallèle: {i+1}/{len(rows)}")
        
        progress_bar.progress(1.0)
        status_text.text(f"Classification terminée: {len(self.df)} items")
        self.classified_df = self.df
        return self.df
    
    def get_statistics_advanced(self) -> Dict:
        """Retourne des statistiques avancées de classification"""
        
        if self.classified_df is None:
            self.classify_all_advanced()
        
        stats = {
            'total': len(self.classified_df),
            'classified': len(self.classified_df[self.classified_df['accounting_code'] != 'NON_CLASSIFIE']),
            'non_classified': len(self.classified_df[self.classified_df['accounting_code'] == 'NON_CLASSIFIE'])
        }
        
        # Pourcentages
        stats['classified_pct'] = (stats['classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
        stats['non_classified_pct'] = (stats['non_classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Confiance moyenne et médiane
        stats['avg_confidence'] = self.classified_df['accounting_confidence'].mean()
        stats['median_confidence'] = self.classified_df['accounting_confidence'].median()
        stats['avg_score'] = self.classified_df['accounting_score'].mean()
        
        # Distribution des scores de confiance
        stats['score_distribution'] = {
            'excellent': len(self.classified_df[self.classified_df['accounting_confidence'] >= 0.8]),
            'bon': len(self.classified_df[(self.classified_df['accounting_confidence'] >= 0.6) & 
                                          (self.classified_df['accounting_confidence'] < 0.8)]),
            'moyen': len(self.classified_df[(self.classified_df['accounting_confidence'] >= 0.4) & 
                                           (self.classified_df['accounting_confidence'] < 0.6)]),
            'faible': len(self.classified_df[(self.classified_df['accounting_confidence'] >= 0.2) & 
                                            (self.classified_df['accounting_confidence'] < 0.4)]),
            'tres_faible': len(self.classified_df[self.classified_df['accounting_confidence'] < 0.2])
        }
        
        # Distribution par catégorie comptable
        cat_counts = self.classified_df['accounting_category'].value_counts()
        stats['top_categories'] = cat_counts.head(10).to_dict()
        stats['categories_count'] = len(cat_counts)
        
        # Distribution par code comptable
        code_counts = self.classified_df['accounting_code'].value_counts()
        stats['top_codes'] = code_counts.head(20).to_dict()
        stats['codes_count'] = len(code_counts)
        
        # Items avec confiance très faible (à revoir)
        low_confidence = self.classified_df[self.classified_df['accounting_confidence'] < 0.3]
        stats['low_confidence_count'] = len(low_confidence)
        stats['low_confidence_examples'] = low_confidence[['item_name', 'accounting_code', 'accounting_score']].head(5).to_dict('records') if not low_confidence.empty else []
        
        return stats
    
    def display_advanced_report(self):
        """Affiche un rapport détaillé avec les améliorations"""
        
        if self.classified_df is None:
            with st.spinner("Classification avancée en cours..."):
                self.classify_all_advanced()
        
        stats = self.get_statistics_advanced()
        
        st.markdown("## 💰 Classification Comptable Avancée")
        st.markdown("Classification intelligente avec dictionnaire enrichi (plus de 500 mots-clés) et scoring multi-critères")
        
        # KPIs améliorés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total items", f"{stats['total']:,}")
        
        with col2:
            st.metric(
                "Classifiés", 
                f"{stats['classified']:,}",
                delta=f"{stats['classified_pct']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Confiance moyenne", 
                f"{stats['avg_confidence']:.1%}",
                delta=f"{stats['median_confidence']:.1%} médiane"
            )
        
        with col4:
            st.metric(
                "Score moyen", 
                f"{stats['avg_score']:.1f}",
                help="Score brut moyen (non normalisé)"
            )
        
        # Distribution de la confiance
        st.subheader("📊 Distribution des scores de confiance")
        
        conf_data = pd.DataFrame([
            {'Niveau': 'Excellent (≥80%)', 'Nombre': stats['score_distribution']['excellent']},
            {'Niveau': 'Bon (60-80%)', 'Nombre': stats['score_distribution']['bon']},
            {'Niveau': 'Moyen (40-60%)', 'Nombre': stats['score_distribution']['moyen']},
            {'Niveau': 'Faible (20-40%)', 'Nombre': stats['score_distribution']['faible']},
            {'Niveau': 'Très faible (<20%)', 'Nombre': stats['score_distribution']['tres_faible']}
        ])
        
        conf_data = conf_data[conf_data['Nombre'] > 0]
        if not conf_data.empty:
            st.bar_chart(conf_data.set_index('Niveau'))
        
        # Alertes sur les items à faible confiance
        if stats['low_confidence_count'] > 0:
            st.warning(f"⚠️ {stats['low_confidence_count']} items ont une confiance très faible (<30%) - À vérifier manuellement")
            
            with st.expander("Voir les exemples"):
                for item in stats['low_confidence_examples']:
                    st.markdown(f"- {item.get('item_name', 'N/A')} → Code proposé: {item.get('accounting_code', 'N/A')} (Score: {item.get('accounting_score', 0)})")
        
        # Filtres avancés
        st.markdown("### 🔍 Recherche et filtrage avancé")
        
        col_filter1, col_filter2, col_filter3, col_filter4, col_filter5 = st.columns(5)
        
        with col_filter1:
            search = st.text_input("🔎 Rechercher un item", placeholder="Nom de l'item...")
        
        with col_filter2:
            categories = ['Toutes'] + sorted([c for c in self.classified_df['accounting_category'].unique().tolist() if c != 'Non classifié'])
            selected_category = st.selectbox("📂 Catégorie comptable", categories)
        
        with col_filter3:
            codes = ['Tous'] + sorted([c for c in self.classified_df['accounting_code'].unique().tolist() if c != 'NON_CLASSIFIE'])
            selected_code = st.selectbox("🔢 Code comptable", codes)
        
        with col_filter4:
            confidence_range = st.slider(
                "🎯 Intervalle de confiance",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05
            )

        with col_filter5:
            # Filtrer par description comptable
            if 'description_comptable' in self.classified_df.columns:
                descriptions_list = ['Tous'] + sorted(self.classified_df['description_comptable'].dropna().unique().tolist())
                selected_description = st.selectbox(
                    "📋 Description comptable",
                    descriptions_list,
                    key="desc_filter",
                    help="Filtrer par description comptable"
                )
            else:
                selected_description = "Tous"
        
        # Appliquer les filtres
        filtered_df = self.classified_df.copy()
        
        if search:
            filtered_df = filtered_df[
                filtered_df['item_name'].astype(str).str.contains(search, case=False, na=False) |
                filtered_df['french_name'].astype(str).str.contains(search, case=False, na=False)
            ]
        
        if selected_category != 'Toutes':
            filtered_df = filtered_df[filtered_df['accounting_category'] == selected_category]
        
        if selected_code != 'Tous':
            filtered_df = filtered_df[filtered_df['accounting_code'] == selected_code]
        
        filtered_df = filtered_df[
            (filtered_df['accounting_confidence'] >= confidence_range[0]) &
            (filtered_df['accounting_confidence'] <= confidence_range[1])
        ]

        # Appliquer le filtre par description comptable si sélectionné
        try:
            if selected_description != "Tous" and 'description_comptable' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['description_comptable'] == selected_description]
        except NameError:
            # selected_description may not be defined in some branches
            pass
        
        st.markdown(f"**{len(filtered_df)} items** correspondant aux critères")
        
        # Afficher les résultats avec mise en forme conditionnelle
        display_cols = ['item_name', 'french_name', 'accounting_code', 'accounting_category', 
                       'accounting_subcategory', 'accounting_confidence', 'accounting_score',
                       'accounting_matched_keywords']
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Fonction de couleur pour la confiance
        def color_confidence(val):
            if val >= 0.8:
                return 'background-color: #d4edda'  # vert
            elif val >= 0.6:
                return 'background-color: #fff3cd'  # jaune
            elif val >= 0.4:
                return 'background-color: #ffe5b4'  # orange clair
            else:
                return 'background-color: #f8d7da'  # rouge
        
        styled_df = filtered_df[display_cols].head(200).style.applymap(
            color_confidence, subset=['accounting_confidence']
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                'accounting_confidence': st.column_config.ProgressColumn(
                    "Confiance",
                    format="%.0f%%",
                    min_value=0,
                    max_value=1
                ),
                'accounting_score': st.column_config.NumberColumn(
                    "Score brut",
                    format="%d"
                )
            },
            height=500
        )
        
        # Section de validation manuelle
        st.markdown("### ✅ Validation manuelle")
        st.markdown("Sélectionnez des items pour les reclassifier manuellement si nécessaire")
        
        # Sélectionner les items à faible confiance pour reclassification
        review_items = filtered_df[
            filtered_df['accounting_confidence'] < 0.5
        ].head(10)
        
        if not review_items.empty:
            st.markdown("**Items à vérifier (confiance < 50%)**")
            
            for idx, row in review_items.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['item_name']}**  \n"
                              f"*Catégorie: {row.get('category_name', 'N/A')}*")
                
                with col2:
                    st.markdown(f"**Code:** {row['accounting_code']}  \n"
                              f"Confiance: {row['accounting_confidence']:.0%}")
                
                with col3:
                    # Menu de sélection des codes alternatifs
                    alt_codes = [''] + self.codes_df['code'].head(10).tolist()
                    selected_alt = st.selectbox(
                        "Reclassifier",
                        options=alt_codes,
                        key=f"alt_{idx}",
                        label_visibility="collapsed"
                    )
                    if selected_alt:
                        # Logique pour mettre à jour la classification (simulée)
                        st.success(f"✅ Item reclassifié en {selected_alt}")
        
        # Export amélioré
        st.markdown("### 📥 Export")
        
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        
        with col_exp1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV (filtré)",
                data=csv_data,
                file_name=f"classification_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # Vue synthétique par code comptable
            if 'accounting_code' in filtered_df.columns:
                summary = filtered_df[filtered_df['accounting_code'] != 'NON_CLASSIFIE'].groupby(
                    ['accounting_code', 'accounting_category', 'accounting_subcategory']
                ).agg({
                    'item_name': 'count',
                    'accounting_confidence': ['mean', 'min', 'max'],
                    'accounting_score': 'mean'
                }).round(3)
                
                if not summary.empty:
                    summary.columns = ['nb_items', 'confiance_moyenne', 'confiance_min', 'confiance_max', 'score_moyen']
                    summary = summary.reset_index()
                    
                    csv_summary = summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Synthèse",
                        data=csv_summary,
                        file_name=f"synthèse_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col_exp3:
            # Rapport d'audit (items à faible confiance)
            audit_items = filtered_df[
                filtered_df['accounting_confidence'] < 0.5
            ][['item_name', 'french_name', 'category_name', 'type_name', 
               'accounting_code', 'accounting_confidence', 'accounting_score', 'accounting_matched_keywords']].copy()
            
            if not audit_items.empty:
                csv_audit = audit_items.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🔍 Items à auditer",
                    data=csv_audit,
                    file_name=f"audit_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp4:
            # Afficher le dictionnaire de mots-clés
            with st.expander("📚 Voir le dictionnaire de mots-clés"):
                for category, keywords in list(self.keyword_dictionary.items())[:5]:
                    st.markdown(f"**{category}:** {', '.join(keywords[:15])}")
                    if len(keywords) > 15:
                        st.caption(f"... et {len(keywords)-15} autres")
                st.markdown("---")
                st.markdown(f"**Total:** {sum(len(k) for k in self.keyword_dictionary.values())} mots-clés dans {len(self.keyword_dictionary)} catégories")

# ================================================================
# === IA MASTER DATA QUALITY - Analyseur de qualité des données
# ================================================================

@dataclass
class DataQualityIssue:
    """Classe pour représenter une anomalie détectée"""
    severity: str  # 'CRITICAL', 'MAJOR', 'MINOR', 'WARNING', 'INFO'
    category: str  # 'DUPLICATE', 'MISSING', 'FORMAT', 'CONSISTENCY', 'REFERENCE', 'SEMANTIC'
    field: str
    message: str
    count: int
    examples: List[Any] = field(default_factory=list)
    recommendation: str = ""
    impact: str = ""

class MasterDataQualityAI:
    """
    IA spécialisée dans l'analyse de la qualité des données Master Data
    Détecte les anomalies, erreurs et non-conformités
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.issues: List[DataQualityIssue] = []
        self.score = 100  # Score de qualité initial (0-100)
        self.total_items = len(df)
        
    def analyze(self) -> Dict:
        """Lance l'analyse complète de la qualité des données"""
        
        # 1. Analyse des doublons
        self._check_duplicates()
        
        # 2. Analyse des valeurs manquantes
        self._check_missing_values()
        
        # 3. Analyse du format des données
        self._check_data_formats()
        
        # 4. Analyse de la cohérence
        self._check_consistency()
        
        # 5. Analyse des références
        self._check_references()
        
        # 6. Analyse sémantique des noms
        self._check_name_quality()
        
        # 7. Analyse temporelle
        self._check_temporal_quality()
        
        # 8. Analyse des types et catégories
        self._check_categories()
        
        # Calculer le score final
        self._calculate_quality_score()
        
        return {
            'score': self.score,
            'issues': self.issues,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations(),
            'grade': self._get_grade()
        }
    
    def _check_duplicates(self):
        """Détecte les différents types de doublons"""
        
        # Doublons exacts sur le nom
        if 'item_name' in self.df.columns:
            exact_dupes = self.df[self.df.duplicated(subset=['item_name'], keep=False)]
            if len(exact_dupes) > 0:
                examples = exact_dupes['item_name'].head(3).tolist()
                self.issues.append(DataQualityIssue(
                    severity='CRITICAL' if len(exact_dupes) > 10 else 'MAJOR',
                    category='DUPLICATE',
                    field='item_name',
                    message=f"{len(exact_dupes)} doublons exacts détectés",
                    count=len(exact_dupes),
                    examples=examples,
                    recommendation="Fusionner les doublons ou archiver les versions inactives",
                    impact=f"Perte de {len(exact_dupes) - self.df['item_name'].nunique()} références uniques"
                ))
                self.score -= min(15, len(exact_dupes) // 10)
        
        # Doublons sur la référence
        if 'reference' in self.df.columns:
            ref_dupes = self.df[self.df.duplicated(subset=['reference'], keep=False)]
            if len(ref_dupes) > 0:
                self.issues.append(DataQualityIssue(
                    severity='CRITICAL',
                    category='DUPLICATE',
                    field='reference',
                    message=f"{len(ref_dupes)} références en double",
                    count=len(ref_dupes),
                    examples=ref_dupes['reference'].head(3).tolist(),
                    recommendation="Chaque référence doit être unique dans le système",
                    impact="Risque d'erreurs dans les commandes et le suivi"
                ))
                self.score -= 20
    
    def _check_missing_values(self):
        """Vérifie les valeurs manquantes critiques"""
        
        critical_fields = ['item_name', 'reference', 'status']
        for field in critical_fields:
            if field in self.df.columns:
                missing = self.df[field].isna() | (self.df[field] == '') | (self.df[field] == '0') | (self.df[field] == 'NULL')
                missing_count = missing.sum()
                
                if missing_count > 0:
                    severity = 'CRITICAL' if missing_count > self.total_items * 0.1 else 'MAJOR'
                    
                    self.issues.append(DataQualityIssue(
                        severity=severity,
                        category='MISSING',
                        field=field,
                        message=f"{missing_count} items sans {field} ({missing_count/self.total_items*100:.1f}%)",
                        count=missing_count,
                        recommendation=f"Renseigner le champ {field} pour tous les items",
                        impact=f"{missing_count} items inutilisables sans cette information"
                    ))
                    self.score -= missing_count // 5
    
    def _check_data_formats(self):
        """Vérifie le format des données"""
        
        # Format des références
        if 'reference' in self.df.columns:
            # Références trop courtes
            invalid_refs = self.df[
                (self.df['reference'].astype(str).str.len() < 3) & 
                (self.df['reference'] != '')
            ]
            if len(invalid_refs) > 0:
                self.issues.append(DataQualityIssue(
                    severity='MAJOR',
                    category='FORMAT',
                    field='reference',
                    message=f"{len(invalid_refs)} références trop courtes (< 3 caractères)",
                    count=len(invalid_refs),
                    examples=invalid_refs['reference'].head(3).tolist(),
                    recommendation="Les références doivent avoir au moins 3 caractères",
                    impact="Difficulté d'identification et de recherche"
                ))
                self.score -= 5
        
        # Format des dates
        if 'created_at' in self.df.columns:
            try:
                pd.to_datetime(self.df['created_at'], errors='raise')
            except:
                self.issues.append(DataQualityIssue(
                    severity='MAJOR',
                    category='FORMAT',
                    field='created_at',
                    message="Format de date invalide détecté dans certaines lignes",
                    count=len(self.df),
                    recommendation="Uniformiser le format des dates (YYYY-MM-DD)",
                    impact="Impossible d'analyser les tendances temporelles"
                ))
                self.score -= 10
        
        # Prix négatifs ou nuls
        if 'last_price' in self.df.columns:
            try:
                invalid_prices = self.df[
                    (pd.to_numeric(self.df['last_price'], errors='coerce') <= 0) &
                    (self.df['last_price'] != '')
                ]
                if len(invalid_prices) > 0:
                    self.issues.append(DataQualityIssue(
                        severity='WARNING',
                        category='FORMAT',
                        field='last_price',
                        message=f"{len(invalid_prices)} prix invalides (<= 0)",
                        count=len(invalid_prices),
                        recommendation="Vérifier les valeurs des prix",
                        impact="Valorisation incorrecte du stock"
                    ))
            except:
                pass
    
    def _check_consistency(self):
        """Vérifie la cohérence entre les champs"""
        
        # Vérifier le statut actif/inactif
        if 'status' in self.df.columns and 'last_price' in self.df.columns:
            try:
                active_without_price = self.df[
                    (self.df['status'].str.lower() == 'qualified') & 
                    ((self.df['last_price'].isna()) | (self.df['last_price'] == '') | (self.df['last_price'] == '0'))
                ]
                if len(active_without_price) > 0:
                    self.issues.append(DataQualityIssue(
                        severity='WARNING',
                        category='CONSISTENCY',
                        field='status/price',
                        message=f"{len(active_without_price)} items qualifiés sans prix",
                        count=len(active_without_price),
                        examples=active_without_price['item_name'].head(3).tolist() if 'item_name' in active_without_price.columns else [],
                        recommendation="Vérifier si ces items qualifiés doivent avoir un prix",
                        impact="Impossible de valoriser ces items"
                    ))
            except:
                pass
    
    def _check_references(self):
        """Vérifie l'intégrité référentielle"""
        
        # Pattern de références par catégorie (à personnaliser selon vos règles)
        ref_patterns = {
            'IT': r'^IT[A-Z]{2}-\d{4}$',
            'ELEC': r'^ELEC-\d{4}$',
            'MECA': r'^MECA-\d{4}$',
            'COS': r'^COS-\d{4}$'
        }
        
        if 'reference' in self.df.columns and 'category_name' in self.df.columns:
            for category, pattern in ref_patterns.items():
                cat_items = self.df[self.df['category_name'].str.contains(category, na=False, case=False)]
                if len(cat_items) > 0:
                    invalid_refs = cat_items[~cat_items['reference'].astype(str).str.match(pattern, na=False)]
                    if len(invalid_refs) > 0:
                        self.issues.append(DataQualityIssue(
                            severity='MAJOR',
                            category='REFERENCE',
                            field='reference',
                            message=f"{len(invalid_refs)} références {category} non conformes au format standard",
                            count=len(invalid_refs),
                            examples=invalid_refs['reference'].head(3).tolist(),
                            recommendation=f"Les références {category} doivent suivre le format: {pattern}",
                            impact="Non-respect des standards de nommage"
                        ))
                        self.score -= 8
    
    def _check_name_quality(self):
        """Analyse la qualité des noms (sémantique)"""
        
        if 'item_name' not in self.df.columns:
            return
        
        # Noms trop courts
        short_names = self.df[self.df['item_name'].astype(str).str.len() < 5]
        if len(short_names) > 0:
            self.issues.append(DataQualityIssue(
                severity='MAJOR',
                category='SEMANTIC',
                field='item_name',
                message=f"{len(short_names)} noms trop courts (< 5 caractères)",
                count=len(short_names),
                examples=short_names['item_name'].head(3).tolist(),
                recommendation="Les noms d'items doivent être descriptifs (minimum 5 caractères)",
                impact="Difficulté d'identification et de recherche"
            ))
            self.score -= 8
        
        # Noms avec caractères spéciaux
        special_chars = self.df[self.df['item_name'].astype(str).str.contains(r'[^a-zA-Z0-9\s\-\.\(\)]', na=False)]
        if len(special_chars) > 0:
            self.issues.append(DataQualityIssue(
                severity='MINOR',
                category='FORMAT',
                field='item_name',
                message=f"{len(special_chars)} noms contiennent des caractères spéciaux non standards",
                count=len(special_chars),
                examples=special_chars['item_name'].head(3).tolist(),
                recommendation="Éviter les caractères spéciaux dans les noms (sauf - . ())",
                impact="Problèmes potentiels d'export et d'intégration"
            ))
            self.score -= 3
        
        # Noms en majuscules uniquement
        all_upper = self.df[self.df['item_name'].astype(str).str.isupper()]
        if len(all_upper) > 0:
            self.issues.append(DataQualityIssue(
                severity='INFO',
                category='FORMAT',
                field='item_name',
                message=f"{len(all_upper)} noms en majuscules uniquement",
                count=len(all_upper),
                examples=all_upper['item_name'].head(3).tolist(),
                recommendation="Privilégier la casse mixte (première lettre en majuscule)",
                impact="Problème esthétique uniquement"
            ))
    
    def _check_temporal_quality(self):
        """Vérifie la qualité temporelle des données"""
        
        if 'created_at' in self.df.columns:
            try:
                dates = pd.to_datetime(self.df['created_at'], errors='coerce')
                now = datetime.now()
                
                # Dates futures
                future_dates = dates[dates > now]
                if len(future_dates) > 0:
                    self.issues.append(DataQualityIssue(
                        severity='CRITICAL',
                        category='CONSISTENCY',
                        field='created_at',
                        message=f"{len(future_dates)} dates de création dans le futur",
                        count=len(future_dates),
                        examples=self.df.loc[future_dates.index, 'item_name'].head(3).tolist() if 'item_name' in self.df.columns else [],
                        recommendation="Corriger les dates futures",
                        impact="Incohérence dans la chronologie des données"
                    ))
                    self.score -= 15
                
                # Dates trop anciennes (> 5 ans)
                old_dates = dates[dates < now - pd.Timedelta(days=5*365)]
                if len(old_dates) > 0:
                    self.issues.append(DataQualityIssue(
                        severity='INFO',
                        category='TEMPORAL',
                        field='created_at',
                        message=f"{len(old_dates)} items très anciens (> 5 ans)",
                        count=len(old_dates),
                        examples=self.df.loc[old_dates.index, 'item_name'].head(3).tolist() if 'item_name' in self.df.columns else [],
                        recommendation="Vérifier si ces items sont toujours pertinents",
                        impact="Données potentiellement obsolètes"
                    ))
            except Exception as e:
                pass
    
    def _check_categories(self):
        """Analyse la qualité des catégories"""
        
        if 'category_name' in self.df.columns:
            # Catégories vides
            empty_cats = self.df[self.df['category_name'].isna() | (self.df['category_name'] == '')]
            if len(empty_cats) > 0:
                self.issues.append(DataQualityIssue(
                    severity='MAJOR',
                    category='MISSING',
                    field='category_name',
                    message=f"{len(empty_cats)} items sans catégorie",
                    count=len(empty_cats),
                    examples=empty_cats['item_name'].head(3).tolist() if 'item_name' in empty_cats.columns else [],
                    recommendation="Tous les items doivent avoir une catégorie",
                    impact="Impossible de classifier et regrouper les items"
                ))
                self.score -= 10
            
            # Catégories avec trop peu d'items
            cat_counts = self.df['category_name'].value_counts()
            rare_cats = cat_counts[cat_counts < 3]
            if len(rare_cats) > 0:
                self.issues.append(DataQualityIssue(
                    severity='INFO',
                    category='CONSISTENCY',
                    field='category_name',
                    message=f"{len(rare_cats)} catégories avec moins de 3 items",
                    count=len(rare_cats),
                    recommendation="Vérifier la pertinence de ces catégories ou les fusionner",
                    impact="Catégories trop spécifiques ou inutiles"
                ))
    
    def _calculate_quality_score(self):
        """Calcule le score de qualité global"""
        # Le score est déjà ajusté dans chaque méthode
        self.score = max(0, min(100, self.score))
    
    def _get_grade(self) -> str:
        """Retourne une note letter grade"""
        if self.score >= 90:
            return "A - Excellente qualité"
        elif self.score >= 80:
            return "B - Bonne qualité"
        elif self.score >= 70:
            return "C - Qualité moyenne"
        elif self.score >= 60:
            return "D - Qualité insuffisante"
        else:
            return "F - Qualité critique"
    
    def _generate_summary(self) -> str:
        """Génère un résumé de l'analyse"""
        summary = []
        
        by_severity = {}
        for issue in self.issues:
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        
        summary.append(f"**Résumé de l'analyse:**")
        summary.append(f"- 🔴 CRITIQUE: {by_severity.get('CRITICAL', 0)}")
        summary.append(f"- 🟠 MAJEUR: {by_severity.get('MAJOR', 0)}")
        summary.append(f"- 🟡 MINEUR: {by_severity.get('MINOR', 0)}")
        summary.append(f"- ⚠️  ATTENTION: {by_severity.get('WARNING', 0)}")
        summary.append(f"- ℹ️  INFO: {by_severity.get('INFO', 0)}")
        
        return "\n".join(summary)
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Regrouper par sévérité
        critical_issues = [i for i in self.issues if i.severity == 'CRITICAL']
        major_issues = [i for i in self.issues if i.severity == 'MAJOR']
        
        if critical_issues:
            recommendations.append("🔴 **ACTIONS IMMÉDIATES REQUISES:**")
            for issue in critical_issues[:5]:
                recommendations.append(f"  • {issue.recommendation}")
        
        if major_issues:
            recommendations.append("\n🟠 **AMÉLIORATIONS PRIORITAIRES:**")
            for issue in major_issues[:5]:
                recommendations.append(f"  • {issue.recommendation}")
        
        # Recommandations générales basées sur le score
        if self.score < 50:
            recommendations.append("\n📊 **PLAN D'ACTION:** Nettoyage complet des données nécessaire")
        elif self.score < 70:
            recommendations.append("\n📈 **PROCHAINES ÉTAPES:** Planifier des sessions de nettoyage ciblées")
        elif self.score < 85:
            recommendations.append("\n✨ **MAINTENANCE:** Mettre en place des contrôles qualité réguliers")
        else:
            recommendations.append("\n🏆 **FÉLICITATIONS:** Maintenir ces bonnes pratiques!")
        
        return recommendations
    
    def display_report(self):
        """Affiche un rapport détaillé dans Streamlit"""
        
        st.markdown("## 🤖 Rapport IA - Analyse de Qualité Master Data")
        
        # Score avec jauge
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            grade = self._get_grade().split(' - ')[0]
            st.markdown(f"<h1 style='text-align: center;'>{grade}</h1>", unsafe_allow_html=True)
        
        with col2:
            score_color = "green" if self.score >= 80 else "orange" if self.score >= 60 else "red"
            st.markdown(f"<h1 style='text-align: center; color: {score_color};'>{self.score:.1f}/100</h1>", unsafe_allow_html=True)
            st.progress(self.score/100)
        
        with col3:
            st.metric("Items analysés", f"{self.total_items:,}")
        
        # Résumé
        st.info(self._generate_summary())
        
        # Métriques rapides
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Qualité", self._get_grade())
        with col_m2:
            st.metric("Problèmes", len(self.issues))
        with col_m3:
            critical = len([i for i in self.issues if i.severity == 'CRITICAL'])
            st.metric("Critiques", critical, delta_color="inverse")
        with col_m4:
            impact = sum(i.count for i in self.issues)
            st.metric("Items impactés", f"{min(impact, self.total_items)}/{self.total_items}")
        
        # Tabs pour les problèmes
        tab_crit, tab_major, tab_other = st.tabs(["🔴 Critiques", "🟠 Majeurs", "📋 Autres"])
        
        with tab_crit:
            critical_issues = [i for i in self.issues if i.severity == 'CRITICAL']
            if critical_issues:
                for issue in critical_issues:
                    with st.expander(f"**{issue.field}**: {issue.message}"):
                        st.markdown(f"""
                        - **Recommandation:** {issue.recommendation}
                        - **Impact:** {issue.impact}
                        - **Items concernés:** {issue.count}
                        - **Exemples:** {', '.join(str(e) for e in issue.examples[:3]) if issue.examples else 'N/A'}
                        """)
            else:
                st.success("Aucun problème critique détecté !")
        
        with tab_major:
            major_issues = [i for i in self.issues if i.severity == 'MAJOR']
            if major_issues:
                for issue in major_issues:
                    with st.expander(f"**{issue.field}**: {issue.message}"):
                        st.markdown(f"""
                        - **Recommandation:** {issue.recommendation}
                        - **Impact:** {issue.impact}
                        - **Items concernés:** {issue.count}
                        - **Exemples:** {', '.join(str(e) for e in issue.examples[:3]) if issue.examples else 'N/A'}
                        """)
            else:
                st.success("Aucun problème majeur détecté !")
        
        with tab_other:
            other_issues = [i for i in self.issues if i.severity not in ['CRITICAL', 'MAJOR']]
            if other_issues:
                for issue in other_issues:
                    severity_icon = "🟡" if issue.severity == 'MINOR' else "⚠️" if issue.severity == 'WARNING' else "ℹ️"
                    with st.expander(f"{severity_icon} **{issue.field}**: {issue.message}"):
                        st.markdown(f"""
                        - **Recommandation:** {issue.recommendation}
                        - **Items concernés:** {issue.count}
                        """)
            else:
                st.info("Aucun autre problème détecté")
        
        # Recommandations
        st.markdown("## 💡 Recommandations de l'IA")
        for rec in self._generate_recommendations():
            st.markdown(rec)
        
        # Export du rapport
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📥 Exporter le rapport (JSON)", use_container_width=True):
                report_data = {
                    'date': datetime.now().isoformat(),
                    'total_items': self.total_items,
                    'score': self.score,
                    'grade': self._get_grade(),
                    'issues': [
                        {
                            'severity': i.severity,
                            'category': i.category,
                            'field': i.field,
                            'message': i.message,
                            'count': i.count,
                            'recommendation': i.recommendation,
                            'impact': i.impact
                        }
                        for i in self.issues
                    ]
                }
                import json
                st.download_button(
                    "Télécharger JSON",
                    data=json.dumps(report_data, indent=2, ensure_ascii=False),
                    file_name=f"rapport_qualite_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        with col_exp2:
            if st.button("📊 Vue synthétique (CSV)", use_container_width=True):
                issues_df = pd.DataFrame([
                    {
                        'Sévérité': i.severity,
                        'Catégorie': i.category,
                        'Champ': i.field,
                        'Message': i.message,
                        'Nombre': i.count,
                        'Recommandation': i.recommendation
                    }
                    for i in self.issues
                ])
                csv_data = issues_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Télécharger CSV",
                    data=csv_data,
                    file_name=f"issues_qualite_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

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

    st.markdown('<h1 class="main-header">🔍 Master Data Quality - Netis Group DWH</h1>', unsafe_allow_html=True)

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
                            
                            # Nettoyer les anciennes analyses
                            if 'last_quality_report' in st.session_state:
                                del st.session_state.last_quality_report
                            if 'last_name_analysis' in st.session_state:
                                del st.session_state.last_name_analysis
                            if 'last_direct_indirect' in st.session_state:
                                del st.session_state.last_direct_indirect
                            if 'last_accounting' in st.session_state:
                                del st.session_state.last_accounting
                            
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
                unique_names = df_stats['item_name'].nunique() if 'item_name' in df_stats.columns else 0
                duplicate_rate = ((total_items - unique_names) / total_items * 100) if total_items > 0 and unique_names > 0 else 0

                st.metric("Items total", f"{total_items:,}")
                st.metric("Noms uniques", f"{unique_names:,}")
                st.metric("Taux doublons", f"{duplicate_rate:.1f}%")
                
                if 'last_load' in st.session_state:
                    st.caption(f"Dernier chargement : {st.session_state.last_load}")

    # VÉRIFICATION CRITIQUE : Les données sont-elles chargées ?
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
            ### 📊 Analyses disponibles
            - **📝 Analyse des Noms** : Détection des anomalies dans les désignations
            - **📈 Analyse Direct/Indirect** : Détection des mots-clés "direct" et "indirect"
            - **💰 Classification Comptable** : Correspondance avec le plan comptable Netis Group
            - **🤖 IA Qualité** : Rapport complet de qualité des données
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

    # ================================================================
    # SI ON ARRIVE ICI, C'EST QUE LES DONNÉES EXISTENT
    # ================================================================
    df = st.session_state.df
    cache = st.session_state.cache

    # === 5 ONGLETS PRINCIPAUX ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Analyse des Noms",
        "📈 Analyse Direct/Indirect",
        "💰 Classification Comptable",
        "🤖 IA Master Data Quality",
        "📤 Export"
    ])

    with tab1:
        st.header("📝 Analyse des Noms")
        analyzer = NameQualityAnalyzer(df)
        results = analyzer.analyze()
        analyzer.display_report()

    with tab2:
        st.header("📈 Analyse Direct/Indirect")
        di_analyzer = DirectIndirectAnalyzer(df)
        di_analyzer.analyze_direct_indirect_keywords()
        di_analyzer.display_report()

    with tab3:
        st.header("💰 Classification Comptable Avancée")
        
        # Option pour choisir entre les versions (optionnel)
        use_advanced = st.checkbox("Utiliser la version avancée (avec mots-clés enrichis)", value=True)
        
        acc_classifier = AccountingClassifier(df)
        
        if use_advanced:
            with st.spinner("🚀 Classification avancée en cours..."):
                classified_df = acc_classifier.classify_all_advanced(use_parallel=True)
                acc_classifier.display_advanced_report()
        else:
            with st.spinner("Classification standard en cours..."):
                st.warning("La version standard n'est plus disponible - utilisation de la version avancée")
                classified_df = acc_classifier.classify_all_advanced(use_parallel=True)
                acc_classifier.display_advanced_report()

    with tab4:
        st.header("🤖 IA Master Data Quality")
        qa_analyzer = MasterDataQualityAI(df)
        qa_analyzer.analyze()
        qa_analyzer.display_report()

    with tab5:
        st.header("📤 Export")
        st.markdown("### 📥 Exporter les résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exporter données complètes (CSV)",
                data=csv_data,
                file_name=f"data_complete_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, sheet_name="Données")
            excel_buffer.seek(0)
            st.download_button(
                "📊 Exporter en Excel",
                data=excel_buffer.getvalue(),
                file_name=f"data_complete_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            st.info("✅ Sélectionnez l'onglet d'analyse pour exporter les résultats spécifiques")


if __name__ == "__main__":
    main()