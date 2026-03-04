import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime

# ================================================================
# Configuration Streamlit
# ================================================================
st.set_page_config(
    page_title="Classification Comptable - Netis Group",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# Correctifs CSS globaux (FORCE SCROLL)
# ================================================================
st.markdown("""
<style>
/* Force le scroll global même si un composant fixe overflow: hidden */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    height: auto !important;
    overflow: auto !important;
}

/* Assure un scroll vertical sur le conteneur de la page */
[data-testid="stAppViewContainer"] {
    overflow-y: auto !important;
}

/* Évite les hauteurs forcées qui bloquent le scroll */
[data-testid="stVerticalBlock"] {
    height: auto !important;
}

/* Le dataframe garde son propre scroll interne sans bloquer celui de la page */
[data-testid="stDataFrame"] div[role="grid"] {
    max-height: 70vh !important;
    overflow: auto !important;
}

/* Ne pas bloquer le comportement de scroll du navigateur */
* { overscroll-behavior: auto !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# CSS personnalisé de l'app
# ================================================================
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💰 Classification Comptable - Netis Group</h1>', unsafe_allow_html=True)

# ================================================================
# === DONNÉES DU PLAN COMPTABLE AVEC MOTS-CLÉS ENRICHIS
# ================================================================
def load_accounting_codes():
    """Charge les 225 codes comptables avec mots-clés enrichis pour Netis Group"""
    data = [
        # VENTES (1111-1112)
        {"code": "1111.001", "description_fr": "Ventes externes de matériaux uniquement", 
         "description_en": "External sales of materials only",
         "keywords_fr": "vente materiaux externe client",
         "keywords_en": "sales materials external customer",
         "type_names": []},
        
        {"code": "1111.002", "description_fr": "Ventes externes de services uniquement", 
         "description_en": "External sales of services only",
         "keywords_fr": "vente services prestation maintenance installation",
         "keywords_en": "sales services maintenance installation",
         "type_names": ["Other Direct Services", "Outsourced services"]},
        
        {"code": "1111.003", "description_fr": "Ventes externes de matériaux et services", 
         "description_en": "External sales of materials and services",
         "keywords_fr": "vente materiaux services mixte",
         "keywords_en": "sales materials services",
         "type_names": []},
        
        {"code": "1112.001", "description_fr": "Ventes interco de matériaux uniquement", 
         "description_en": "Interco sales of materials only",
         "keywords_fr": "vente materiaux interco groupe",
         "keywords_en": "sales materials intercompany",
         "type_names": []},
        
        {"code": "1112.002", "description_fr": "Ventes interco de services uniquement", 
         "description_en": "Interco sales of services only",
         "keywords_fr": "vente services interco prestation interne",
         "keywords_en": "sales services intercompany",
         "type_names": []},
        
        {"code": "1112.003", "description_fr": "Ventes interco de matériaux et services", 
         "description_en": "Interco sales of materials and services",
         "keywords_fr": "vente materiaux services interco mixte",
         "keywords_en": "sales materials services intercompany",
         "type_names": []},

        # MAIN D'OEUVRE (1211-1213)
        {"code": "1211.001", "description_fr": "Main d'œuvre", 
         "description_en": "Internal manpower (blue-collar and hourly paid staff)",
         "keywords_fr": "main oeuvre ouvrier horaire technicien personnel chantier mo salary",
         "keywords_en": "manpower worker hourly technician staff site",
         "type_names": []},
        
        {"code": "1211.002", "description_fr": "Intérim horaires", 
         "description_en": "External manpower (blue-collar and hourly paid staff)",
         "keywords_fr": "interim temporaire interimaire externe ouvrier",
         "keywords_en": "temporary agency external worker",
         "type_names": ["Outsourced staff", "Outsourced staff Admin", "Outsourcing"]},
        
        {"code": "1212.001", "description_fr": "Salaires encadrement expat", 
         "description_en": "Internal expat supervision and executive staff",
         "keywords_fr": "expat expatrie cadre superviseur manager direction",
         "keywords_en": "expat executive supervisor manager",
         "type_names": []},
        
        {"code": "1212.002", "description_fr": "Salaires encadrement local", 
         "description_en": "Internal local supervision and executive staff",
         "keywords_fr": "cadre superviseur local manager direction supervision",
         "keywords_en": "supervisor local manager executive",
         "type_names": []},
        
        {"code": "1212.003", "description_fr": "Encadrement intérimaire", 
         "description_en": "External supervision and executive staff",
         "keywords_fr": "cadre interim superviseur externe manager temporaire",
         "keywords_en": "supervisor interim external manager temporary",
         "type_names": []},
        
        {"code": "1213.001", "description_fr": "Missions et voyages", 
         "description_en": "Missions and travel costs",
         "keywords_fr": "mission voyage deplacement transport avion hotel per diem",
         "keywords_en": "travel mission transport flight hotel per diem",
         "type_names": ["Transport & Travel", "Transport & Travel admin"]},
        
        {"code": "1213.002", "description_fr": "Notes de frais et réceptions", 
         "description_en": "Expense claims and receptions costs",
         "keywords_fr": "note frais restaurant reception client repas",
         "keywords_en": "expense restaurant reception client meal",
         "type_names": []},
        
        {"code": "1213.003", "description_fr": "Formation", 
         "description_en": "Training costs",
         "keywords_fr": "formation training stage cours certification",
         "keywords_en": "training course certification",
         "type_names": ["Training Expense Admin", "TRAINING"]},
        
        {"code": "1213.004", "description_fr": "Assurance santé, médecine du travail et autres frais médicaux", 
         "description_en": "Health insurance and other health costs",
         "keywords_fr": "assurance sante medical medecin travail pharmacie hopital",
         "keywords_en": "health insurance medical hospital medicine",
         "type_names": []},
        
        {"code": "1213.005", "description_fr": "Frais divers personnel", 
         "description_en": "Misc. HR costs",
         "keywords_fr": "frais divers personnel rh ressource humaine",
         "keywords_en": "misc hr human resource staff",
         "type_names": []},
        
        {"code": "1213.006", "description_fr": "Restauration", 
         "description_en": "Catering costs",
         "keywords_fr": "restauration cantine repas nourriture catering",
         "keywords_en": "catering meal food",
         "type_names": ["Cafeteria"]},
        
        {"code": "1213.007", "description_fr": "Loyers & charges locatives terrains & locaux (habitation)", 
         "description_en": "Rents, land & premises and related expenses (living)",
         "keywords_fr": "logement loyer maison appartement habitation camp base vie",
         "keywords_en": "housing rent apartment house accommodation",
         "type_names": ["Accommodation"]},

        # MATÉRIAUX GÉNIE CIVIL (1221)
        {"code": "1221.001", "description_fr": "Béton prêt à l'emploi", 
         "description_en": "Ready-mix concrete",
         "keywords_fr": "beton béton ciment mortier fondation dalle",
         "keywords_en": "concrete cement foundation slab",
         "type_names": []},
        
        {"code": "1221.002", "description_fr": "Cailloux, sables et autres matériaux de remblai", 
         "description_en": "Rocks, sands and other backfill materials",
         "keywords_fr": "cailloux sable gravier remblai granulat agregat",
         "keywords_en": "rocks sand gravel backfill aggregate",
         "type_names": []},
        
        {"code": "1221.003", "description_fr": "Aciers pour armatures et armatures façonnées", 
         "description_en": "Steel rebars and shaped steel frame",
         "keywords_fr": "acier fer armature rond treillis metal",
         "keywords_en": "steel rebar reinforcement metal",
         "type_names": []},
        
        {"code": "1221.004", "description_fr": "Parpaings & briques", 
         "description_en": "Bricks and building blocks",
         "keywords_fr": "parpaing brique bloc agglo maçonnerie",
         "keywords_en": "brick block masonry",
         "type_names": []},
        
        {"code": "1221.005", "description_fr": "Géotextiles", 
         "description_en": "Geotextiles",
         "keywords_fr": "geotextile bidim drainage tissu",
         "keywords_en": "geotextile drainage",
         "type_names": []},
        
        {"code": "1221.006", "description_fr": "Étanchéité", 
         "description_en": "Sealing",
         "keywords_fr": "etancheite etanche joint silicone membrane",
         "keywords_en": "sealing waterproof membrane joint",
         "type_names": []},
        
        {"code": "1221.007", "description_fr": "Divers préfabriqués béton", 
         "description_en": "Misc precasted concrete elements",
         "keywords_fr": "prefabrique beton element poutre dalle",
         "keywords_en": "precast concrete element beam slab",
         "type_names": []},
        
        {"code": "1221.008", "description_fr": "Tuyaux fonte", 
         "description_en": "Cast iron pipes",
         "keywords_fr": "tuyau fonte conduite canalisation",
         "keywords_en": "pipe cast iron conduit",
         "type_names": []},
        
        {"code": "1221.009", "description_fr": "Tuyaux PVC", 
         "description_en": "PVC pipes",
         "keywords_fr": "tuyau pvc plastique conduite",
         "keywords_en": "pipe pvc plastic conduit",
         "type_names": ["PVC PIPE"]},
        
        {"code": "1221.010", "description_fr": "Grillage", 
         "description_en": "Fence",
         "keywords_fr": "grillage clôture barriere grillage",
         "keywords_en": "fence mesh wire",
         "type_names": []},
        
        {"code": "1221.011", "description_fr": "Bois", 
         "description_en": "Wood",
         "keywords_fr": "bois planche coffrage contreplaque madrier",
         "keywords_en": "wood timber plywood formwork",
         "type_names": []},
        
        {"code": "1221.012", "description_fr": "Divers matériaux GC incorporés", 
         "description_en": "Miscellaneous civil works incorporated materials",
         "keywords_fr": "materiaux gc divers divers chantier construction",
         "keywords_en": "misc materials civil works construction",
         "type_names": ["Building / Chantier"]},

        # ÉQUIPEMENTS TÉLÉCOMS ET PYLÔNES (1222)
        {"code": "1222.001", "description_fr": "Pylônes telecom greenfield", 
         "description_en": "Pylon telecom greenfield",
         "keywords_fr": "pylone greenfield tour telecom antenne mât self supporting",
         "keywords_en": "pylon tower telecom greenfield antenna mast",
         "type_names": []},
        
        {"code": "1222.002", "description_fr": "Pylônes telecom toit-terrasse", 
         "description_en": "Pylon telecom rooftop",
         "keywords_fr": "pylone toit terrasse rooftop telecom antenne",
         "keywords_en": "pylon rooftop telecom antenna",
         "type_names": []},
        
        {"code": "1222.003", "description_fr": "Pylônes électriques", 
         "description_en": "Electric pylons",
         "keywords_fr": "pylone electrique haute tension ligne",
         "keywords_en": "pylon electric power line",
         "type_names": []},
        
        {"code": "1222.004", "description_fr": "Support panneaux solaires", 
         "description_en": "Solar panel supports",
         "keywords_fr": "panneau solaire support photovoltaique pv solar",
         "keywords_en": "solar panel support photovoltaic",
         "type_names": []},
        
        {"code": "1222.005", "description_fr": "Profilés métalliques", 
         "description_en": "Metal profiles and beams",
         "keywords_fr": "profilé metal poutre ipe hea corniere fer",
         "keywords_en": "profile metal beam steel",
         "type_names": ["Metalic Structure-Others"]},
        
        {"code": "1222.006", "description_fr": "Ancrage, barres, boulons", 
         "description_en": "Anchorage, rods, bolts and nuts",
         "keywords_fr": "ancrage boulon vis ecrou rondelle barre tige",
         "keywords_en": "anchor bolt nut screw washer rod",
         "type_names": ["SCREW"]},
        
        {"code": "1222.007", "description_fr": "Câbles métalliques / aciers", 
         "description_en": "Steel/metal ropes and cables",
         "keywords_fr": "cable metal acier tresse hauban",
         "keywords_en": "cable steel rope guy wire",
         "type_names": []},
        
        {"code": "1222.008", "description_fr": "Tôles métalliques", 
         "description_en": "Metal sheets",
         "keywords_fr": "tole metal plaque acier",
         "keywords_en": "sheet metal plate steel",
         "type_names": []},
        
        {"code": "1222.009", "description_fr": "Produits et accessoires anti-corrosion", 
         "description_en": "Anti corrosive products and accessories",
         "keywords_fr": "anticorrosion anti corrosion peinture traitement",
         "keywords_en": "anticorrosion anti corrosion paint treatment",
         "type_names": []},
        
        {"code": "1222.010", "description_fr": "Divers matériaux métalliques incorporés", 
         "description_en": "Miscellaneous metallic incorporated materials",
         "keywords_fr": "materiaux metallique divers fer acier inox",
         "keywords_en": "misc metallic materials steel iron",
         "type_names": []},

        # FIBRE OPTIQUE (1223)
        {"code": "1223.001", "description_fr": "Poteaux bois", 
         "description_en": "Wooden poles",
         "keywords_fr": "poteau bois support aerien",
         "keywords_en": "pole wood aerial",
         "type_names": []},
        
        {"code": "1223.002", "description_fr": "Poteaux béton", 
         "description_en": "Concrete poles",
         "keywords_fr": "poteau beton support aerien",
         "keywords_en": "pole concrete aerial",
         "type_names": []},
        
        {"code": "1223.003", "description_fr": "Poteaux métalliques", 
         "description_en": "Metal poles",
         "keywords_fr": "poteau metal acier support aerien",
         "keywords_en": "pole metal steel aerial",
         "type_names": []},
        
        {"code": "1223.004", "description_fr": "Connecteurs FO", 
         "description_en": "FO connectors",
         "keywords_fr": "connecteur fo fibre optique lc sc fc adaptateur pig tail pigtail",
         "keywords_en": "connector fo fiber fibre optic lc sc fc adapter pigtail",
         "type_names": ["Fibre Optic-Consumables", "Fibre Optic-Spare parts"]},
        
        {"code": "1223.005", "description_fr": "Câble FO", 
         "description_en": "Cable FO",
         "keywords_fr": "cable fo fibre optique brin fiber single mode multimode sm mm",
         "keywords_en": "cable fo fiber fibre optic strand single mode multimode",
         "type_names": ["Fibre Optic-Consumables", "Fibre Optic-Spare parts", "Fiber Optic - Equipment"]},
        
        {"code": "1223.006", "description_fr": "Boîtier FO", 
         "description_en": "Box FO",
         "keywords_fr": "boitier fibre raccordement fermeture closure epissure splice",
         "keywords_en": "box closure splice enclosure fiber",
         "type_names": ["Fibre Optic-Consumables", "Fibre Optic-Spare parts", "Street cabinet"]},
        
        {"code": "1223.007", "description_fr": "Panneaux et armoires FO", 
         "description_en": "FO panels and cabinets",
         "keywords_fr": "panneau armoire fibre odf distribution tiroir drawer",
         "keywords_en": "panel cabinet odf distribution drawer",
         "type_names": ["Fibre Optic-Equipment"]},
        
        {"code": "1223.008", "description_fr": "Divers matériaux FO incorporés", 
         "description_en": "Miscellaneous FO incorporated materials",
         "keywords_fr": "materiaux fo divers accessoire fibre",
         "keywords_en": "misc fo materials fiber accessories",
         "type_names": ["Fibre Optic-Consumables", "Fibre Optic-Spare parts"]},

        # ÉQUIPEMENTS INDUSTRIELS (1224)
        {"code": "1224.001", "description_fr": "Système protection anti-incendie", 
         "description_en": "Fire protection system",
         "keywords_fr": "incendie feu extincteur detection alarme sprinkler",
         "keywords_en": "fire extinguisher detection alarm sprinkler",
         "type_names": ["Detector"]},
        
        {"code": "1224.002", "description_fr": "Lampadaires", 
         "description_en": "Streetlights",
         "keywords_fr": "lampadaire eclairage public lumiere mât",
         "keywords_en": "streetlight lighting lamp",
         "type_names": ["LED LIGHT"]},
        
        {"code": "1224.003", "description_fr": "Système d'éclairage", 
         "description_en": "Lighting systems",
         "keywords_fr": "eclairage lumiere lampe led projecteur",
         "keywords_en": "lighting lamp led projector",
         "type_names": ["LED LIGHT"]},
        
        {"code": "1224.004", "description_fr": "Système vidéo surveillance", 
         "description_en": "CCTV systems",
         "keywords_fr": "video surveillance camera cctv ip monitoring",
         "keywords_en": "cctv camera surveillance video",
         "type_names": ["Remote motoring system"]},
        
        {"code": "1224.005", "description_fr": "Pompes et compresseurs industriels", 
         "description_en": "Industrial pumps and compressors",
         "keywords_fr": "pompe compresseur water pump air pressure",
         "keywords_en": "pump compressor industrial",
         "type_names": []},
        
        {"code": "1224.006", "description_fr": "Instrumentation et système de mesure", 
         "description_en": "Instrumentation and metering systems",
         "keywords_fr": "instrumentation mesure capteur sensor compteur",
         "keywords_en": "instrumentation metering sensor gauge",
         "type_names": []},
        
        {"code": "1224.007", "description_fr": "Système de climatisation et refroidissement", 
         "description_en": "Cooling and air conditioning systems",
         "keywords_fr": "climatisation refroidissement hvac cooling air conditioning",
         "keywords_en": "air conditioning cooling hvac",
         "type_names": ["AIR CONDITIONER", "Inventer AC"]},
        
        {"code": "1224.008", "description_fr": "Autres équipements industriels incorporés", 
         "description_en": "Other incorporated industrial equipment",
         "keywords_fr": "equipement industriel divers industrie",
         "keywords_en": "misc industrial equipment",
         "type_names": ["VOLTAGE REGULATOR"]},

        # GROUPES ÉLECTROGÈNES ET ALIMENTATION (1225) - ENRICHIS POUR GEN
        {"code": "1225.001", "description_fr": "Groupes électrogènes", 
         "description_en": "Generators",
         "keywords_fr": "groupe electrogene generateur genset diesel generator gen power electrique",
         "keywords_en": "generator genset diesel power electric gen",
         "type_names": ["Generators-Equipment", "Generators-Consumables", "Generators-Spare parts"]},
        
        {"code": "1225.002", "description_fr": "Cuves à gasoil", 
         "description_en": "Fuel tanks",
         "keywords_fr": "cuve gasoil fuel reservoir tank diesel stockage",
         "keywords_en": "tank fuel diesel storage",
         "type_names": []},
        
        {"code": "1225.003", "description_fr": "Inverseurs & ATS", 
         "description_en": "Inverters & ATS",
         "keywords_fr": "inverseur ats transfert automatique commutateur",
         "keywords_en": "inverter ats transfer switch",
         "type_names": []},
        
        {"code": "1225.004", "description_fr": "Batteries", 
         "description_en": "Batteries",
         "keywords_fr": "batterie accumulateur plomb acide opzv opzs vrla",
         "keywords_en": "battery accumulator lead acid vrla",
         "type_names": []},
        
        {"code": "1225.005", "description_fr": "Panneaux solaires", 
         "description_en": "Solar panels",
         "keywords_fr": "panneau solaire photovoltaique pv solar",
         "keywords_en": "panel solar photovoltaic pv",
         "type_names": ["Renewable energy-Equipments"]},
        
        {"code": "1225.006", "description_fr": "Redresseurs", 
         "description_en": "Rectifiers",
         "keywords_fr": "redresseur rectifier alimentation power supply",
         "keywords_en": "rectifier power supply",
         "type_names": []},
        
        {"code": "1225.007", "description_fr": "Câbles électriques", 
         "description_en": "Electrical cables",
         "keywords_fr": "cable electrique cuivre aluminium section mm2 multiconducteur",
         "keywords_en": "cable electrical copper aluminum conductor",
         "type_names": ["Electrical-Consumables", "Electrical-Spare parts", "Power cables"]},
        
        {"code": "1225.008", "description_fr": "Paratonnerre et matériel de mise à la terre", 
         "description_en": "Lightning rods and grounding equipment",
         "keywords_fr": "paratonnerre foudre terre ground lightning piquet earth cable",
         "keywords_en": "lightning rod grounding earth cable",
         "type_names": []},
        
        {"code": "1225.009", "description_fr": "Compteurs et armoires électriques", 
         "description_en": "Electrical panels and cabinets",
         "keywords_fr": "compteur armoire electrique tableau distribution enel",
         "keywords_en": "panel cabinet electrical distribution meter",
         "type_names": ["Electrical-Equipment"]},
        
        {"code": "1225.010", "description_fr": "Divers fournitures électriques incorporées", 
         "description_en": "Misc electrical incorporated materials",
         "keywords_fr": "fourniture electrique divers materiel electrique",
         "keywords_en": "misc electrical materials supplies",
         "type_names": ["Electrical-Consumables", "Electrical-Spare parts", "TIE WRAP"]},

        # ÉQUIPEMENTS TÉLÉCOMS ACTIFS (1226)
        {"code": "1226.001", "description_fr": "Divers équipements télécoms actifs", 
         "description_en": "Misc active telecom equipment",
         "keywords_fr": "equipement telecom actif antenne radio bts",
         "keywords_en": "active telecom equipment antenna radio bts",
         "type_names": ["Telecom-Equipment"]},

        # MAINTENANCE ET PDR (1227)
        {"code": "1227.001", "description_fr": "PDR pour entretien courant et réparations diverses (sites clients)", 
         "description_en": "Spare parts for daily and minor unplanned maintenance (customers sites)",
         "keywords_fr": "pdr piece rechange spare entretien maintenance courant reparation",
         "keywords_en": "spare part maintenance repair",
         "type_names": ["Spare parts", "Spare parts-Others", "Maintenance", "Maintenance Admin"]},
        
        {"code": "1227.002", "description_fr": "PDR pour grande maintenance planifiée & entretien majeur (sites clients)", 
         "description_en": "Spare parts for major and planned maintenance (customers sites)",
         "keywords_fr": "pdr piece rechange grande maintenance planifiee majeure",
         "keywords_en": "spare part major planned maintenance",
         "type_names": ["Spare parts", "Maintenance"]},
        
        {"code": "1227.003", "description_fr": "PDR pour entretien & réparation suite accident ou casse (sites clients)", 
         "description_en": "Repairs after accident and breakdown (customers sites)",
         "keywords_fr": "pdr piece rechange accident casse reparations",
         "keywords_en": "spare part accident breakdown repair",
         "type_names": []},
        
        {"code": "1227.004", "description_fr": "Huiles & graisses pour maintenance (sites clients)", 
         "description_en": "Oils, lubricants & grease for maintenance (customers sites)",
         "keywords_fr": "huile graisse lubrifiant maintenance vidange wd40 degraissant",
         "keywords_en": "oil grease lubricant maintenance wd40",
         "type_names": ["OIL FILTER", "FUEL FILTER"]},
        
        {"code": "1227.005", "description_fr": "Divers matériaux et consommables pour la maintenance des sites clients", 
         "description_en": "Misc materials and consumables for customers sites",
         "keywords_fr": "materiaux consommables maintenance divers",
         "keywords_en": "misc materials consumables maintenance",
         "type_names": ["Consumables for Project", "Consumables-Others", "Consumables for Admin"]},

        # ÉQUIPEMENTS RÉSEAUX (1228)
        {"code": "1228.001", "description_fr": "Autres équipements réseaux", 
         "description_en": "Misc network equipment",
         "keywords_fr": "equipement reseau network switch routeur",
         "keywords_en": "network equipment switch router",
         "type_names": []},
        
        {"code": "1228.002", "description_fr": "Peinture", 
         "description_en": "Painting",
         "keywords_fr": "peinture peint color couleur weather guard forest green",
         "keywords_en": "paint painting color weather guard",
         "type_names": []},
        
        {"code": "1228.003", "description_fr": "Autres matériaux incorporés", 
         "description_en": "Other incorporated materials",
         "keywords_fr": "autres materiaux divers materiel",
         "keywords_en": "other materials misc",
         "type_names": []},

        # CARBURANT (1229)
        {"code": "1229.001", "description_fr": "Carburant pour refueling des sites clients", 
         "description_en": "Customers sites refueling",
         "keywords_fr": "carburant gasoil diesel essence fuel refueling site client",
         "keywords_en": "fuel diesel gasoline refueling customer",
         "type_names": ["Customer Site Fuel", "FUEL"]},

        # OUTILLAGE (1231-1234)
        {"code": "1231.001", "description_fr": "Petit outillage", 
         "description_en": "Other tools and small equipment",
         "keywords_fr": "outil outillage marteau tournevis pince cle scie cutter",
         "keywords_en": "tool hammer screwdriver plier cutter wrench",
         "type_names": ["Small Tools & Equipments", "Small Tools & Equipments Admin", "Small Equipment", "Handling material"]},
        
        {"code": "1231.002", "description_fr": "Consommable, outillage de soudure", 
         "description_en": "Welding tools and consumables",
         "keywords_fr": "soudure soudage electrode poste welding",
         "keywords_en": "welding electrode",
         "type_names": []},
        
        {"code": "1231.003", "description_fr": "Accessoires et outils de levage", 
         "description_en": "Lifting and handling tools and accessories",
         "keywords_fr": "levage manutention palan sangles élingue",
         "keywords_en": "lifting handling hoist sling",
         "type_names": ["Handling material"]},
        
        {"code": "1231.004", "description_fr": "Consommables coffrages", 
         "description_en": "Formwork consumables",
         "keywords_fr": "coffrage consommable banche etai",
         "keywords_en": "formwork consumable shoring",
         "type_names": []},
        
        {"code": "1232.001", "description_fr": "Échafaudage et garde-corps", 
         "description_en": "Scaffoldings and guardrails",
         "keywords_fr": "echafaudage garde corps scaffolding",
         "keywords_en": "scaffolding guardrail",
         "type_names": []},
        
        {"code": "1232.002", "description_fr": "Autres outils temporaires", 
         "description_en": "Other temporary tools",
         "keywords_fr": "outil temporaire provisoire chantier",
         "keywords_en": "temporary tool site",
         "type_names": []},
        
        {"code": "1233.001", "description_fr": "Consommables environnement", 
         "description_en": "Environment consumables",
         "keywords_fr": "environnement absorbeur kit antipollution",
         "keywords_en": "environment spill kit",
         "type_names": []},
        
        {"code": "1233.002", "description_fr": "Matériel signalisation", 
         "description_en": "Signs, buoys and anchors",
         "keywords_fr": "signalisation panneau cône balise",
         "keywords_en": "signs cone beacon marker",
         "type_names": []},
        
        {"code": "1233.003", "description_fr": "Équipement de Protection Individuel (EPI)", 
         "description_en": "PPE and safety expenses",
         "keywords_fr": "epi casque gant chaussure securite lunette harnais protection",
         "keywords_en": "ppe helmet glove shoe safety glasses harness",
         "type_names": ["WORK PROTECTIVE EQUIPMENT", "Safety equipment", "Work Protective Equipment"]},
        
        {"code": "1234.001", "description_fr": "Eau", 
         "description_en": "Water",
         "keywords_fr": "eau eau potable",
         "keywords_en": "water drinking water",
         "type_names": []},
        
        {"code": "1234.002", "description_fr": "Divers consommables", 
         "description_en": "Other misc. consumables",
         "keywords_fr": "consommable divers divers",
         "keywords_en": "misc consumables",
         "type_names": ["Consumables-Others"]},

        # LOCATIONS (1241)
        {"code": "1241.001", "description_fr": "Location externe voiture, pickup & suv", 
         "description_en": "Car, pickup & suv external rental",
         "keywords_fr": "location voiture pickup suv 4x4 vehicule rental",
         "keywords_en": "rental car pickup suv vehicle",
         "type_names": ["VEHICLE"]},
        
        {"code": "1241.002", "description_fr": "Location externe motos & triporteurs", 
         "description_en": "Motorbikes & three wheelers external rental",
         "keywords_fr": "location moto motocycle triporteur",
         "keywords_en": "rental motorcycle three wheeler",
         "type_names": ["MOTORCYCLE & TRICYCLE"]},
        
        {"code": "1241.003", "description_fr": "Location externe véhicules pour refueling", 
         "description_en": "Refueling vehicles external rental",
         "keywords_fr": "location vehicule refueling carburant",
         "keywords_en": "rental refueling vehicle",
         "type_names": []},
        
        {"code": "1241.004", "description_fr": "Location externe autres camions", 
         "description_en": "Other trucks external rental",
         "keywords_fr": "location camion poids lourd",
         "keywords_en": "rental truck heavy duty",
         "type_names": []},
        
        {"code": "1241.005", "description_fr": "Location externe grue mobile", 
         "description_en": "Mobile crane external rental",
         "keywords_fr": "location grue mobile crane",
         "keywords_en": "rental mobile crane",
         "type_names": []},
        
        {"code": "1241.006", "description_fr": "Location externe autres matériel de levage et manutention", 
         "description_en": "Other lifting and handling equipment external rental",
         "keywords_fr": "location levage manutention chariot",
         "keywords_en": "rental lifting handling forklift",
         "type_names": []},
        
        {"code": "1241.007", "description_fr": "Location externe de pelle", 
         "description_en": "HEX external rental",
         "keywords_fr": "pelle excavatrice hex bull tractopelle excavator",
         "keywords_en": "excavator hex digger",
         "type_names": []},
        
        {"code": "1241.008", "description_fr": "Location externe de chargeur", 
         "description_en": "Wheel loader external rental",
         "keywords_fr": "chargeur wheel loader",
         "keywords_en": "wheel loader",
         "type_names": []},
        
        {"code": "1241.009", "description_fr": "Location externe matériel compactage", 
         "description_en": "Compaction equipment external rental",
         "keywords_fr": "compactage compacteur rouleau",
         "keywords_en": "compaction roller compactor",
         "type_names": []},
        
        {"code": "1241.010", "description_fr": "Location externe matériel de réglage", 
         "description_en": "Grader external rental",
         "keywords_fr": "reglage grader niveleuse",
         "keywords_en": "grader",
         "type_names": []},
        
        {"code": "1241.011", "description_fr": "Location externe autre matériel de terrassement", 
         "description_en": "Other earth-moving equipment external rental",
         "keywords_fr": "terrassement engin travaux publics earthmoving",
         "keywords_en": "earthmoving equipment",
         "type_names": []},
        
        {"code": "1241.012", "description_fr": "Location externe échafaudage et garde-corps", 
         "description_en": "Scaffoldings and guardrails external rental",
         "keywords_fr": "echafaudage garde corps scaffolding",
         "keywords_en": "scaffolding guardrail",
         "type_names": []},
        
        {"code": "1241.013", "description_fr": "Location externe matériel production énergie et éclairage", 
         "description_en": "Generator and lighting equipment external rental",
         "keywords_fr": "location groupe electrogene eclairage generator gen",
         "keywords_en": "rental generator lighting",
         "type_names": []},
        
        {"code": "1241.014", "description_fr": "Location externe matériel de forage et passage de câbles", 
         "description_en": "Drilling and cable routing equipment external rental",
         "keywords_fr": "location forage cable tirage",
         "keywords_en": "rental drilling cable pulling",
         "type_names": []},
        
        {"code": "1241.015", "description_fr": "Location externe matériel divers", 
         "description_en": "Other misc equipment external rental",
         "keywords_fr": "location materiel divers",
         "keywords_en": "rental misc equipment",
         "type_names": []},
        
        {"code": "1241.016", "description_fr": "Location externe outillage et machines FO", 
         "description_en": "Other tooling and machinery FO external rental",
         "keywords_fr": "location outillage machine fo fibre",
         "keywords_en": "rental tool machinery fiber optic",
         "type_names": []},
        
        {"code": "1241.017", "description_fr": "Location externe autres outillage et machines", 
         "description_en": "Other tooling and machinery external rental",
         "keywords_fr": "location outillage machine divers",
         "keywords_en": "rental tool machinery misc",
         "type_names": []},

        # AMORTISSEMENTS (1242)
        {"code": "1242.001", "description_fr": "Amortissements voiture, pickup & suv", 
         "description_en": "Car, pickup & suv depreciations",
         "keywords_fr": "amortissement voiture pickup suv",
         "keywords_en": "depreciation car pickup suv",
         "type_names": []},
        
        {"code": "1242.002", "description_fr": "Amortissements motos & triporteurs", 
         "description_en": "Motorbikes & three wheelers depreciations",
         "keywords_fr": "amortissement moto triporteur",
         "keywords_en": "depreciation motorcycle three wheeler",
         "type_names": []},
        
        {"code": "1242.003", "description_fr": "Amortissements véhicules pour refueling", 
         "description_en": "Refueling vehicles depreciations",
         "keywords_fr": "amortissement vehicule refueling",
         "keywords_en": "depreciation refueling vehicle",
         "type_names": []},
        
        {"code": "1242.004", "description_fr": "Amortissements autres camions", 
         "description_en": "Other trucks depreciations",
         "keywords_fr": "amortissement camion poids lourd",
         "keywords_en": "depreciation truck heavy duty",
         "type_names": []},
        
        {"code": "1242.005", "description_fr": "Amortissements grue mobile", 
         "description_en": "Mobile crane depreciations",
         "keywords_fr": "amortissement grue mobile crane",
         "keywords_en": "depreciation mobile crane",
         "type_names": []},
        
        {"code": "1242.006", "description_fr": "Amortissements autres matériel de levage et manutention", 
         "description_en": "Other lifting and handling equipment depreciations",
         "keywords_fr": "amortissement levage manutention",
         "keywords_en": "depreciation lifting handling",
         "type_names": []},
        
        {"code": "1242.007", "description_fr": "Amortissements de pelle", 
         "description_en": "HEX depreciations",
         "keywords_fr": "amortissement pelle excavatrice",
         "keywords_en": "depreciation excavator",
         "type_names": []},
        
        {"code": "1242.008", "description_fr": "Amortissements de chargeur", 
         "description_en": "Wheel loader depreciations",
         "keywords_fr": "amortissement chargeur loader",
         "keywords_en": "depreciation wheel loader",
         "type_names": []},
        
        {"code": "1242.009", "description_fr": "Amortissements matériel compactage", 
         "description_en": "Compaction equipment depreciations",
         "keywords_fr": "amortissement compactage compacteur",
         "keywords_en": "depreciation compaction compactor",
         "type_names": []},
        
        {"code": "1242.010", "description_fr": "Amortissements matériel de réglage", 
         "description_en": "Grader depreciations",
         "keywords_fr": "amortissement reglage grader",
         "keywords_en": "depreciation grader",
         "type_names": []},
        
        {"code": "1242.011", "description_fr": "Amortissements autre matériel de terrassement", 
         "description_en": "Other earth-moving equipment depreciations",
         "keywords_fr": "amortissement terrassement engin",
         "keywords_en": "depreciation earthmoving equipment",
         "type_names": []},
        
        {"code": "1242.012", "description_fr": "Amortissements échafaudage et garde-corps", 
         "description_en": "Scaffoldings and guardrails depreciations",
         "keywords_fr": "amortissement echafaudage garde corps",
         "keywords_en": "depreciation scaffolding guardrail",
         "type_names": []},
        
        {"code": "1242.013", "description_fr": "Amortissements matériel production énergie et éclairage", 
         "description_en": "Generator and lighting equipment depreciations",
         "keywords_fr": "amortissement groupe electrogene eclairage",
         "keywords_en": "depreciation generator lighting",
         "type_names": []},
        
        {"code": "1242.014", "description_fr": "Amortissements matériel de forage et passage de câbles", 
         "description_en": "Drilling and cable routing equipment depreciations",
         "keywords_fr": "amortissement forage cable",
         "keywords_en": "depreciation drilling cable",
         "type_names": []},
        
        {"code": "1242.015", "description_fr": "Amortissements matériel divers", 
         "description_en": "Other misc equipment depreciations",
         "keywords_fr": "amortissement materiel divers",
         "keywords_en": "depreciation misc equipment",
         "type_names": []},
        
        {"code": "1242.016", "description_fr": "Amortissements outillage et machines FO", 
         "description_en": "Other tooling and machinery FO depreciations",
         "keywords_fr": "amortissement outillage machine fo",
         "keywords_en": "depreciation tooling machinery fiber optic",
         "type_names": []},
        
        {"code": "1242.017", "description_fr": "Amortissements autres outillage et machines", 
         "description_en": "Other tooling and machinery depreciations",
         "keywords_fr": "amortissement outillage machine",
         "keywords_en": "depreciation tooling machinery",
         "type_names": []},

        # CARBURANT ET LUBRIFIANTS INTERNES (1243)
        {"code": "1243.001", "description_fr": "Essence / Diesel pour VL", 
         "description_en": "Petrol and diesel for LV",
         "keywords_fr": "essence diesel carburant vl",
         "keywords_en": "petrol diesel fuel car",
         "type_names": ["Fuel Admin", "OFFICE FUEL SUPPLY"]},
        
        {"code": "1243.002", "description_fr": "Essence / Diesel pour autre usage interne", 
         "description_en": "Petrol and diesel for other internal use",
         "keywords_fr": "essence diesel carburant interne",
         "keywords_en": "petrol diesel fuel internal",
         "type_names": ["Fuel"]},
        
        {"code": "1243.003", "description_fr": "Huiles & graisses pour usage interne", 
         "description_en": "Oils, lubricants & grease for internal use",
         "keywords_fr": "huile graisse lubrifiant interne",
         "keywords_en": "oil grease lubricant internal",
         "type_names": []},

        # ENTRETIEN ET RÉPARATIONS INTERNES (1244)
        {"code": "1244.001", "description_fr": "Entretiens, réparations et PDR pour VL", 
         "description_en": "Maintenances, repairs and spare parts for LV",
         "keywords_fr": "entretien reparation pdr vl",
         "keywords_en": "maintenance repair spare part car",
         "type_names": ["Motor Vehicles spares", "Motor Vehicales spares Admin", "Motor Vehicle Expneses", "Motor Vehicle Expneses Admin"]},
        
        {"code": "1244.002", "description_fr": "Entretiens, réparations et PDR pour autres équipements internes", 
         "description_en": "Maintenances, repairs and spare parts for other internal equipment",
         "keywords_fr": "entretien reparation pdr equipement interne",
         "keywords_en": "maintenance repair spare part internal equipment",
         "type_names": ["Office Maintenance"]},

        # FRET IMPORT (1251)
        {"code": "1251.001", "description_fr": "Import - fret aérien OPEX", 
         "description_en": "Import - air freight OPEX",
         "keywords_fr": "import fret aerien opex",
         "keywords_en": "import air freight opex",
         "type_names": ["Inland Freight"]},
        
        {"code": "1251.002", "description_fr": "Import - fret maritime OPEX", 
         "description_en": "Import - sea freight OPEX",
         "keywords_fr": "import fret maritime opex",
         "keywords_en": "import sea freight opex",
         "type_names": ["Inland Freight"]},
        
        {"code": "1251.003", "description_fr": "Import - fret aérien CAPEX", 
         "description_en": "Import - air freight CAPEX",
         "keywords_fr": "import fret aerien capex",
         "keywords_en": "import air freight capex",
         "type_names": []},
        
        {"code": "1251.004", "description_fr": "Import - fret maritime CAPEX", 
         "description_en": "Import - sea freight CAPEX",
         "keywords_fr": "import fret maritime capex",
         "keywords_en": "import sea freight capex",
         "type_names": []},

        # FRET EXPORT (1252)
        {"code": "1252.001", "description_fr": "Export - fret aérien OPEX", 
         "description_en": "Export - air freight OPEX",
         "keywords_fr": "export fret aerien opex",
         "keywords_en": "export air freight opex",
         "type_names": []},
        
        {"code": "1252.002", "description_fr": "Export - fret maritime OPEX", 
         "description_en": "Export - sea freight OPEX",
         "keywords_fr": "export fret maritime opex",
         "keywords_en": "export sea freight opex",
         "type_names": []},
        
        {"code": "1252.003", "description_fr": "Export - fret aérien CAPEX", 
         "description_en": "Export - air freight CAPEX",
         "keywords_fr": "export fret aerien capex",
         "keywords_en": "export air freight capex",
         "type_names": []},
        
        {"code": "1252.004", "description_fr": "Export - fret maritime CAPEX", 
         "description_en": "Export - sea freight CAPEX",
         "keywords_fr": "export fret maritime capex",
         "keywords_en": "export sea freight capex",
         "type_names": []},

        # DOUANES (1253)
        {"code": "1253.001", "description_fr": "Import & Export - droits de douane et coûts associés OPEX", 
         "description_en": "Import & Export - custom duties and related costs OPEX",
         "keywords_fr": "douane droit import export opex",
         "keywords_en": "custom duty import export opex",
         "type_names": []},
        
        {"code": "1253.002", "description_fr": "Import & Export - droits de douane et coûts associés CAPEX", 
         "description_en": "Import & Export - custom duties and related costs CAPEX",
         "keywords_fr": "douane droit import export capex",
         "keywords_en": "custom duty import export capex",
         "type_names": []},

        # ASSURANCES TRANSPORT (1254)
        {"code": "1254.001", "description_fr": "Assurances transport OPEX", 
         "description_en": "Transport insurances OPEX",
         "keywords_fr": "assurance transport opex",
         "keywords_en": "transport insurance opex",
         "type_names": ["Insurance", "Insurance Admin"]},
        
        {"code": "1254.002", "description_fr": "Assurances transport CAPEX", 
         "description_en": "Transport insurances CAPEX",
         "keywords_fr": "assurance transport capex",
         "keywords_en": "transport insurance capex",
         "type_names": []},
        
        {"code": "1254.003", "description_fr": "Autres frais de transport OPEX", 
         "description_en": "Other transportation costs OPEX",
         "keywords_fr": "autre frais transport opex",
         "keywords_en": "other transport cost opex",
         "type_names": []},
        
        {"code": "1254.004", "description_fr": "Autres frais de transport CAPEX", 
         "description_en": "Other transportation costs CAPEX",
         "keywords_fr": "autre frais transport capex",
         "keywords_en": "other transport cost capex",
         "type_names": []},

        # ÉTUDES TECHNIQUES EXTERNES (1261)
        {"code": "1261.001", "description_fr": "Frais d'études techniques externes (design and methods)", 
         "description_en": "External technical studies (design and methods)",
         "keywords_fr": "etude technique externe design methodes",
         "keywords_en": "external technical study design methods",
         "type_names": []},
        
        {"code": "1261.002", "description_fr": "Géomètres, géotechniciens et autres techniciens extérieurs", 
         "description_en": "External topographic, geotechnical and other technical surveys",
         "keywords_fr": "geometre geotechnicien topographie externe",
         "keywords_en": "surveyor geotechnical external",
         "type_names": []},
        
        {"code": "1261.003", "description_fr": "Frais de laboratoire", 
         "description_en": "Laboratory studies costs",
         "keywords_fr": "laboratoire essai analyse",
         "keywords_en": "laboratory test analysis",
         "type_names": []},
        
        {"code": "1261.004", "description_fr": "Frais d'études partenaires de groupement", 
         "description_en": "Studies and surveys partner JV and consortium",
         "keywords_fr": "etude partenaire groupement consortium",
         "keywords_en": "study partner joint venture consortium",
         "type_names": []},

        # FRAIS INTERNES (1262)
        {"code": "1262.001", "description_fr": "Frais internes études techniques", 
         "description_en": "Internal technical study costs",
         "keywords_fr": "frais interne etude technique",
         "keywords_en": "internal technical study",
         "type_names": []},
        
        {"code": "1262.002", "description_fr": "Frais tendering internes", 
         "description_en": "Internal tendering costs",
         "keywords_fr": "frais interne appel offre tendering",
         "keywords_en": "internal tendering bid",
         "type_names": []},
        
        {"code": "1262.003", "description_fr": "Frais d'achats et logistique internes", 
         "description_en": "Procurement, logistics and post-order internal services",
         "keywords_fr": "frais achat logistique interne",
         "keywords_en": "procurement logistics internal",
         "type_names": []},
        
        {"code": "1262.004", "description_fr": "Autres refacturations internes de services", 
         "description_en": "Other costs internal services",
         "keywords_fr": "refacturation interne service",
         "keywords_en": "internal service rebilling",
         "type_names": []},

        # CERTIFICATIONS (1263)
        {"code": "1263.001", "description_fr": "Frais contrôle technique / homologation diverse", 
         "description_en": "Certifications costs",
         "keywords_fr": "controle technique homologation certification",
         "keywords_en": "technical control certification",
         "type_names": []},

        # AUTRES PRESTATIONS (1264)
        {"code": "1264.001", "description_fr": "Autres prestations", 
         "description_en": "Other services provided",
         "keywords_fr": "autre prestation service",
         "keywords_en": "other service",
         "type_names": ["Installation and Commissioning"]},

        # INSTALLATIONS PRO (1271)
        {"code": "1271.001", "description_fr": "Loyers & charges locatives terrains & locaux (pro)", 
         "description_en": "Rents, land & premises and related expenses (pro)",
         "keywords_fr": "loyer terrain local professionnel",
         "keywords_en": "rent land premises professional",
         "type_names": ["Rent", "Rent Admin"]},
        
        {"code": "1271.002", "description_fr": "Location externe bungalows et divers installations pro", 
         "description_en": "Site facilities and bungalows external rental",
         "keywords_fr": "location bungalow installation chantier",
         "keywords_en": "rental bungalow site facility",
         "type_names": []},
        
        {"code": "1271.003", "description_fr": "Amortissements bungalows et divers installations pro", 
         "description_en": "Site facilities and bungalows depreciations",
         "keywords_fr": "amortissement bungalow installation",
         "keywords_en": "depreciation bungalow facility",
         "type_names": []},
        
        {"code": "1271.004", "description_fr": "Électricité, gaz et eau des installations pro", 
         "description_en": "Electricity, gas and water of sites facilities",
         "keywords_fr": "electricite gaz eau installation",
         "keywords_en": "electricity gas water facility",
         "type_names": []},
        
        {"code": "1271.005", "description_fr": "Gardiennage, frais de sécurité", 
         "description_en": "Security expenses",
         "keywords_fr": "gardiennage securite surveillance",
         "keywords_en": "security guarding",
         "type_names": ["Security", "Security Admin", "SHERQ"]},
        
        {"code": "1271.006", "description_fr": "Autres coûts d'installations et d'infrastructures OPEX", 
         "description_en": "Other installations and facilities costs OPEX",
         "keywords_fr": "autre cout installation infrastructure opex",
         "keywords_en": "other cost installation facility opex",
         "type_names": []},
        
        {"code": "1271.007", "description_fr": "Autres coûts d'installations et d'infrastructures - amortissements", 
         "description_en": "Other installations and facilities costs depreciations",
         "keywords_fr": "autre cout installation infrastructure amortissement",
         "keywords_en": "other cost installation facility depreciation",
         "type_names": []},
        
        {"code": "1271.008", "description_fr": "Mise en décharge matériaux divers et prestations environnementales", 
         "description_en": "Landfilling activity & other environmental service",
         "keywords_fr": "decharge dechet environnement",
         "keywords_en": "landfill waste environmental",
         "type_names": []},
        
        {"code": "1271.009", "description_fr": "Prestation, entretien et nettoyage", 
         "description_en": "Cleaning, maintenance and garbage disposals",
         "keywords_fr": "prestation entretien nettoyage",
         "keywords_en": "cleaning maintenance garbage",
         "type_names": []},

        # FRAIS JURIDIQUES (1272)
        {"code": "1272.001", "description_fr": "Frais d'actes et contentieux", 
         "description_en": "Deeds and disputes charges",
         "keywords_fr": "frais acte contentieux juridique",
         "keywords_en": "deed dispute legal",
         "type_names": ["Legal"]},
        
        {"code": "1272.002", "description_fr": "Auditeurs", 
         "description_en": "Auditors",
         "keywords_fr": "auditeur audit commissaire compte",
         "keywords_en": "auditor audit",
         "type_names": []},
        
        {"code": "1272.003", "description_fr": "Conseil fiscal", 
         "description_en": "Tax consultant",
         "keywords_fr": "conseil fiscal impot",
         "keywords_en": "tax consultant",
         "type_names": []},
        
        {"code": "1272.004", "description_fr": "Documentation & traduction", 
         "description_en": "Documentation and translation costs",
         "keywords_fr": "documentation traduction",
         "keywords_en": "documentation translation",
         "type_names": ["Documentation & Research"]},
        
        {"code": "1272.005", "description_fr": "Autres conseils et honoraires", 
         "description_en": "Other consultancy and fees",
         "keywords_fr": "autre conseil honoraire consultant",
         "keywords_en": "other consultancy fee",
         "type_names": ["Consultancy services", "Consultancy Fees", "Consultancy Fees Admin"]},

        # FRAIS INFORMATIQUES (1273)
        {"code": "1273.001", "description_fr": "Frais de télécommunication", 
         "description_en": "Telecommunication costs",
         "keywords_fr": "telecommunication telephone internet",
         "keywords_en": "telecom phone internet",
         "type_names": ["Communication Admin"]},
        
        {"code": "1273.002", "description_fr": "Coûts informatiques - matériel OPEX", 
         "description_en": "IT costs hardware OPEX",
         "keywords_fr": "informatique materiel hardware opex",
         "keywords_en": "it hardware opex",
         "type_names": ["IT Hardware", "Hardware", "Hardware accessories", "Lap Top Accessories"]},
        
        {"code": "1273.003", "description_fr": "Coûts informatiques - logiciels et licences OPEX", 
         "description_en": "IT costs software and licenses OPEX",
         "keywords_fr": "informatique logiciel licence opex",
         "keywords_en": "it software license opex",
         "type_names": ["IT Software", "Software", "Software Admin"]},
        
        {"code": "1273.004", "description_fr": "Coûts informatiques - amortissements du matériel", 
         "description_en": "IT costs hardware depreciations",
         "keywords_fr": "informatique materiel amortissement",
         "keywords_en": "it hardware depreciation",
         "type_names": []},
        
        {"code": "1273.005", "description_fr": "Coûts informatiques - amortissements des logiciels et licences", 
         "description_en": "IT costs software and licenses depreciations",
         "keywords_fr": "informatique logiciel licence amortissement",
         "keywords_en": "it software license depreciation",
         "type_names": []},

        # ASSURANCES ET FRAIS BANCAIRES (1274)
        {"code": "1274.001", "description_fr": "Assurances véhicules légers", 
         "description_en": "LV insurances",
         "keywords_fr": "assurance vehicule leger",
         "keywords_en": "car insurance",
         "type_names": ["Insurance", "Insurance Admin"]},
        
        {"code": "1274.002", "description_fr": "Assurance tous risques chantier", 
         "description_en": "Contractor's all risks insurance",
         "keywords_fr": "assurance tous risques chantier",
         "keywords_en": "contractor all risk insurance",
         "type_names": ["Insurance"]},
        
        {"code": "1274.003", "description_fr": "Assurances multirisques", 
         "description_en": "Multi-risks insurances",
         "keywords_fr": "assurance multirisque",
         "keywords_en": "multi risk insurance",
         "type_names": ["Insurance"]},
        
        {"code": "1274.004", "description_fr": "Autres assurances", 
         "description_en": "Other insurances",
         "keywords_fr": "autre assurance",
         "keywords_en": "other insurance",
         "type_names": ["Insurance"]},
        
        {"code": "1274.005", "description_fr": "Assurance bris de machine", 
         "description_en": "Machinery breakdown insurance",
         "keywords_fr": "assurance bris machine",
         "keywords_en": "machinery breakdown insurance",
         "type_names": ["Insurance"]},
        
        {"code": "1274.006", "description_fr": "Frais bancaires", 
         "description_en": "Banking charges",
         "keywords_fr": "frais bancaire banque",
         "keywords_en": "bank charge",
         "type_names": []},
        
        {"code": "1274.007", "description_fr": "Frais de garantie et cautions", 
         "description_en": "Bank guarantees and bonds costs",
         "keywords_fr": "frais garantie caution banque",
         "keywords_en": "guarantee bond bank",
         "type_names": []},

        # FRAIS ADMINISTRATIFS (1275)
        {"code": "1275.001", "description_fr": "Fournitures de bureau et consommables informatiques", 
         "description_en": "Office consumables and supplies",
         "keywords_fr": "fourniture bureau consommable informatique",
         "keywords_en": "office supply consumable",
         "type_names": ["IT consumables", "IT consumables Admin"]},
        
        {"code": "1275.002", "description_fr": "Charges administratives diverses", 
         "description_en": "Miscellaneous administrative charges",
         "keywords_fr": "charge administrative divers",
         "keywords_en": "misc administrative",
         "type_names": ["Other expenses Admin"]},
        
        {"code": "1275.003", "description_fr": "Business promotion", 
         "description_en": "Business promotion",
         "keywords_fr": "business promotion marketing publicite",
         "keywords_en": "business promotion marketing",
         "type_names": ["Marketing"]},
        
        {"code": "1275.004", "description_fr": "Dépenses RSE", 
         "description_en": "RSE & CSR expenses",
         "keywords_fr": "rse responsabilite sociale",
         "keywords_en": "csr corporate social responsibility",
         "type_names": []},
        
        {"code": "1275.005", "description_fr": "Publicité, annonces insertions", 
         "description_en": "Advertising expenses",
         "keywords_fr": "publicite annonce insertion",
         "keywords_en": "advertising",
         "type_names": ["Marketing"]},
        
        {"code": "1275.006", "description_fr": "Matériel bureau OPEX", 
         "description_en": "Office equipment costs OPEX",
         "keywords_fr": "materiel bureau opex",
         "keywords_en": "office equipment opex",
         "type_names": ["OFFICE FURNITURES", "MOBILIER ET FOURNITURES DE BUREAU/ OFFICE FURNITURES"]},
        
        {"code": "1275.007", "description_fr": "Matériel bureau amortissements", 
         "description_en": "Office equipment costs depreciations",
         "keywords_fr": "materiel bureau amortissement",
         "keywords_en": "office equipment depreciation",
         "type_names": []},

        # IMPÔTS (1276)
        {"code": "1276.001", "description_fr": "Divers impôts et taxes (hors douanes, IS et RAS sur dividendes)", 
         "description_en": "Misc. taxes (excl. Customs, CIT and WHT on dividends)",
         "keywords_fr": "impot taxe divers",
         "keywords_en": "misc tax",
         "type_names": ["Taxes & Fines"]},

        # SOUS-TRAITANCE (1281-1285)
        {"code": "1281.001", "description_fr": "Sous-traitants génie civil", 
         "description_en": "Civil works subcontractors",
         "keywords_fr": "sous traitance sous traitant genie civil terrassement beton maçonnerie",
         "keywords_en": "subcontractor civil works",
         "type_names": ["Subcontracting", "Subcontractor", "Travaux de construction ou d’aménagement / Construction or development work"]},
        
        {"code": "1282.001", "description_fr": "Sous-traitants électricité", 
         "description_en": "Electrical subcontractors",
         "keywords_fr": "sous traitance electricite electrique cablage installation",
         "keywords_en": "subcontractor electrical",
         "type_names": ["Subcontracting"]},
        
        {"code": "1283.001", "description_fr": "Sous-traitants fibre optique", 
         "description_en": "FO subcontractors",
         "keywords_fr": "sous traitance fibre optique fo tirage soudure fusion",
         "keywords_en": "subcontractor fiber fibre optic",
         "type_names": ["Subcontracting"]},
        
        {"code": "1284.001", "description_fr": "Sous-traitants télécoms", 
         "description_en": "TELCO subcontractors",
         "keywords_fr": "sous traitance telecom antenne pylone installation",
         "keywords_en": "subcontractor telecom telco",
         "type_names": ["Subcontracting"]},
        
        {"code": "1285.001", "description_fr": "Sous-traitants autres", 
         "description_en": "Other subcontractors",
         "keywords_fr": "sous traitance autre divers",
         "keywords_en": "other subcontractor",
         "type_names": ["Subcontracting"]},

        # REFACTURATIONS INTERCO (1311)
        {"code": "1311.001", "description_fr": "Refacturation interco de l'assistance administrative", 
         "description_en": "Interco rebilling of admin assistance",
         "keywords_fr": "refacturation interco assistance administrative",
         "keywords_en": "interco rebilling admin assistance",
         "type_names": []},
        
        {"code": "1311.002", "description_fr": "Refacturation interco des coûts d'appel d'offres et assistance commerciale", 
         "description_en": "Interco rebilling of tender costs and commercial assistance",
         "keywords_fr": "refacturation interco appel offre assistance commerciale",
         "keywords_en": "interco rebilling tender commercial",
         "type_names": []},
        
        {"code": "1311.003", "description_fr": "Refacturation interco des études et assistance technique", 
         "description_en": "Interco rebilling of studies and technical assistance",
         "keywords_fr": "refacturation interco etude assistance technique",
         "keywords_en": "interco rebilling study technical",
         "type_names": []},
        
        {"code": "1311.004", "description_fr": "Autres transferts et refacturations interco de coûts", 
         "description_en": "Interco other transfer and rebilling of costs",
         "keywords_fr": "refacturation interco autre transfert",
         "keywords_en": "interco other rebilling",
         "type_names": []},

        # AUTRES PRODUITS (1312)
        {"code": "1312.001", "description_fr": "Autres produits d'activité annexe", 
         "description_en": "Other operating revenues",
         "keywords_fr": "autre produit activite annexe",
         "keywords_en": "other operating revenue",
         "type_names": []},

        # AJUSTEMENTS STOCKS (1321)
        {"code": "1321.001", "description_fr": "Ajustements divers de stocks opérationnels", 
         "description_en": "Misc stocks adjustments operational",
         "keywords_fr": "ajustement stock operationnel",
         "keywords_en": "stock adjustment operational",
         "type_names": []},

        # PROVISIONS (1322)
        {"code": "1322.001", "description_fr": "Dotations et reprises aux provisions d'exploitation", 
         "description_en": "Operating provisions and reversal",
         "keywords_fr": "dotation reprise provision exploitation",
         "keywords_en": "operating provision reversal",
         "type_names": []},

        # IFRS (1323)
        {"code": "1323.001", "description_fr": "Correctifs de lissage IFRS 15 & PAT", 
         "description_en": "IFRS 15 progress adjustments and loss provision",
         "keywords_fr": "ifrs lissage correction",
         "keywords_en": "ifrs adjustment",
         "type_names": []},

        # AUTRES CHARGES (1324)
        {"code": "1324.001", "description_fr": "Autres charges opérationnelles diverses", 
         "description_en": "Other misc operating costs",
         "keywords_fr": "autre charge operationnelle divers",
         "keywords_en": "other operating cost",
         "type_names": []},

        # FRAIS DE GESTION (1331-1332)
        {"code": "1331.001", "description_fr": "Collecte interco des frais de gestion", 
         "description_en": "Interco collect of management fees",
         "keywords_fr": "collecte interco frais gestion",
         "keywords_en": "interco management fee",
         "type_names": []},
        
        {"code": "1332.001", "description_fr": "Charges de frais de gestion", 
         "description_en": "Management fees costs",
         "keywords_fr": "charge frais gestion",
         "keywords_en": "management fee cost",
         "type_names": []},

        # GAINS/PERTES DE CHANGE OPÉRATIONNELS (1411)
        {"code": "1411.001", "description_fr": "Gain opérationnel de change", 
         "description_en": "Operational FX gain",
         "keywords_fr": "gain operationnel change forex",
         "keywords_en": "operational fx gain",
         "type_names": []},
        
        {"code": "1411.002", "description_fr": "Perte opérationnelle de change", 
         "description_en": "Operational FX loss",
         "keywords_fr": "perte operationnelle change forex",
         "keywords_en": "operational fx loss",
         "type_names": []},

        # DIVERS NON REPORTÉS (1511)
        {"code": "1511.001", "description_fr": "Divers produits et charges opérationnels non reportés", 
         "description_en": "Misc non reporting operating revenues & costs",
         "keywords_fr": "divers non reporte operationnel",
         "keywords_en": "misc non reporting operating",
         "type_names": []},

        # PRODUITS FINANCIERS (2111)
        {"code": "2111.001", "description_fr": "Autres produits financiers", 
         "description_en": "Other financial revenues",
         "keywords_fr": "autre produit financier",
         "keywords_en": "other financial revenue",
         "type_names": []},

        # CHARGES FINANCIÈRES (2121)
        {"code": "2121.001", "description_fr": "Intérêts et autres frais financiers", 
         "description_en": "Interests and other financial charges",
         "keywords_fr": "interet frais financier",
         "keywords_en": "interest financial charge",
         "type_names": []},

        # PROVISIONS FINANCIÈRES (2211)
        {"code": "2211.001", "description_fr": "Dotations et reprises aux provisions financières", 
         "description_en": "Financial provisions and reversal",
         "keywords_fr": "dotation reprise provision financiere",
         "keywords_en": "financial provision reversal",
         "type_names": []},

        # GAINS/PERTES DE CHANGE FINANCIERS (2411)
        {"code": "2411.001", "description_fr": "Gain financier de change", 
         "description_en": "Financial FX gain",
         "keywords_fr": "gain financier change forex",
         "keywords_en": "financial fx gain",
         "type_names": []},
        
        {"code": "2411.002", "description_fr": "Perte financière de change", 
         "description_en": "Financial FX loss",
         "keywords_fr": "perte financiere change forex",
         "keywords_en": "financial fx loss",
         "type_names": []},

        # DIVIDENDES (2511)
        {"code": "2511.001", "description_fr": "Dividendes (élément non reporté)", 
         "description_en": "Dividends (non reporting item)",
         "keywords_fr": "dividende non reporte",
         "keywords_en": "dividend non reporting",
         "type_names": []},
        
        {"code": "2511.002", "description_fr": "Divers produits et charges financiers non reportés", 
         "description_en": "Misc non reporting financial revenues & costs",
         "keywords_fr": "divers financier non reporte",
         "keywords_en": "misc financial non reporting",
         "type_names": []},

        # RESTRUCTURATION (3111)
        {"code": "3111.001", "description_fr": "Frais de restructuration", 
         "description_en": "Restructuring costs",
         "keywords_fr": "frais restructuration reorganisation",
         "keywords_en": "restructuring cost",
         "type_names": []},

        # CESSIONS (3211)
        {"code": "3211.001", "description_fr": "Produits de cession d'immobilisations", 
         "description_en": "Assets resale revenues",
         "keywords_fr": "produit cession immobilisation vente actif",
         "keywords_en": "asset resale revenue",
         "type_names": []},
        
        {"code": "3211.002", "description_fr": "Valeur nette comptable des immobilisations cédées", 
         "description_en": "Assets resale costs (net book value)",
         "keywords_fr": "valeur nette comptable cession immobilisation",
         "keywords_en": "net book value asset sold",
         "type_names": []},

        # AMENDES (3221)
        {"code": "3221.001", "description_fr": "Amendes et pénalités", 
         "description_en": "Penalties",
         "keywords_fr": "amende penalite",
         "keywords_en": "penalty fine",
         "type_names": []},
        
        {"code": "3221.002", "description_fr": "Dotations et reprises aux provisions exceptionnelles", 
         "description_en": "Exceptional provisions and reversal",
         "keywords_fr": "dotation reprise provision exceptionnelle",
         "keywords_en": "exceptional provision reversal",
         "type_names": []},

        # AJUSTEMENTS EXERCICES ANTÉRIEURS (3311)
        {"code": "3311.001", "description_fr": "Ajustements exercices antérieurs", 
         "description_en": "Prior year adjustments",
         "keywords_fr": "ajustement exercice anterieur",
         "keywords_en": "prior year adjustment",
         "type_names": []},
        
        {"code": "3311.002", "description_fr": "Autres produits exceptionnels", 
         "description_en": "Other exceptional revenues",
         "keywords_fr": "autre produit exceptionnel",
         "keywords_en": "other exceptional revenue",
         "type_names": []},
        
        {"code": "3311.003", "description_fr": "Autres charges exceptionnelles", 
         "description_en": "Other exceptional costs",
         "keywords_fr": "autre charge exceptionnelle",
         "keywords_en": "other exceptional cost",
         "type_names": []},

        # GAINS/PERTES EXCEPTIONNELS DE CHANGE (3411)
        {"code": "3411.001", "description_fr": "Gain exceptionnel de change", 
         "description_en": "Exceptional FX gain",
         "keywords_fr": "gain exceptionnel change forex",
         "keywords_en": "exceptional fx gain",
         "type_names": []},
        
        {"code": "3411.002", "description_fr": "Perte exceptionnelle de change", 
         "description_en": "Exceptional FX loss",
         "keywords_fr": "perte exceptionnelle change forex",
         "keywords_en": "exceptional fx loss",
         "type_names": []},

        # DIVERS EXCEPTIONNELS (3511)
        {"code": "3511.001", "description_fr": "Divers produits et charges exceptionnels non reportés", 
         "description_en": "Misc non reporting exceptional revenues & costs",
         "keywords_fr": "divers exceptionnel non reporte",
         "keywords_en": "misc exceptional non reporting",
         "type_names": []},

        # IMPÔTS (4111)
        {"code": "4111.001", "description_fr": "Impôt sur les sociétés & impôt minimum forfaitaire", 
         "description_en": "CIT & Minimum LS tax",
         "keywords_fr": "impot societe is minimum forfaitaire",
         "keywords_en": "corporate income tax cit",
         "type_names": []},
        
        {"code": "4111.002", "description_fr": "Retenue à la source sur dividendes", 
         "description_en": "WHT on dividends",
         "keywords_fr": "retenue source dividende",
         "keywords_en": "withholding tax dividend",
         "type_names": []},
        
        {"code": "4111.003", "description_fr": "Autres impôts et taxes non opérationnels", 
         "description_en": "Other non operating taxes",
         "keywords_fr": "autre impot taxe non operationnel",
         "keywords_en": "other non operating tax",
         "type_names": []},
        # CONSEIL ET HONORAIRES INFORMATIQUES
        {"code": "1262.005", "description_fr": "Conseil et honoraires informatiques", 
         "description_en": "IT consultancy and fees",
         "keywords_fr": "conseil consultation honoraires informatique software logiciel",
         "keywords_en": "consultancy fees it software",
         "type_names": ["IT Services", "Consultancy Fees", "IT Services", "Software"]},

        # MAINTENANCE ÉQUIPEMENTS INFORMATIQUES
        {"code": "1244.003", "description_fr": "Maintenance équipements informatiques", 
         "description_en": "IT equipment maintenance",
         "keywords_fr": "maintenance réparation laptop ordinateur imprimante desktop pc",
         "keywords_en": "maintenance repair laptop computer printer desktop",
         "type_names": ["IT Services", "Maintenance", "IT consumables Admin"]},

        # INSTALLATION SYSTÈMES DE SÉCURITÉ
        {"code": "1224.004", "description_fr": "Installation système vidéo surveillance", 
         "description_en": "CCTV installation",
         "keywords_fr": "installation camera vidéo surveillance cctv",
         "keywords_en": "installation camera cctv surveillance",
         "type_names": ["IT Services", "Security", "Remote motoring system"]},

        # SERVICES DE MANUTENTION
        {"code": "1231.003", "description_fr": "Services de manutention", 
         "description_en": "Handling services",
         "keywords_fr": "manutention handling levage service",
         "keywords_en": "handling lifting service",
         "type_names": ["Maintenance", "Handling material", "Small Tools & Equipments"]},

        # ÉTANCHÉITÉ
        {"code": "1221.006", "description_fr": "Travaux d'étanchéité", 
         "description_en": "Waterproofing works",
         "keywords_fr": "étanchéité waterproof etancheite toiture",
         "keywords_en": "waterproofing sealing roof",
         "type_names": ["Maintenance Admin", "Building / Chantier"]}
    ]
    
    # Charger les codes de base
    base_codes = [
        {"code": "1111.001", "description_fr": "Ventes externes de matériaux uniquement", "description_en": "External sales of materials only"},
        {"code": "1111.002", "description_fr": "Ventes externes de services uniquement", "description_en": "External sales of services only"},
        {"code": "1111.003", "description_fr": "Ventes externes de matériaux et services", "description_en": "External sales of materials and services"},
        {"code": "1112.001", "description_fr": "Ventes interco de matériaux uniquement", "description_en": "Interco sales of materials only"},
        {"code": "1112.002", "description_fr": "Ventes interco de services uniquement", "description_en": "Interco sales of services only"},
        {"code": "1112.003", "description_fr": "Ventes interco de matériaux et services", "description_en": "Interco sales of materials and services"},
        {"code": "1211.001", "description_fr": "Main d'œuvre", "description_en": "Internal manpower (blue-collar and hourly paid staff)"},
        {"code": "1211.002", "description_fr": "Intérim horaires", "description_en": "External manpower (blue-collar and hourly paid staff)"},
        {"code": "1212.001", "description_fr": "Salaires encadrement expat", "description_en": "Internal expat supervision and executive staff"},
        {"code": "1212.002", "description_fr": "Salaires encadrement local", "description_en": "Internal local supervision and executive staff"},
        {"code": "1212.003", "description_fr": "Encadrement intérimaire", "description_en": "External supervision and executive staff"},
        {"code": "1213.001", "description_fr": "Missions et voyages", "description_en": "Missions and travel costs"},
        {"code": "1213.002", "description_fr": "Notes de frais et réceptions", "description_en": "Expense claims and receptions costs"},
        {"code": "1213.003", "description_fr": "Formation", "description_en": "Training costs"},
        {"code": "1213.004", "description_fr": "Assurance santé, médecine du travail et autres frais médicaux", "description_en": "Health insurance and other health costs"},
        {"code": "1213.005", "description_fr": "Frais divers personnel", "description_en": "Misc. HR costs"},
        {"code": "1213.006", "description_fr": "Restauration", "description_en": "Catering costs"},
        {"code": "1213.007", "description_fr": "Loyers & charges locatives terrains & locaux (habitation)", "description_en": "Rents, land & premises and related expenses (living)"},
        {"code": "1221.001", "description_fr": "Béton prêt à l'emploi", "description_en": "Ready-mix concrete"},
        {"code": "1221.002", "description_fr": "Cailloux, sables et autres matériaux de remblai", "description_en": "Rocks, sands and other backfill materials"},
        {"code": "1221.003", "description_fr": "Aciers pour armatures et armatures façonnées", "description_en": "Steel rebars and shaped steel frame"},
        {"code": "1221.004", "description_fr": "Parpaings & briques", "description_en": "Bricks and building blocks"},
        {"code": "1221.005", "description_fr": "Géotextiles", "description_en": "Geotextiles"},
        {"code": "1221.006", "description_fr": "Étanchéité", "description_en": "Sealing"},
        {"code": "1221.007", "description_fr": "Divers préfabriqués béton", "description_en": "Misc precasted concrete elements"},
        {"code": "1221.008", "description_fr": "Tuyaux fonte", "description_en": "Cast iron pipes"},
        {"code": "1221.009", "description_fr": "Tuyaux PVC", "description_en": "PVC pipes"},
        {"code": "1221.010", "description_fr": "Grillage", "description_en": "Fence"},
        {"code": "1221.011", "description_fr": "Bois", "description_en": "Wood"},
        {"code": "1221.012", "description_fr": "Divers matériaux GC incorporés", "description_en": "Miscellaneous civil works incorporated materials"},
        {"code": "1222.001", "description_fr": "Pylônes telecom greenfield", "description_en": "Pylon telecom greenfield"},
        {"code": "1222.002", "description_fr": "Pylônes telecom toit-terrasse", "description_en": "Pylon telecom rooftop"},
        {"code": "1222.003", "description_fr": "Pylônes électriques", "description_en": "Electric pylons"},
        {"code": "1222.004", "description_fr": "Support panneaux solaires", "description_en": "Solar panel supports"},
        {"code": "1222.005", "description_fr": "Profilés métalliques", "description_en": "Metal profiles and beams"},
        {"code": "1222.006", "description_fr": "Ancrage, barres, boulons", "description_en": "Anchorage, rods, bolts and nuts"},
        {"code": "1222.007", "description_fr": "Câbles métalliques / aciers", "description_en": "Steel/metal ropes and cables"},
        {"code": "1222.008", "description_fr": "Tôles métalliques", "description_en": "Metal sheets"},
        {"code": "1222.009", "description_fr": "Produits et accessoires anti-corrosion", "description_en": "Anti corrosive products and accessories"},
        {"code": "1222.010", "description_fr": "Divers matériaux métalliques incorporés", "description_en": "Miscellaneous metallic incorporated materials"},
        {"code": "1223.001", "description_fr": "Poteaux bois", "description_en": "Wooden poles"},
        {"code": "1223.002", "description_fr": "Poteaux béton", "description_en": "Concrete poles"},
        {"code": "1223.003", "description_fr": "Poteaux métalliques", "description_en": "Metal poles"},
        {"code": "1223.004", "description_fr": "Connecteurs FO", "description_en": "FO connectors"},
        {"code": "1223.005", "description_fr": "Câble FO", "description_en": "Cable FO"},
        {"code": "1223.006", "description_fr": "Boîtier FO", "description_en": "Box FO"},
        {"code": "1223.007", "description_fr": "Panneaux et armoires FO", "description_en": "FO panels and cabinets"},
        {"code": "1223.008", "description_fr": "Divers matériaux FO incorporés", "description_en": "Miscellaneous FO incorporated materials"},
        {"code": "1224.001", "description_fr": "Système protection anti-incendie", "description_en": "Fire protection system"},
        {"code": "1224.002", "description_fr": "Lampadaires", "description_en": "Streetlights"},
        {"code": "1224.003", "description_fr": "Système d'éclairage", "description_en": "Lighting systems"},
        {"code": "1224.004", "description_fr": "Système vidéo surveillance", "description_en": "CCTV systems"},
        {"code": "1224.005", "description_fr": "Pompes et compresseurs industriels", "description_en": "Industrial pumps and compressors"},
        {"code": "1224.006", "description_fr": "Instrumentation et système de mesure", "description_en": "Instrumentation and metering systems"},
        {"code": "1224.007", "description_fr": "Système de climatisation et refroidissement", "description_en": "Cooling and air conditioning systems"},
        {"code": "1224.008", "description_fr": "Autres équipements industriels incorporés", "description_en": "Other incorporated industrial equipment"},
        {"code": "1225.001", "description_fr": "Groupes électrogènes", "description_en": "Generators"},
        {"code": "1225.002", "description_fr": "Cuves à gasoil", "description_en": "Fuel tanks"},
        {"code": "1225.003", "description_fr": "Inverseurs & ATS", "description_en": "Inverters & ATS"},
        {"code": "1225.004", "description_fr": "Batteries", "description_en": "Batteries"},
        {"code": "1225.005", "description_fr": "Panneaux solaires", "description_en": "Solar panels"},
        {"code": "1225.006", "description_fr": "Redresseurs", "description_en": "Rectifiers"},
        {"code": "1225.007", "description_fr": "Câbles électriques", "description_en": "Electrical cables"},
        {"code": "1225.008", "description_fr": "Paratonnerre et matériel de mise à la terre", "description_en": "Lightning rods and grounding equipment"},
        {"code": "1225.009", "description_fr": "Compteurs et armoires électriques", "description_en": "Electrical panels and cabinets"},
        {"code": "1225.010", "description_fr": "Divers fournitures électriques incorporées", "description_en": "Misc electrical incorporated materials"},
        {"code": "1226.001", "description_fr": "Divers équipements télécoms actifs", "description_en": "Misc active telecom equipment"},
        {"code": "1227.001", "description_fr": "PDR pour entretien courant et réparations diverses (sites clients)", "description_en": "Spare parts for daily and minor unplanned maintenance (customers sites)"},
        {"code": "1227.002", "description_fr": "PDR pour grande maintenance planifiée & entretien majeur (sites clients)", "description_en": "Spare parts for major and planned maintenance (customers sites)"},
        {"code": "1227.003", "description_fr": "PDR pour entretien & réparation suite accident ou casse (sites clients)", "description_en": "Repairs after accident and breakdown (customers sites)"},
        {"code": "1227.004", "description_fr": "Huiles & graisses pour maintenance (sites clients)", "description_en": "Oils, lubricants & grease for maintenance (customers sites)"},
        {"code": "1227.005", "description_fr": "Divers matériaux et consommables pour la maintenance des sites clients", "description_en": "Misc materials and consumables for customers sites"},
        {"code": "1228.001", "description_fr": "Autres équipements réseaux", "description_en": "Misc network equipment"},
        {"code": "1228.002", "description_fr": "Peinture", "description_en": "Painting"},
        {"code": "1228.003", "description_fr": "Autres matériaux incorporés", "description_en": "Other incorporated materials"},
        {"code": "1229.001", "description_fr": "Carburant pour refueling des sites clients", "description_en": "Customers sites refueling"},
        {"code": "1231.001", "description_fr": "Petit outillage", "description_en": "Other tools and small equipment"},
        {"code": "1231.002", "description_fr": "Consommable, outillage de soudure", "description_en": "Welding tools and consumables"},
        {"code": "1231.003", "description_fr": "Accessoires et outils de levage", "description_en": "Lifting and handling tools and accessories"},
        {"code": "1231.004", "description_fr": "Consommables coffrages", "description_en": "Formwork consumables"},
        {"code": "1232.001", "description_fr": "Échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails"},
        {"code": "1232.002", "description_fr": "Autres outils temporaires", "description_en": "Other temporary tools"},
        {"code": "1233.001", "description_fr": "Consommables environnement", "description_en": "Environment consumables"},
        {"code": "1233.002", "description_fr": "Matériel signalisation", "description_en": "Signs, buoys and anchors"},
        {"code": "1233.003", "description_fr": "Équipement de Protection Individuel (EPI)", "description_en": "PPE and safety expenses"},
        {"code": "1234.001", "description_fr": "Eau", "description_en": "Water"},
        {"code": "1234.002", "description_fr": "Divers consommables", "description_en": "Other misc. consumables"},
        {"code": "1241.001", "description_fr": "Location externe voiture, pickup & suv", "description_en": "Car, pickup & suv external rental"},
        {"code": "1241.002", "description_fr": "Location externe motos & triporteurs", "description_en": "Motorbikes & three wheelers external rental"},
        {"code": "1241.003", "description_fr": "Location externe véhicules pour refueling", "description_en": "Refueling vehicles external rental"},
        {"code": "1241.004", "description_fr": "Location externe autres camions", "description_en": "Other trucks external rental"},
        {"code": "1241.005", "description_fr": "Location externe grue mobile", "description_en": "Mobile crane external rental"},
        {"code": "1241.006", "description_fr": "Location externe autres matériel de levage et manutention", "description_en": "Other lifting and handling equipment external rental"},
        {"code": "1241.007", "description_fr": "Location externe de pelle", "description_en": "HEX external rental"},
        {"code": "1241.008", "description_fr": "Location externe de chargeur", "description_en": "Wheel loader external rental"},
        {"code": "1241.009", "description_fr": "Location externe matériel compactage", "description_en": "Compaction equipment external rental"},
        {"code": "1241.010", "description_fr": "Location externe matériel de réglage", "description_en": "Grader external rental"},
        {"code": "1241.011", "description_fr": "Location externe autre matériel de terrassement", "description_en": "Other earth-moving equipment external rental"},
        {"code": "1241.012", "description_fr": "Location externe échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails external rental"},
        {"code": "1241.013", "description_fr": "Location externe matériel production énergie et éclairage", "description_en": "Generator and lighting equipment external rental"},
        {"code": "1241.014", "description_fr": "Location externe matériel de forage et passage de câbles", "description_en": "Drilling and cable routing equipment external rental"},
        {"code": "1241.015", "description_fr": "Location externe matériel divers", "description_en": "Other misc equipment external rental"},
        {"code": "1241.016", "description_fr": "Location externe outillage et machines FO", "description_en": "Other tooling and machinery FO external rental"},
        {"code": "1241.017", "description_fr": "Location externe autres outillage et machines", "description_en": "Other tooling and machinery external rental"},
        {"code": "1242.001", "description_fr": "Amortissements voiture, pickup & suv", "description_en": "Car, pickup & suv depreciations"},
        {"code": "1242.002", "description_fr": "Amortissements motos & triporteurs", "description_en": "Motorbikes & three wheelers depreciations"},
        {"code": "1242.003", "description_fr": "Amortissements véhicules pour refueling", "description_en": "Refueling vehicles depreciations"},
        {"code": "1242.004", "description_fr": "Amortissements autres camions", "description_en": "Other trucks depreciations"},
        {"code": "1242.005", "description_fr": "Amortissements grue mobile", "description_en": "Mobile crane depreciations"},
        {"code": "1242.006", "description_fr": "Amortissements autres matériel de levage et manutention", "description_en": "Other lifting and handling equipment depreciations"},
        {"code": "1242.007", "description_fr": "Amortissements de pelle", "description_en": "HEX depreciations"},
        {"code": "1242.008", "description_fr": "Amortissements de chargeur", "description_en": "Wheel loader depreciations"},
        {"code": "1242.009", "description_fr": "Amortissements matériel compactage", "description_en": "Compaction equipment depreciations"},
        {"code": "1242.010", "description_fr": "Amortissements matériel de réglage", "description_en": "Grader depreciations"},
        {"code": "1242.011", "description_fr": "Amortissements autre matériel de terrassement", "description_en": "Other earth-moving equipment depreciations"},
        {"code": "1242.012", "description_fr": "Amortissements échafaudage et garde-corps", "description_en": "Scaffoldings and guardrails depreciations"},
        {"code": "1242.013", "description_fr": "Amortissements matériel production énergie et éclairage", "description_en": "Generator and lighting equipment depreciations"},
        {"code": "1242.014", "description_fr": "Amortissements matériel de forage et passage de câbles", "description_en": "Drilling and cable routing equipment depreciations"},
        {"code": "1242.015", "description_fr": "Amortissements matériel divers", "description_en": "Other misc equipment depreciations"},
        {"code": "1242.016", "description_fr": "Amortissements outillage et machines FO", "description_en": "Other tooling and machinery FO depreciations"},
        {"code": "1242.017", "description_fr": "Amortissements autres outillage et machines", "description_en": "Other tooling and machinery depreciations"},
        {"code": "1243.001", "description_fr": "Essence / Diesel pour VL", "description_en": "Petrol and diesel for LV"},
        {"code": "1243.002", "description_fr": "Essence / Diesel pour autre usage interne", "description_en": "Petrol and diesel for other internal use"},
        {"code": "1243.003", "description_fr": "Huiles & graisses pour usage interne", "description_en": "Oils, lubricants & grease for internal use"},
        {"code": "1244.001", "description_fr": "Entretiens, réparations et PDR pour VL", "description_en": "Maintenances, repairs and spare parts for LV"},
        {"code": "1244.002", "description_fr": "Entretiens, réparations et PDR pour autres équipements internes", "description_en": "Maintenances, repairs and spare parts for other internal equipment"},
        {"code": "1251.001", "description_fr": "Import - fret aérien OPEX", "description_en": "Import - air freight OPEX"},
        {"code": "1251.002", "description_fr": "Import - fret maritime OPEX", "description_en": "Import - sea freight OPEX"},
        {"code": "1251.003", "description_fr": "Import - fret aérien CAPEX", "description_en": "Import - air freight CAPEX"},
        {"code": "1251.004", "description_fr": "Import - fret maritime CAPEX", "description_en": "Import - sea freight CAPEX"},
        {"code": "1252.001", "description_fr": "Export - fret aérien OPEX", "description_en": "Export - air freight OPEX"},
        {"code": "1252.002", "description_fr": "Export - fret maritime OPEX", "description_en": "Export - sea freight OPEX"},
        {"code": "1252.003", "description_fr": "Export - fret aérien CAPEX", "description_en": "Export - air freight CAPEX"},
        {"code": "1252.004", "description_fr": "Export - fret maritime CAPEX", "description_en": "Export - sea freight CAPEX"},
        {"code": "1253.001", "description_fr": "Import & Export - droits de douane et coûts associés OPEX", "description_en": "Import & Export - custom duties and related costs OPEX"},
        {"code": "1253.002", "description_fr": "Import & Export - droits de douane et coûts associés CAPEX", "description_en": "Import & Export - custom duties and related costs CAPEX"},
        {"code": "1254.001", "description_fr": "Assurances transport OPEX", "description_en": "Transport insurances OPEX"},
        {"code": "1254.002", "description_fr": "Assurances transport CAPEX", "description_en": "Transport insurances CAPEX"},
        {"code": "1254.003", "description_fr": "Autres frais de transport OPEX", "description_en": "Other transportation costs OPEX"},
        {"code": "1254.004", "description_fr": "Autres frais de transport CAPEX", "description_en": "Other transportation costs CAPEX"},
        {"code": "1261.001", "description_fr": "Frais d'études techniques externes (design and methods)", "description_en": "External technical studies (design and methods)"},
        {"code": "1261.002", "description_fr": "Géomètres, géotechniciens et autres techniciens extérieurs", "description_en": "External topographic, geotechnical and other technical surveys"},
        {"code": "1261.003", "description_fr": "Frais de laboratoire", "description_en": "Laboratory studies costs"},
        {"code": "1261.004", "description_fr": "Frais d'études partenaires de groupement", "description_en": "Studies and surveys partner JV and consortium"},
        {"code": "1262.001", "description_fr": "Frais internes études techniques", "description_en": "Internal technical study costs"},
        {"code": "1262.002", "description_fr": "Frais tendering internes", "description_en": "Internal tendering costs"},
        {"code": "1262.003", "description_fr": "Frais d'achats et logistique internes", "description_en": "Procurement, logistics and post-order internal services"},
        {"code": "1262.004", "description_fr": "Autres refacturations internes de services", "description_en": "Other costs internal services"},
        {"code": "1263.001", "description_fr": "Frais contrôle technique / homologation diverse", "description_en": "Certifications costs"},
        {"code": "1264.001", "description_fr": "Autres prestations", "description_en": "Other services provided"},
        {"code": "1271.001", "description_fr": "Loyers & charges locatives terrains & locaux (pro)", "description_en": "Rents, land & premises and related expenses (pro)"},
        {"code": "1271.002", "description_fr": "Location externe bungalows et divers installations pro", "description_en": "Site facilities and bungalows external rental"},
        {"code": "1271.003", "description_fr": "Amortissements bungalows et divers installations pro", "description_en": "Site facilities and bungalows depreciations"},
        {"code": "1271.004", "description_fr": "Électricité, gaz et eau des installations pro", "description_en": "Electricity, gas and water of sites facilities"},
        {"code": "1271.005", "description_fr": "Gardiennage, frais de sécurité", "description_en": "Security expenses"},
        {"code": "1271.006", "description_fr": "Autres coûts d'installations et d'infrastructures OPEX", "description_en": "Other installations and facilities costs OPEX"},
        {"code": "1271.007", "description_fr": "Autres coûts d'installations et d'infrastructures - amortissements", "description_en": "Other installations and facilities costs depreciations"},
        {"code": "1271.008", "description_fr": "Mise en décharge matériaux divers et prestations environnementales", "description_en": "Landfilling activity & other environmental service"},
        {"code": "1271.009", "description_fr": "Prestation, entretien et nettoyage", "description_en": "Cleaning, maintenance and garbage disposals"},
        {"code": "1272.001", "description_fr": "Frais d'actes et contentieux", "description_en": "Deeds and disputes charges"},
        {"code": "1272.002", "description_fr": "Auditeurs", "description_en": "Auditors"},
        {"code": "1272.003", "description_fr": "Conseil fiscal", "description_en": "Tax consultant"},
        {"code": "1272.004", "description_fr": "Documentation & traduction", "description_en": "Documentation and translation costs"},
        {"code": "1272.005", "description_fr": "Autres conseils et honoraires", "description_en": "Other consultancy and fees"},
        {"code": "1273.001", "description_fr": "Frais de télécommunication", "description_en": "Telecommunication costs"},
        {"code": "1273.002", "description_fr": "Coûts informatiques - matériel OPEX", "description_en": "IT costs hardware OPEX"},
        {"code": "1273.003", "description_fr": "Coûts informatiques - logiciels et licences OPEX", "description_en": "IT costs software and licenses OPEX"},
        {"code": "1273.004", "description_fr": "Coûts informatiques - amortissements du matériel", "description_en": "IT costs hardware depreciations"},
        {"code": "1273.005", "description_fr": "Coûts informatiques - amortissements des logiciels et licences", "description_en": "IT costs software and licenses depreciations"},
        {"code": "1274.001", "description_fr": "Assurances véhicules légers", "description_en": "LV insurances"},
        {"code": "1274.002", "description_fr": "Assurance tous risques chantier", "description_en": "Contractor's all risks insurance"},
        {"code": "1274.003", "description_fr": "Assurances multirisques", "description_en": "Multi-risks insurances"},
        {"code": "1274.004", "description_fr": "Autres assurances", "description_en": "Other insurances"},
        {"code": "1274.005", "description_fr": "Assurance bris de machine", "description_en": "Machinery breakdown insurance"},
        {"code": "1274.006", "description_fr": "Frais bancaires", "description_en": "Banking charges"},
        {"code": "1274.007", "description_fr": "Frais de garantie et cautions", "description_en": "Bank guarantees and bonds costs"},
        {"code": "1275.001", "description_fr": "Fournitures de bureau et consommables informatiques", "description_en": "Office consumables and supplies"},
        {"code": "1275.002", "description_fr": "Charges administratives diverses", "description_en": "Miscellaneous administrative charges"},
        {"code": "1275.003", "description_fr": "Business promotion", "description_en": "Business promotion"},
        {"code": "1275.004", "description_fr": "Dépenses RSE", "description_en": "RSE & CSR expenses"},
        {"code": "1275.005", "description_fr": "Publicité, annonces insertions", "description_en": "Advertising expenses"},
        {"code": "1275.006", "description_fr": "Matériel bureau OPEX", "description_en": "Office equipment costs OPEX"},
        {"code": "1275.007", "description_fr": "Matériel bureau amortissements", "description_en": "Office equipment costs depreciations"},
        {"code": "1276.001", "description_fr": "Divers impôts et taxes (hors douanes, IS et RAS sur dividendes)", "description_en": "Misc. taxes (excl. Customs, CIT and WHT on dividends)"},
        {"code": "1281.001", "description_fr": "Sous-traitants génie civil", "description_en": "Civil works subcontractors"},
        {"code": "1282.001", "description_fr": "Sous-traitants électricité", "description_en": "Electrical subcontractors"},
        {"code": "1283.001", "description_fr": "Sous-traitants fibre optique", "description_en": "FO subcontractors"},
        {"code": "1284.001", "description_fr": "Sous-traitants télécoms", "description_en": "TELCO subcontractors"},
        {"code": "1285.001", "description_fr": "Sous-traitants autres", "description_en": "Other subcontractors"},
        {"code": "1311.001", "description_fr": "Refacturation interco de l'assistance administrative", "description_en": "Interco rebilling of admin assistance"},
        {"code": "1311.002", "description_fr": "Refacturation interco des coûts d'appel d'offres et assistance commerciale", "description_en": "Interco rebilling of tender costs and commercial assistance"},
        {"code": "1311.003", "description_fr": "Refacturation interco des études et assistance technique", "description_en": "Interco rebilling of studies and technical assistance"},
        {"code": "1311.004", "description_fr": "Autres transferts et refacturations interco de coûts", "description_en": "Interco other transfer and rebilling of costs"},
        {"code": "1312.001", "description_fr": "Autres produits d'activité annexe", "description_en": "Other operating revenues"},
        {"code": "1321.001", "description_fr": "Ajustements divers de stocks opérationnels", "description_en": "Misc stocks adjustments operational"},
        {"code": "1322.001", "description_fr": "Dotations et reprises aux provisions d'exploitation", "description_en": "Operating provisions and reversal"},
        {"code": "1323.001", "description_fr": "Correctifs de lissage IFRS 15 & PAT", "description_en": "IFRS 15 progress adjustments and loss provision"},
        {"code": "1324.001", "description_fr": "Autres charges opérationnelles diverses", "description_en": "Other misc operating costs"},
        {"code": "1331.001", "description_fr": "Collecte interco des frais de gestion", "description_en": "Interco collect of management fees"},
        {"code": "1332.001", "description_fr": "Charges de frais de gestion", "description_en": "Management fees costs"},
        {"code": "1411.001", "description_fr": "Gain opérationnel de change", "description_en": "Operational FX gain"},
        {"code": "1411.002", "description_fr": "Perte opérationnelle de change", "description_en": "Operational FX loss"},
        {"code": "1511.001", "description_fr": "Divers produits et charges opérationnels non reportés", "description_en": "Misc non reporting operating revenues & costs"},
        {"code": "2111.001", "description_fr": "Autres produits financiers", "description_en": "Other financial revenues"},
        {"code": "2121.001", "description_fr": "Intérêts et autres frais financiers", "description_en": "Interests and other financial charges"},
        {"code": "2211.001", "description_fr": "Dotations et reprises aux provisions financières", "description_en": "Financial provisions and reversal"},
        {"code": "2411.001", "description_fr": "Gain financier de change", "description_en": "Financial FX gain"},
        {"code": "2411.002", "description_fr": "Perte financière de change", "description_en": "Financial FX loss"},
        {"code": "2511.001", "description_fr": "Dividendes (élément non reporté)", "description_en": "Dividends (non reporting item)"},
        {"code": "2511.002", "description_fr": "Divers produits et charges financiers non reportés", "description_en": "Misc non reporting financial revenues & costs"},
        {"code": "3111.001", "description_fr": "Frais de restructuration", "description_en": "Restructuring costs"},
        {"code": "3211.001", "description_fr": "Produits de cession d'immobilisations", "description_en": "Assets resale revenues"},
        {"code": "3211.002", "description_fr": "Valeur nette comptable des immobilisations cédées", "description_en": "Assets resale costs (net book value)"},
        {"code": "3221.001", "description_fr": "Amendes et pénalités", "description_en": "Penalties"},
        {"code": "3221.002", "description_fr": "Dotations et reprises aux provisions exceptionnelles", "description_en": "Exceptional provisions and reversal"},
        {"code": "3311.001", "description_fr": "Ajustements exercices antérieurs", "description_en": "Prior year adjustments"},
        {"code": "3311.002", "description_fr": "Autres produits exceptionnels", "description_en": "Other exceptional revenues"},
        {"code": "3311.003", "description_fr": "Autres charges exceptionnelles", "description_en": "Other exceptional costs"},
        {"code": "3411.001", "description_fr": "Gain exceptionnel de change", "description_en": "Exceptional FX gain"},
        {"code": "3411.002", "description_fr": "Perte exceptionnelle de change", "description_en": "Exceptional FX loss"},
        {"code": "3511.001", "description_fr": "Divers produits et charges exceptionnels non reportés", "description_en": "Misc non reporting exceptional revenues & costs"},
        {"code": "4111.001", "description_fr": "Impôt sur les sociétés & impôt minimum forfaitaire", "description_en": "CIT & Minimum LS tax"},
        {"code": "4111.002", "description_fr": "Retenue à la source sur dividendes", "description_en": "WHT on dividends"},
        {"code": "4111.003", "description_fr": "Autres impôts et taxes non opérationnels", "description_en": "Other non operating taxes"},

      ]
    
    # Convertir en DataFrame
    df = pd.DataFrame(data)
    
    # Ajouter les colonnes keywords si elles n'existent pas
    if 'keywords_fr' not in df.columns:
        df['keywords_fr'] = ''
    if 'keywords_en' not in df.columns:
        df['keywords_en'] = ''
    
    # Mettre à jour les keywords pour les codes que nous avons enrichis
    enriched_data = {item['code']: item for item in data}
    for idx, row in df.iterrows():
        if row['code'] in enriched_data:
            enriched = enriched_data[row['code']]
            df.at[idx, 'keywords_fr'] = enriched.get('keywords_fr', '')
            df.at[idx, 'keywords_en'] = enriched.get('keywords_en', '')
    
    return df

# ================================================================
# === FONCTIONS DE NETTOYAGE DE TEXTE
# ================================================================
def clean_text(text):
    """Nettoie un texte pour la recherche"""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text):
    """Extrait les mots-clés significatifs d'un texte"""
    if not text or pd.isna(text):
        return []
    words = clean_text(text).split()
    # Garder les mots de plus de 2 caractères
    return [w for w in words if len(w) > 2]

# ================================================================
# === CLASSIFICATEUR COMPTABLE AMÉLIORÉ
# ================================================================
class AccountingClassifier:
    def __init__(self, codes_df):
        self.codes_df = codes_df
        self.prepare_search_index()
    
    def prepare_search_index(self):
        """Prépare l'index de recherche pour les codes comptables avec mots-clés enrichis"""
        self.search_index = []
        for _, row in self.codes_df.iterrows():
            # Texte de recherche combiné (français + anglais)
            search_text = f"{row['description_fr']} {row['description_en']}"
            
            # Ajouter les mots-clés enrichis
            if 'keywords_fr' in row and row['keywords_fr'] and not pd.isna(row['keywords_fr']):
                search_text += f" {row['keywords_fr']}"
            if 'keywords_en' in row and row['keywords_en'] and not pd.isna(row['keywords_en']):
                search_text += f" {row['keywords_en']}"
            
            # Ajouter les type_names (qui sont des listes)
            type_names = row.get('type_names', [])
            if isinstance(type_names, list) and len(type_names) > 0:
                search_text += f" {' '.join(type_names)}"
            
            keywords = extract_keywords(search_text)
            code_prefix = row['code'].split('.')[0] if '.' in row['code'] else row['code']
            
            # Stocker séparément les mots-clés spécifiques
            specific_keywords = []
            if 'keywords_fr' in row and row['keywords_fr'] and not pd.isna(row['keywords_fr']):
                specific_keywords.extend(extract_keywords(row['keywords_fr']))
            if 'keywords_en' in row and row['keywords_en'] and not pd.isna(row['keywords_en']):
                specific_keywords.extend(extract_keywords(row['keywords_en']))
                specific_keywords.extend(extract_keywords(row['keywords_en']))
            
            self.search_index.append({
                'code': row['code'],
                'description_fr': row['description_fr'],
                'description_en': row['description_en'],
                'keywords': keywords,
                'specific_keywords': specific_keywords,
                'code_prefix': code_prefix,
                'type_names': row.get('type_names', []),
                'search_text': search_text.lower()
            })
    
    def calculate_score(self, item_text, code_entry):
        """Calcule un score de correspondance entre un item et un code avec pondération"""
        item_words = set(extract_keywords(item_text))
        code_keywords = set(code_entry['keywords'])
        specific_keywords = set(code_entry['specific_keywords'])
        
        # Correspondance exacte (poids normal)
        exact_matches = item_words.intersection(code_keywords)
        exact_score = len(exact_matches) * 5
        
        # Correspondance exacte sur les mots-clés spécifiques (poids plus élevé)
        specific_matches = item_words.intersection(specific_keywords)
        specific_score = len(specific_matches) * 20
        
        # Correspondance partielle
        partial_score = 0
        for item_word in item_words:
            for keyword in code_keywords:
                if len(item_word) > 3 and len(keyword) > 3:
                    if keyword in item_word or item_word in keyword:
                        partial_score += 2
                        break
        
        # Bonus pour le préfixe du code
        bonus = 0
        clean_item_text = clean_text(item_text)
        if code_entry['code_prefix'] in clean_item_text:
            bonus += 10
        
        # Bonus pour les termes très spécifiques à Netis
        netis_terms = {
            '1225': ['groupe', 'electrogene', 'generator', 'genset', 'fuel', 'gasoil', 'batterie', 'panneau solaire', 'wd40', 'gen'],
            '1222': ['pylone', 'tower', 'greenfield', 'rooftop', 'antenne'],
            '1223': ['fibre', 'optique', 'fo', 'connecteur', 'cable', 'brin', 'fusion', 'splicer', 'pigtail'],
            '1281': ['genie civil', 'terrassement', 'construction'],
            '1282': ['electricite', 'electrique', 'cablage'],
            '1283': ['fibre', 'optique', 'fo'],
            '1244': ['moteur', 'vehicule', 'spare', 'parts'],
            '1273': ['informatique', 'it', 'hardware', 'software', 'ordinateur'],
        }
        
        for prefix, terms in netis_terms.items():
            if code_entry['code_prefix'] == prefix:
                for term in terms:
                    if term in clean_item_text:
                        bonus += 15
                        break
        
        total_score = exact_score + specific_score + partial_score + bonus
        
        # Normalisation
        confidence = min(total_score / 80, 1.0) if total_score > 0 else 0
        
        return {
            'score': total_score,
            'confidence': confidence,
            'matches': list(exact_matches)[:5],
            'specific_matches': list(specific_matches)[:5]
        }
        if code_entry['code_prefix'] in clean_item_text:
            bonus += 10
        
        # Bonus pour les termes très spécifiques à Netis
        netis_terms = {
            '1225': ['groupe', 'electrogene', 'generator', 'genset', 'fuel', 'gasoil', 'batterie', 'panneau solaire', 'wd40'],
            '1222': ['pylone', 'tower', 'greenfield', 'rooftop', 'antenne'],
            '1223': ['fibre', 'optique', 'fo', 'connecteur', 'cable', 'brin', 'fusion', 'splicer'],
            '1283': ['sous traitant fibre', 'fo subcontractor'],
        }
        
        for prefix, terms in netis_terms.items():
            if code_entry['code_prefix'] == prefix:
                for term in terms:
                    if term in clean_item_text:
                        bonus += 15
                        break
        
        total_score = exact_score + specific_score + partial_score + bonus
        
        # Normalisation (max théorique ajusté)
        confidence = min(total_score / 80, 1.0) if total_score > 0 else 0
        
        return {
            'score': total_score,
            'confidence': confidence,
            'matches': list(exact_matches)[:5],
            'specific_matches': list(specific_matches)[:5]
        }
    
    def classify_item(self, item_name, french_name="", category="", type_name=""):
        """Classifie un item unique avec priorité au type_name"""
        
        # NETTOYAGE DES VALEURS
        # ---------------------
        french_name_clean = ""
        if french_name and str(french_name).lower() != 'nan' and str(french_name).strip():
            french_name_clean = str(french_name).strip().lower()
        
        item_name_clean = ""
        if item_name and str(item_name).lower() != 'nan' and str(item_name).strip():
            item_name_clean = str(item_name).strip().lower()
        
        category_clean = ""
        if category and str(category).lower() != 'nan' and str(category).strip():
            category_clean = str(category).strip().lower()
        
        type_name_clean = ""
        if type_name and str(type_name).lower() != 'nan' and str(type_name).strip():
            type_name_clean = str(type_name).strip().lower()
        
        # ÉTAPE 1 : RECHERCHE PAR TYPE_NAME (PRIORITÉ MAXIMALE)
        # -----------------------------------------------------
        if type_name_clean:
            best_type_match = None
            best_type_score = 0
            
            for code_entry in self.search_index:
                # Vérifier si le code a des type_names associés
                if 'type_names' in code_entry and code_entry['type_names']:
                    # type_names est une liste, la joindre en chaîne
                    if isinstance(code_entry['type_names'], list):
                        code_type_names = ' '.join(code_entry['type_names']).lower()
                    else:
                        code_type_names = str(code_entry['type_names']).lower()
                    
                    # Correspondance exacte (poids très élevé)
                    if type_name_clean in code_type_names:
                        # Calculer un score basé sur la longueur de la correspondance
                        # Plus le type_name est long et spécifique, meilleur est le score
                        match_score = len(type_name_clean) * 3
                        
                        # Bonus pour les termes très spécifiques
                        if "generators" in type_name_clean and "1225" in code_entry['code']:
                            match_score += 50
                        if "fibre" in type_name_clean and "1223" in code_entry['code']:
                            match_score += 50
                        if "electrical" in type_name_clean and "1225" in code_entry['code']:
                            match_score += 30
                        
                        if match_score > best_type_score:
                            best_type_score = match_score
                            best_type_match = code_entry
            
            # Si on a trouvé une correspondance de type_name avec un bon score
            if best_type_match and best_type_score > 20:
                confidence = min(best_type_score / 100, 0.95)  # Max 95% de confiance
                return {
                    'code': best_type_match['code'],
                    'description_fr': best_type_match['description_fr'],
                    'confidence': confidence,
                    'score': best_type_score,
                    'matches': [f"type_match:{type_name_clean}"],
                    'method': 'type_name_exact'
                }
        
        # ÉTAPE 2 : RECHERCHE COMBINÉE AVEC TOUS LES CHAMPS
        # -------------------------------------------------
        search_text_parts = []
        
        # Pondération intelligente :
        # 1. Le type_name a un poids très élevé
        if type_name_clean:
            # Répéter le type_name 5 fois pour lui donner la priorité
            for _ in range(5):
                search_text_parts.append(type_name_clean)
        
        # 2. Le nom français a un poids élevé
        if french_name_clean:
            for _ in range(3):
                search_text_parts.append(french_name_clean)
        
        # 3. Le nom anglais a un poids moyen
        if item_name_clean:
            for _ in range(2):
                search_text_parts.append(item_name_clean)
        
        # 4. La catégorie a un poids faible
        if category_clean:
            search_text_parts.append(category_clean)
        
        if not search_text_parts:
            return {
                'code': 'NON_CLASSIFIE',
                'description_fr': 'Non classifié',
                'confidence': 0,
                'score': 0,
                'matches': [],
                'method': 'no_data'
            }
        
        search_text = ' '.join(search_text_parts)
        
        # Recherche du meilleur match
        best_match = None
        best_score = -1
        best_confidence = 0
        best_result = {'matches': []}
        
        for code_entry in self.search_index:
            result = self.calculate_score(search_text, code_entry)
            
            # Bonus supplémentaire si le type_name correspond à ce code
            if type_name_clean and 'type_names' in code_entry and code_entry['type_names']:
                if isinstance(code_entry['type_names'], list):
                    code_type_names = ' '.join(code_entry['type_names']).lower()
                else:
                    code_type_names = str(code_entry['type_names']).lower()
                if type_name_clean in code_type_names:
                    result['score'] += 30  # Bonus important
                    result['confidence'] = min(result['confidence'] + 0.2, 1.0)
            
            if result['score'] > best_score:
                best_score = result['score']
                best_confidence = result['confidence']
                best_match = code_entry
                best_result = result
        
        # Seuil minimum pour considérer une classification
        if best_score >= 15 and best_confidence >= 0.25:
            return {
                'code': best_match['code'],
                'description_fr': best_match['description_fr'],
                'confidence': best_confidence,
                'score': best_score,
                'matches': best_result['matches'],
                'method': 'text_search'
            }
        else:
            return {
                'code': 'NON_CLASSIFIE',
                'description_fr': 'Non classifié',
                'confidence': 0,
                'score': best_score,
                'matches': [],
                'method': 'low_confidence'
            }
    
    def classify_batch(self, df):
        """Classifie un lot d'items"""
        results = df.copy()
        
        # Initialiser les colonnes
        results['code_comptable'] = ''
        results['description_comptable'] = ''
        results['confiance'] = 0.0
        results['score_classification'] = 0
        
        total = len(df)
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Utiliser enumerate pour un comptage fiable
        for i, (orig_idx, row) in enumerate(df.iterrows(), start=1):
            item_name = str(row.get('name', '')) if pd.notna(row.get('name', '')) else ''
            french_name = str(row.get('french_name', '')) if pd.notna(row.get('french_name', '')) else ''
            category = str(row.get('category_name', '')) if pd.notna(row.get('category_name', '')) else ''
            type_name = str(row.get('type_name', '')) if 'type_name' in row and pd.notna(row.get('type_name', '')) else ''
            
            classification = self.classify_item(item_name, french_name, category, type_name)
            
            results.at[orig_idx, 'code_comptable'] = classification['code']
            results.at[orig_idx, 'description_comptable'] = classification['description_fr']
            results.at[orig_idx, 'confiance'] = classification['confidence']
            results.at[orig_idx, 'score_classification'] = classification['score']
            
            if i % 50 == 0 or i == total:
                progress_bar.progress(min(i / total, 1.0))
                status_text.text(f"Classification: {i}/{total}")
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Classification terminée: {total} items")
        
        return results


# ================================================================
# === CLASSIFICATEUR BASÉ SUR DES RÈGLES MÉTIER
# ================================================================
class RuleBasedClassifier:
    """Classificateur basé sur des règles métier explicites"""
    
    def __init__(self):
        self.rules = [
            # Règle 1: Services de conseil informatique
            {
                'pattern': r'software.*consult|consult.*software|conseil.*logiciel|it.*consultancy',
                'type_names': ['IT Services', 'Consultancy Fees', 'IT Services', 'Software'],
                'category': ['SERVICE'],
                'code': '1262.005',
                'description': 'Conseil et honoraires informatiques',
                'confidence': 0.95,
                'exclude_codes': ['1111.002']  # Exclure les codes de vente
            },
            
            # Règle 2: Maintenance informatique
            {
                'pattern': r'laptop.*maintenance|desktop.*maintenance|printer.*maintenance|réparation.*ordinateur|maintenance.*informatique|computer.*repair',
                'type_names': ['IT Services', 'Maintenance', 'IT consumables Admin'],
                'category': ['SERVICE'],
                'code': '1244.003',
                'description': 'Maintenance équipements informatiques',
                'confidence': 0.95
            },
            
            # Règle 3: Installation caméra/surveillance
            {
                'pattern': r'camera.*installation|installation.*caméra|vidéo.*surveillance|cctv.*installation|surveillance.*system',
                'type_names': ['IT Services', 'Security', 'Remote motoring system'],
                'category': ['SERVICE'],
                'code': '1224.004',
                'description': 'Installation système vidéo surveillance',
                'confidence': 0.95
            },
            
            # Règle 4: Travaux d'étanchéité
            {
                'pattern': r'waterproofing|étanchéité|etancheite|toiture|roof.*sealing|waterproof.*work',
                'type_names': ['Maintenance Admin', 'Building / Chantier', 'Maintenance'],
                'category': ['SERVICE'],
                'code': '1221.006',
                'description': 'Travaux d\'étanchéité',
                'confidence': 0.95
            },
            
            # Règle 5: Services de manutention
            {
                'pattern': r'handling.*services|services.*manutention|manutention|levage|handling.*material',
                'type_names': ['Maintenance', 'Handling material', 'Small Tools & Equipments'],
                'category': ['SERVICE'],
                'code': '1231.003',
                'description': 'Services de manutention',
                'confidence': 0.95
            },
            
            # Règle 6: Formation
            {
                'pattern': r'training|formation|cours|stage|workshop|séminaire',
                'type_names': ['Training Expense Admin', 'TRAINING', 'IT Services'],
                'category': ['SERVICE'],
                'code': '1213.003',
                'description': 'Formation',
                'confidence': 0.95
            },
            
            # Règle 7: Nettoyage
            {
                'pattern': r'cleaning|nettoyage|entretien.*locaux|janitor|ménage',
                'type_names': ['Maintenance Admin', 'Office Maintenance'],
                'category': ['SERVICE'],
                'code': '1271.009',
                'description': 'Prestation, entretien et nettoyage',
                'confidence': 0.95
            },
            
            # Règle 8: Sécurité/gardiennage
            {
                'pattern': r'security|gardiennage|surveillance|guard|sécurité',
                'type_names': ['Security', 'Security Admin', 'SHERQ'],
                'category': ['SERVICE'],
                'code': '1271.005',
                'description': 'Gardiennage, frais de sécurité',
                'confidence': 0.95
            },
            
            # Règle 9: Transport/Voyages
            {
                'pattern': r'travel|voyage|transport|déplacement|mission|déplacement',
                'type_names': ['Transport & Travel', 'Transport & Travel admin'],
                'category': ['SERVICE'],
                'code': '1213.001',
                'description': 'Missions et voyages',
                'confidence': 0.95
            },
            
            # Règle 10: Restauration
            {
                'pattern': r'catering|cafétéria|cantine|restaurant|repas|meal|cafeteria',
                'type_names': ['Cafeteria', 'Other expenses Admin'],
                'category': ['SERVICE'],
                'code': '1213.006',
                'description': 'Restauration',
                'confidence': 0.95
            },
            
            # Règle 11: Carburant
            {
                'pattern': r'fuel|carburant|gasoil|diesel|essence|gazole|petrol',
                'type_names': ['Fuel', 'Fuel Admin', 'Customer Site Fuel', 'OFFICE FUEL SUPPLY'],
                'category': ['SERVICE', 'CONSOMMABLE'],
                'code': '1243.001',
                'description': 'Essence / Diesel pour VL',
                'confidence': 0.95
            },
            
            # Règle 12: Pièces détachées véhicules
            {
                'pattern': r'spare parts.*vehicle|pièces.*véhicule|motor.*spares|auto.*parts|car.*spares',
                'type_names': ['Motor Vehicles spares', 'Motor Vehicales spares Admin', 'Motor Vehicle Expneses'],
                'category': ['SERVICE', 'CONSOMMABLE'],
                'code': '1244.001',
                'description': 'Entretiens, réparations et PDR pour VL',
                'confidence': 0.95
            },
            
            # Règle 13: Location véhicules
            {
                'pattern': r'vehicle.*rental|location.*voiture|car.*rental|4x4.*rental|rent.*car',
                'type_names': ['VEHICLE', 'Transport & Travel'],
                'category': ['SERVICE'],
                'code': '1241.001',
                'description': 'Location externe voiture, pickup & suv',
                'confidence': 0.95
            },
            
            # Règle 14: Télécommunications
            {
                'pattern': r'telecom|télécom|internet|téléphone|phone|communication|mobile',
                'type_names': ['Communication Admin', 'IT Services'],
                'category': ['SERVICE'],
                'code': '1273.001',
                'description': 'Frais de télécommunication',
                'confidence': 0.95
            },
            
            # Règle 15: Fournitures de bureau
            {
                'pattern': r'office.*supplies|fournitures.*bureau|paper|stylo|cartouche|toner|encre',
                'type_names': ['IT consumables Admin', 'Consumables for Admin', 'OFFICE FURNITURES'],
                'category': ['CONSOMMABLE'],
                'code': '1275.001',
                'description': 'Fournitures de bureau et consommables informatiques',
                'confidence': 0.95
            },
            
            # Règle 16: Groupes électrogènes (GEN)
            {
                'pattern': r'gen|generator|groupe.*électrogène|genset|power.*generator',
                'type_names': ['Generators-Equipment', 'Generators-Spare parts', 'Generators-Consumables'],
                'category': ['STOCKABLE', 'EQUIPMENT'],
                'code': '1225.001',
                'description': 'Groupes électrogènes',
                'confidence': 0.95
            },
            
            # Règle 17: Câbles électriques
            {
                'pattern': r'cable.*électrique|electrical.*cable|power.*cable|fil.*électrique',
                'type_names': ['Electrical-Consumables', 'Electrical-Spare parts', 'Power cables'],
                'category': ['STOCKABLE'],
                'code': '1225.007',
                'description': 'Câbles électriques',
                'confidence': 0.95
            },
            
            # Règle 18: Fibre optique
            {
                'pattern': r'fibre.*optique|fiber.*optic|fo|fusion.*splicer|câble.*fo',
                'type_names': ['Fibre Optic-Equipment', 'Fibre Optic-Consumables', 'Fibre Optic-Spare parts'],
                'category': ['STOCKABLE'],
                'code': '1223.005',
                'description': 'Câble FO',
                'confidence': 0.95
            },
            
            # Règle 19: Peinture
            {
                'pattern': r'peinture|paint|weather.*guard|forest.*green',
                'type_names': [],
                'category': ['STOCKABLE'],
                'code': '1228.002',
                'description': 'Peinture',
                'confidence': 0.95
            },
            
            # Règle 20: WD40 / Dégraissant
            {
                'pattern': r'wd40|dégraissant|degreasant|lubrifiant|graisse|huile',
                'type_names': ['Generators-Consumables', 'OIL FILTER', 'FUEL FILTER'],
                'category': ['STOCKABLE'],
                'code': '1227.004',
                'description': 'Huiles & graisses pour maintenance',
                'confidence': 0.95
            },
        ]

    
    def apply_rules(self, item_name, french_name, category, type_name):
        """Applique les règles de classification"""
        import re
        
        # Combiner tous les textes pour la recherche
        item_text = str(item_name).lower() if item_name else ""
        french_text = str(french_name).lower() if french_name else ""
        category_text = str(category).lower() if category else ""
        type_text = str(type_name).lower() if type_name else ""
        
        full_text = f"{item_text} {french_text} {category_text} {type_text}"
        
        best_match = None
        best_score = 0
        
        for rule in self.rules:
            score = 0
            
            # Vérifier le pattern regex
            if re.search(rule['pattern'], full_text, re.IGNORECASE):
                score += 50  # Pattern trouvé = score élevé
                
                # Vérifier le type_name si spécifié
                if rule['type_names'] and type_text:
                    for tn in rule['type_names']:
                        if tn.lower() in type_text:
                            score += 30
                            break
                
                # Vérifier la catégorie si spécifiée
                if rule['category'] and category_text:
                    for cat in rule['category']:
                        if cat.lower() in category_text:
                            score += 20
                            break
                
                # Bonus pour les mots exacts dans le nom
                for word in rule['pattern'].replace('.*', ' ').replace('|', ' ').split():
                    if len(word) > 3 and word in full_text:
                        score += 10
                
                if score > best_score:
                    best_score = score
                    best_match = rule
        
        if best_match and best_score > 30:  # Seuil minimum
            confidence = min(best_score / 100, 0.95)
            return {
                'code': best_match['code'],
                'description_fr': best_match['description'],
                'confidence': confidence,
                'score': best_score,
                'method': 'rule_based',
                'rule_pattern': best_match['pattern']
            }
        
        return None    
# ================================================================
# === MAPPING TYPE_NAME VERS CODES COMPTABLES
# ================================================================
TYPE_NAME_MAPPING = {
    # IT et Informatique
    'IT Services': {
        'primary': '1244.003',  # Maintenance par défaut
        'description': 'Maintenance équipements informatiques',
        'keywords': {
            'software|consultancy|conseil': ('1262.005', 'Conseil et honoraires informatiques'),
            'maintenance|repair|réparation': ('1244.003', 'Maintenance équipements informatiques'),
            'installation|setup': ('1273.002', 'Coûts informatiques - matériel'),
            'development|développement': ('1262.005', 'Conseil et honoraires informatiques'),
        }
    },
    'IT Software': {
        'primary': '1273.003',
        'description': 'Coûts informatiques - logiciels et licences',
        'keywords': {
            'license|licence': ('1273.003', 'Coûts informatiques - logiciels et licences'),
            'development|développement': ('1262.005', 'Conseil et honoraires informatiques'),
            'consultancy|conseil': ('1262.005', 'Conseil et honoraires informatiques'),
            'cloud|saas': ('1273.003', 'Coûts informatiques - logiciels et licences'),
        }
    },
    'IT Hardware': {
        'primary': '1273.002',
        'description': 'Coûts informatiques - matériel',
        'keywords': {
            'computer|ordinateur': ('1273.002', 'Coûts informatiques - matériel'),
            'server|serveur': ('1273.002', 'Coûts informatiques - matériel'),
            'peripheral|périphérique': ('1273.002', 'Coûts informatiques - matériel'),
            'laptop|desktop': ('1273.002', 'Coûts informatiques - matériel'),
        }
    },
    'IT consumables Admin': {
        'primary': '1275.001',
        'description': 'Fournitures de bureau et consommables informatiques',
        'keywords': {
            'toner|cartouche': ('1275.001', 'Fournitures de bureau'),
            'paper|papier': ('1275.001', 'Fournitures de bureau'),
            'usb|cable|hdmi': ('1273.002', 'Coûts informatiques - matériel'),
        }
    },
    
    # Maintenance
    'Maintenance': {
        'primary': '1227.001',
        'description': 'PDR pour entretien courant',
        'keywords': {
            'building|bâtiment': ('1221.006', 'Travaux d\'étanchéité'),
            'equipment|équipement': ('1227.001', 'PDR pour entretien courant'),
            'preventive|préventive': ('1227.002', 'PDR pour maintenance planifiée'),
            'handling|manutention': ('1231.003', 'Services de manutention'),
        }
    },
    'Maintenance Admin': {
        'primary': '1227.001',
        'description': 'PDR pour entretien courant',
        'keywords': {
            'cleaning|nettoyage': ('1271.009', 'Prestation, entretien et nettoyage'),
            'waterproof|étanchéité': ('1221.006', 'Travaux d\'étanchéité'),
            'repair|réparation': ('1227.001', 'PDR pour entretien courant'),
        }
    },
    
    # Sécurité
    'Security': {
        'primary': '1271.005',
        'description': 'Gardiennage, frais de sécurité',
        'keywords': {
            'guard|gardiennage': ('1271.005', 'Gardiennage'),
            'camera|cctv': ('1224.004', 'Installation système vidéo surveillance'),
            'alarm|alarme': ('1224.001', 'Système protection anti-incendie'),
        }
    },
    'Security Admin': {
        'primary': '1271.005',
        'description': 'Gardiennage, frais de sécurité',
        'keywords': {
            'guard|gardiennage': ('1271.005', 'Gardiennage'),
            'surveillance': ('1224.004', 'Installation système vidéo surveillance'),
        }
    },
    'SHERQ': {
        'primary': '1271.005',
        'description': 'Gardiennage, frais de sécurité',
        'keywords': {
            'safety|sécurité': ('1271.005', 'Gardiennage'),
            'protection': ('1233.003', 'Équipement de Protection Individuel'),
            'health|santé': ('1213.004', 'Assurance santé'),
        }
    },
    
    # Formation et voyages
    'Training Expense Admin': {
        'primary': '1213.003',
        'description': 'Formation',
        'keywords': {}
    },
    'TRAINING': {
        'primary': '1213.003',
        'description': 'Formation',
        'keywords': {}
    },
    'Transport & Travel': {
        'primary': '1213.001',
        'description': 'Missions et voyages',
        'keywords': {
            'flight|avion': ('1213.001', 'Missions et voyages'),
            'hotel': ('1213.001', 'Missions et voyages'),
            'taxi': ('1213.001', 'Missions et voyages'),
            'train': ('1213.001', 'Missions et voyages'),
        }
    },
    'Transport & Travel admin': {
        'primary': '1213.001',
        'description': 'Missions et voyages',
        'keywords': {}
    },
    
    # Véhicules
    'Motor Vehicles spares': {
        'primary': '1244.001',
        'description': 'Entretiens, réparations et PDR pour VL',
        'keywords': {}
    },
    'Motor Vehicales spares Admin': {
        'primary': '1244.001',
        'description': 'Entretiens, réparations et PDR pour VL',
        'keywords': {}
    },
    'Motor Vehicle Expneses': {
        'primary': '1244.001',
        'description': 'Entretiens, réparations et PDR pour VL',
        'keywords': {
            'fuel|carburant': ('1243.001', 'Essence / Diesel pour VL'),
            'repair|réparation': ('1244.001', 'Entretiens et réparations'),
        }
    },
    'VEHICLE': {
        'primary': '1241.001',
        'description': 'Location externe voiture, pickup & suv',
        'keywords': {
            'rental|location': ('1241.001', 'Location externe'),
            'lease|leasing': ('1241.001', 'Location externe'),
        }
    },
    
    # Carburant
    'Fuel': {
        'primary': '1243.002',
        'description': 'Essence / Diesel pour autre usage interne',
        'keywords': {
            'vehicle|véhicule': ('1243.001', 'Essence / Diesel pour VL'),
            'generator|groupe': ('1229.001', 'Carburant pour refueling des sites clients'),
            'office|bureau': ('1243.002', 'Essence / Diesel pour autre usage'),
        }
    },
    'Fuel Admin': {
        'primary': '1243.002',
        'description': 'Essence / Diesel pour autre usage interne',
        'keywords': {}
    },
    'Customer Site Fuel': {
        'primary': '1229.001',
        'description': 'Carburant pour refueling des sites clients',
        'keywords': {}
    },
    'OFFICE FUEL SUPPLY': {
        'primary': '1243.002',
        'description': 'Essence / Diesel pour autre usage interne',
        'keywords': {}
    },
    
    # Services divers
    'Consultancy Fees': {
        'primary': '1272.005',
        'description': 'Autres conseils et honoraires',
        'keywords': {
            'it|informatique': ('1262.005', 'Conseil et honoraires informatiques'),
            'tax|fiscal': ('1272.003', 'Conseil fiscal'),
            'legal|juridique': ('1272.001', 'Frais d\'actes et contentieux'),
        }
    },
    'Consultancy Fees Admin': {
        'primary': '1272.005',
        'description': 'Autres conseils et honoraires',
        'keywords': {}
    },
    'Consultancy services': {
        'primary': '1272.005',
        'description': 'Autres conseils et honoraires',
        'keywords': {}
    },
    
    # Communications
    'Communication Admin': {
        'primary': '1273.001',
        'description': 'Frais de télécommunication',
        'keywords': {}
    },
    
    # Restauration
    'Cafeteria': {
        'primary': '1213.006',
        'description': 'Restauration',
        'keywords': {}
    },
    
    # Travaux et construction
    'Building / Chantier': {
        'primary': '1221.012',
        'description': 'Divers matériaux GC incorporés',
        'keywords': {
            'waterproof|étanchéité': ('1221.006', 'Travaux d\'étanchéité'),
            'concrete|béton': ('1221.001', 'Béton prêt à l\'emploi'),
            'electrical|électrique': ('1225.010', 'Divers fournitures électriques'),
        }
    },
    'Handling material': {
        'primary': '1231.003',
        'description': 'Accessoires et outils de levage',
        'keywords': {}
    },
    
    # Équipements divers
    'Remote motoring system': {
        'primary': '1224.004',
        'description': 'Système vidéo surveillance',
        'keywords': {}
    },
    'WORK PROTECTIVE EQUIPMENT': {
        'primary': '1233.003',
        'description': 'Équipement de Protection Individuel (EPI)',
        'keywords': {}
    },
    'Safety equipment': {
        'primary': '1233.003',
        'description': 'Équipement de Protection Individuel (EPI)',
        'keywords': {}
    },
    
    # Location
    'Rent': {
        'primary': '1271.001',
        'description': 'Loyers & charges locatives',
        'keywords': {}
    },
    'Rent Admin': {
        'primary': '1271.001',
        'description': 'Loyers & charges locatives',
        'keywords': {}
    },
    
    # Assurances
    'Insurance': {
        'primary': '1274.004',
        'description': 'Autres assurances',
        'keywords': {
            'vehicle|véhicule': ('1274.001', 'Assurances véhicules légers'),
            'car|auto': ('1274.001', 'Assurances véhicules légers'),
            'site|chantier': ('1274.002', 'Assurance tous risques chantier'),
            'machine': ('1274.005', 'Assurance bris de machine'),
        }
    },
    'Insurance Admin': {
        'primary': '1274.004',
        'description': 'Autres assurances',
        'keywords': {}
    },
    
    # Équipements électriques
    'Electrical-Equipment': {
        'primary': '1225.009',
        'description': 'Compteurs et armoires électriques',
        'keywords': {}
    },
    'Electrical-Consumables': {
        'primary': '1225.010',
        'description': 'Divers fournitures électriques',
        'keywords': {}
    },
    'Electrical-Spare parts': {
        'primary': '1225.010',
        'description': 'Divers fournitures électriques',
        'keywords': {}
    },
    
    # Générateurs
    'Generators-Equipment': {
        'primary': '1225.001',
        'description': 'Groupes électrogènes',
        'keywords': {}
    },
    'Generators-Spare parts': {
        'primary': '1225.001',
        'description': 'Groupes électrogènes',
        'keywords': {}
    },
    'Generators-Consumables': {
        'primary': '1225.001',
        'description': 'Groupes électrogènes',
        'keywords': {}
    },
    
    # Fibre optique
    'Fibre Optic-Equipment': {
        'primary': '1223.007',
        'description': 'Panneaux et armoires FO',
        'keywords': {}
    },
    'Fibre Optic-Consumables': {
        'primary': '1223.004',
        'description': 'Connecteurs FO',
        'keywords': {}
    },
    'Fibre Optic-Spare parts': {
        'primary': '1223.004',
        'description': 'Connecteurs FO',
        'keywords': {}
    },
    
    # Sous-traitance
    'Subcontracting': {
        'primary': '1285.001',
        'description': 'Sous-traitants autres',
        'keywords': {
            'civil|gc|génie': ('1281.001', 'Sous-traitants génie civil'),
            'electrical|électricité': ('1282.001', 'Sous-traitants électricité'),
            'fibre|fo|optique': ('1283.001', 'Sous-traitants fibre optique'),
            'telecom|télécom': ('1284.001', 'Sous-traitants télécoms'),
        }
    },
    'Subcontractor': {
        'primary': '1285.001',
        'description': 'Sous-traitants autres',
        'keywords': {}
    },
}
# ================================================================
# === FONCTION DE CHARGEMENT DU FICHIER
# ================================================================
def load_data_from_file(uploaded_file):
    """Charge les données depuis un fichier CSV ou Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Utilisez CSV ou Excel.")
            return None
        
        # Nettoyer les noms de colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Afficher les colonnes disponibles
        st.info(f"Colonnes disponibles: {', '.join(df.columns)}")
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

# ================================================================
# === INTERFACE PRINCIPALE
# ================================================================
def main():
    # Charger les codes comptables une seule fois
    if 'codes_df' not in st.session_state:
        st.session_state.codes_df = load_accounting_codes()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Chargement des données")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV ou Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Sélectionnez le fichier exporté depuis le Data Warehouse"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Charger le fichier", use_container_width=True):
                with st.spinner("Chargement en cours..."):
                    df = load_data_from_file(uploaded_file)
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.success(f"✅ {len(df)} items chargés")
                        st.rerun()
        
        if st.button("🔄 Réinitialiser", use_container_width=True):
            if 'df' in st.session_state:
                del st.session_state.df
            if 'classified_df' in st.session_state:
                del st.session_state.classified_df
            st.rerun()
        
        if 'df' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Statistiques")
            st.metric("Items chargés", len(st.session_state.df))
            st.metric("Codes comptables", len(st.session_state.codes_df))
    
    # Page principale
    st.markdown("## 🎯 Classification Comptable Automatique")
    
    # Aperçu des codes comptables - VERSION CORRIGÉE
    with st.expander("📋 Voir les 225 codes comptables enrichis"):
        # Créer une copie pour l'affichage sans modifier l'original
        display_df = st.session_state.codes_df[['code', 'description_fr']].copy()
        display_df.columns = ['Code', 'Description']
        
        # Ajouter les mots-clés sous forme de texte simple (pas de listes)
        if 'keywords_fr' in st.session_state.codes_df.columns:
            # Convertir les NaN en chaînes vides et s'assurer que c'est du texte
            keywords_fr = st.session_state.codes_df['keywords_fr'].fillna('')
            display_df['Mots-clés'] = keywords_fr.astype(str)
        
        st.dataframe(display_df, use_container_width=True, height=300)
        
        # Afficher quelques statistiques sur l'enrichissement
        nb_enriched = len(st.session_state.codes_df[st.session_state.codes_df['keywords_fr'] != ''])
        st.info(f"📊 {nb_enriched} codes comptables enrichis avec des mots-clés spécifiques")
    
    # Vérifier si des données sont chargées
    if 'df' not in st.session_state:
        st.info("👈 Chargez un fichier CSV ou Excel pour commencer")
        
        # Exemple de données avec les items de votre fichier
        st.markdown("### 📝 Exemple de données (vos items)")
        example_df = pd.DataFrame({
            'name': [
                'Yellow EC-J labels from A to Z; 500/roll',
                'Yellow 150mm2 & Green Earth Cable',
                'X-910 Fiber Fusion Splicer',
                'Weather Guard Forest Green Paint',
                'WD40 DEGRAISSANT 500ml',
                'waterproof box 250 x200x180',
                'Water Pump 1307010-X03'
            ],
            'french_name': [
                'Etiquette EC-J jaune A à Z ; 500/ROLL',
                'Câble de terre jaune et vert 150 mm2',
                'X-910 Fiber Fusion Splicer',
                'Peinture verte forêt Weather Guard',
                'DEGRAISSANT WD40 500ml',
                'boite étanche 250 x200x180',
                'Water Pump 1307010-X03'
            ],
            'category_name': ['STOCKABLE'] * 7,
        })
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Exemples de classifications attendues :**
        - **Câble terre jaune/vert** → 1225.008 (Paratonnerre et mise à la terre)
        - **X-910 Fiber Fusion Splicer** → 1223.004/1223.005 (Équipements FO)
        - **Peinture Weather Guard** → 1228.002 (Peinture)
        - **WD40** → 1227.004 (Huiles & graisses)
        - **Water Pump** → 1224.005 (Pompes industrielles)
        """)
        return
    
    # Données chargées - proposer la classification
    df = st.session_state.df
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Items à classifier", len(df))
    with col2:
        st.metric("Codes disponibles", len(st.session_state.codes_df))
    with col3:
        if st.button("🚀 Lancer la classification", type="primary", use_container_width=True):
            classifier = AccountingClassifier(st.session_state.codes_df)
            with st.spinner("Classification en cours..."):
                classified_df = classifier.classify_batch(df)
                st.session_state.classified_df = classified_df
                st.success("✅ Classification terminée !")
                st.rerun()
    
    # Afficher les résultats si disponibles
    if 'classified_df' in st.session_state:
        classified_df = st.session_state.classified_df
        
        # Statistiques
        st.markdown("---")
        st.markdown("### 📊 Résultats de la classification")
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            classified = len(classified_df[classified_df['code_comptable'] != 'NON_CLASSIFIE'])
            pct = (classified / len(classified_df)) * 100 if len(classified_df) > 0 else 0
            st.metric("Classifiés", f"{classified}", delta=f"{pct:.1f}%")
        with col_stats2:
            non_classified = len(classified_df[classified_df['code_comptable'] == 'NON_CLASSIFIE'])
            st.metric("Non classifiés", non_classified)
        with col_stats3:
            high_confidence = len(classified_df[classified_df['confiance'] >= 0.7])
            st.metric("Haute confiance", high_confidence)
        with col_stats4:
            avg_confidence = classified_df['confiance'].mean() if len(classified_df) > 0 else 0
            st.metric("Confiance moyenne", f"{avg_confidence:.1%}")
        
        # Filtres
        st.markdown("### 🔍 Filtrer les résultats")

        # Créer une ligne avec 4 colonnes pour plus de filtres
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns([2, 1, 1, 2])

        with col_filter1:
            search = st.text_input("🔎 Rechercher", placeholder="Nom de l'item...", 
                                  help="Recherche dans le nom et le nom français")

        with col_filter2:
            conf_min = st.slider("Confiance minimum", 0.0, 1.0, 0.0, 0.05,
                                help="Filtre par niveau de confiance")

        with col_filter3:
            show_only = st.selectbox(
                "Afficher",
                ["Tous", "Classifiés uniquement", "Non classifiés", "Haute confiance (≥70%)"],
                help="Filtre par état de classification"
            )

        with col_filter4:
            # Récupérer tous les type_name uniques présents dans les données
            if 'type_name' in classified_df.columns:
                type_names_list = ['Tous'] + sorted(classified_df['type_name'].dropna().unique().tolist())
                selected_type = st.selectbox(
                    "📦 Type d'équipement",
                    type_names_list,
                    help="Filtrer par type d'équipement (type_name)"
                )
            else:
                selected_type = "Tous"
                st.info("Colonne 'type_name' non trouvée")

        # Appliquer les filtres
        filtered_df = classified_df.copy()

        # Filtre par recherche textuelle
        if search:
            filtered_df = filtered_df[
                filtered_df['name'].astype(str).str.contains(search, case=False, na=False) |
                filtered_df['french_name'].astype(str).str.contains(search, case=False, na=False)
            ]

        # Filtre par confiance minimum
        filtered_df = filtered_df[filtered_df['confiance'] >= conf_min]

        # Filtre par état de classification
        if show_only == "Classifiés uniquement":
            filtered_df = filtered_df[filtered_df['code_comptable'] != 'NON_CLASSIFIE']
        elif show_only == "Non classifiés":
            filtered_df = filtered_df[filtered_df['code_comptable'] == 'NON_CLASSIFIE']
        elif show_only == "Haute confiance (≥70%)":
            filtered_df = filtered_df[filtered_df['confiance'] >= 0.7]

        # Filtre par type_name
        if selected_type != "Tous" and 'type_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['type_name'] == selected_type]

        st.markdown(f"**{len(filtered_df)} items** correspondant aux critères")

        # Afficher le tableau avec mise en forme
        display_cols = ['name', 'french_name', 'category_name', 'type_name', 'code_comptable', 'description_comptable', 'confiance']
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        # Fonction pour colorer la colonne confiance
        def color_confidence_col(s: pd.Series):
            return [
                'background-color: #d4edda; color: #155724' if v >= 0.8 else
                'background-color: #fff3cd; color: #856404' if v >= 0.6 else
                'background-color: #ffe5b4; color: #856404' if v >= 0.4 else
                'background-color: #f8d7da; color: #721c24'
                for v in s
            ]

        # Fonction pour colorer les codes comptables
        def color_code_col(s: pd.Series):
            return [
                'background-color: #cce5ff; color: #004085; font-weight: bold' if v != 'NON_CLASSIFIE' else
                'background-color: #f8d7da; color: #721c24; font-style: italic'
                for v in s
            ]

        # Limiter l'affichage pour éviter les problèmes de performance
        display_df = filtered_df[display_cols].head(1000).copy()

        # Appliquer les styles
        styled_df = (display_df.style
                     .format({'confiance': '{:.1%}'})
                     .apply(color_confidence_col, subset=['confiance'])
                     .apply(color_code_col, subset=['code_comptable']))

        # Renommer les colonnes pour un meilleur affichage
        column_names = {
            'name': 'Nom',
            'french_name': 'Nom français',
            'category_name': 'Catégorie',
            'type_name': 'Type équipement',
            'code_comptable': 'Code comptable',
            'description_comptable': 'Description',
            'confiance': 'Confiance'
        }
        styled_df = styled_df.format_index(axis=1, formatter=lambda x: column_names.get(x, x))

        st.dataframe(styled_df, use_container_width=True, height=600)

        # Statistiques supplémentaires
        if 'type_name' in filtered_df.columns and len(filtered_df) > 0:
            with st.expander("📊 Statistiques par type d'équipement"):
                type_stats = filtered_df.groupby('type_name').agg({
                    'name': 'count',
                    'confiance': 'mean',
                    'code_comptable': lambda x: (x != 'NON_CLASSIFIE').sum()
                }).rename(columns={
                    'name': 'Nombre items',
                    'confiance': 'Confiance moyenne',
                    'code_comptable': 'Items classifiés'
                })
                type_stats['% classifié'] = (type_stats['Items classifiés'] / type_stats['Nombre items'] * 100).round(1)
                type_stats['Confiance moyenne'] = type_stats['Confiance moyenne'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(type_stats, use_container_width=True)
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Export
        st.markdown("### 📥 Export des résultats")
        
        # Synthèse par code comptable
        summary = filtered_df[filtered_df['code_comptable'] != 'NON_CLASSIFIE'].groupby(
            ['code_comptable', 'description_comptable'], dropna=False
        ).agg({
            'name': 'count',
            'confiance': 'mean'
        }).rename(columns={'name': 'nb_items', 'confiance': 'confiance_moyenne'})
        
        col_export1, col_export2, col_export3 = st.columns(3)
        with col_export1:
            csv = filtered_df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                "📥 CSV (résultats filtrés)",
                data=csv,
                file_name=f"classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_export2:
            if not summary.empty:
                csv_summary = summary.reset_index().to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "📊 Synthèse par code",
                    data=csv_summary,
                    file_name=f"synthese_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with col_export3:
            to_review_cols = [c for c in ['name', 'french_name', 'category_name', 'type_name'] if c in filtered_df.columns]
            to_review = filtered_df[filtered_df['code_comptable'] == 'NON_CLASSIFIE'][to_review_cols]
            if not to_review.empty:
                csv_review = to_review.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "🔍 Items à revoir",
                    data=csv_review,
                    file_name=f"a_revoir_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Export Excel complet
        st.markdown("### 📊 Export Excel complet")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Classification', index=False)
            if not summary.empty:
                summary.reset_index().to_excel(writer, sheet_name='Synthese', index=False)
        output.seek(0)
        
        st.download_button(
            "📥 Télécharger Excel complet",
            data=output,
            file_name=f"classification_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if __name__ == "__main__":
    main()