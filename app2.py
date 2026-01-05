# ================================================================
# Détection Intelligente de Doublons - Optimisé pour vos colonnes
# Auteur : Zineb FAKKAR – Janv 2026
# Colonnes supportées : id, reference, name, french_name, uom_name, etc.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime
import hashlib
from functools import lru_cache
import time
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')
import json

# -------- Configuration Streamlit --------
st.set_page_config(
    page_title="Détection Intelligente de Doublons", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Noms de colonnes exacts --------
COLUMNS_MAP = {
    # Vos colonnes exactes
    'id': 'id',
    'reference': 'reference',
    'name': 'name',  # Anciennement item_name
    'french_name': 'french_name',
    'status': 'status',
    'uom_name': 'uom_name',
    'type_name': 'type_name',
    'sub_category_name': 'sub_category_name',
    'category_name': 'category_name',
    'last_use': 'last_use',
    'last_price': 'last_price',
    'requestor_name': 'requestor_name',
    'company_name': 'company_name',
    'department_name': 'department_name',
    'created_at': 'created_at',
    'updated_at': 'updated_at'
}

EXPECTED_COLUMNS = list(COLUMNS_MAP.keys())

# -------- Fonction pour détecter automatiquement les colonnes --------
def detect_columns_automatically(df: pd.DataFrame) -> Dict[str, str]:
    """
    Tente de détecter automatiquement quelle colonne correspond à quel champ
    en analysant le contenu des colonnes.
    """
    column_mapping = {}
    
    # Convertir toutes les colonnes en string pour l'analyse
    df_str = df.astype(str)
    
    # Règles de détection
    detection_rules = {
        'id': {
            'keywords': ['id', 'identifiant', 'numéro', 'code'],
            'type_check': lambda x: x.str.isnumeric().any() or x.str.match(r'^[A-Za-z0-9\-_]+$').any()
        },
        'reference': {
            'keywords': ['référence', 'ref', 'code', 'sku', 'article'],
            'type_check': lambda x: x.str.match(r'^[A-Za-z0-9\-_/.]+$').any()
        },
        'name': {
            'keywords': ['nom', 'name', 'désignation', 'description', 'item', 'produit'],
            'type_check': lambda x: x.str.len().mean() > 5 and x.str.contains(r'[a-zA-Z]').any()
        },
        'french_name': {
            'keywords': ['français', 'french', 'nom français', 'désignation fr'],
            'type_check': lambda x: x.str.contains(r'[éèêàâùûîïôœ]', regex=True).any()
        },
        'category_name': {
            'keywords': ['catégorie', 'category', 'famille', 'groupe'],
            'type_check': lambda x: True
        },
        'type_name': {
            'keywords': ['type', 'typologie', 'modèle', 'variante'],
            'type_check': lambda x: True
        },
        'company_name': {
            'keywords': ['entreprise', 'company', 'fournisseur', 'vendor', 'société'],
            'type_check': lambda x: True
        }
    }
    
    # Analyser chaque colonne
    for col in df.columns:
        col_lower = str(col).lower()
        col_data = df_str[col]
        
        # Chercher la correspondance la plus probable
        best_match = None
        best_score = 0
        
        for field, rules in detection_rules.items():
            score = 0
            
            # Score basé sur le nom de la colonne
            for keyword in rules['keywords']:
                if keyword in col_lower:
                    score += 3
            
            # Score basé sur le contenu
            try:
                if rules['type_check'](col_data):
                    score += 2
            except:
                pass
            
            # Vérifier les patterns spécifiques
            if field == 'id' and col_lower == 'id':
                score += 10
            elif field == 'name' and ('name' in col_lower or 'nom' in col_lower):
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = field
        
        # Si on a trouvé une correspondance raisonnable
        if best_match and best_score >= 2:
            column_mapping[best_match] = col
    
    return column_mapping

# -------- Fonction pour mapper les colonnes --------
def map_columns_interactively(df: pd.DataFrame) -> Dict[str, str]:
    """
    Permet à l'utilisateur de mapper interactivement les colonnes
    """
    st.markdown("### 🗺️ Mapping des colonnes")
    st.info("Votre fichier a des colonnes génériques (Colonne1, Colonne2...). Aidez-nous à les comprendre.")
    
    column_mapping = {}
    
    # Afficher un aperçu des données
    with st.expander("📋 Aperçu des données"):
        st.dataframe(df.head(), use_container_width=True)
    
    # Pour chaque champ important, laisser l'utilisateur choisir
    important_fields = ['name', 'reference', 'category_name', 'type_name', 'id']
    
    for field in important_fields:
        st.markdown(f"**{field.upper()}** - Sélectionnez la colonne correspondante :")
        
        # Suggestions automatiques
        suggestions = []
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Vérifier si le nom de colonne contient des indices
            if field == 'name' and ('nom' in col_lower or 'name' in col_lower or 'désignation' in col_lower):
                suggestions.insert(0, col)
            elif field == 'reference' and ('ref' in col_lower or 'référence' in col_lower or 'code' in col_lower):
                suggestions.insert(0, col)
            elif field == 'id' and col_lower == 'id':
                suggestions.insert(0, col)
            else:
                suggestions.append(col)
        
        # Sélectionner la colonne
        selected_col = st.selectbox(
            f"Colonne pour '{field}' :",
            options=suggestions,
            key=f"map_{field}"
        )
        
        if selected_col:
            column_mapping[field] = selected_col
    
    # Bouton pour mapper automatiquement les autres champs
    if st.button("🔄 Compléter automatiquement les autres champs"):
        auto_mapping = detect_columns_automatically(df)
        for field, col in auto_mapping.items():
            if field not in column_mapping:
                column_mapping[field] = col
        st.success("Mapping complété automatiquement !")
    
    return column_mapping

# -------- Base de Connaissance pour vos domaines spécifiques --------
class DomainKnowledgeBase:
    """Base de connaissance spécialisée pour vos domaines (fibre optique, générateurs, etc.)"""
    
    # Groupes sémantiques spécifiques à vos activités
    DOMAIN_GROUPS = {
        # Fibre optique & Télécom
        'fibre_optique': {
            'fibre', 'fibre optique', 'câble fibre', 'câble optique', 'ftth',
            'connecteur fibre', 'épissure', 'splice', 'pon', 'olt', 'onu',
            'souple', 'rigide', 'monomode', 'multimode', 'sc/apc', 'sc/upc',
            'lc', 'fc', 'st', 'mpo', 'cassette', 'panneau', 'pigtail',
            'adaptateur', 'attenuateur', 'coupleur', 'diviseur'
        },
        
        # Générateurs & Énergie
        'generateurs': {
            'générateur', 'groupe électrogène', 'alternateur', 'génératrice',
            'diesel', 'essence', 'gaz', 'silencieux', 'insonorisé',
            'portable', 'stationnaire', 'onduleur', 'ups', 'batterie',
            'chargeur', 'transformateur', 'redresseur', 'convertisseur',
            'kva', 'kw', 'kva', 'puissance', 'moteur', 'démarreur'
        },
        
        # Câbles & Câblage
        'cables': {
            'câble', 'cordon', 'fil', 'câblage', 'électrique', 'alimentation',
            'rj45', 'cat5', 'cat6', 'cat7', 'coaxial', 'hdmi', 'usb',
            'vga', 'dvi', 'displayport', 'audio', 'vidéo', 'données',
            'paire torsadée', 'blindé', 'non blindé', 'utp', 'stp', 'ftp'
        },
        
        # Connecteurs & Prise
        'connecteurs': {
            'connecteur', 'prise', 'fiche', 'jack', 'port', 'terminal',
            'borne', 'barrette', 'domino', 'wago', 'dīn', 'rail',
            'rallonge', 'multiprise', 'parafoudre', 'interrupteur',
            'disjoncteur', 'fusible', 'sectionneur', 'contacteur'
        },
        
        # Outillage & Matériel
        'outillage': {
            'outil', 'pince', 'tournevis', 'perceuse', 'visseuse', 'scie',
            'marteau', 'clé', 'niveau', 'mètre', 'ruban', 'multimètre',
            'testeur', 'analyseur', 'oscilloscope', 'générateur de signaux',
            'soudure', 'fer à souder', 'défonceuse', 'ponceuse', 'meuleuse'
        },
        
        # Sécurité & Surveillance
        'securite': {
            'caméra', 'cctv', 'ip', 'analogique', 'dôme', 'bullet',
            'ptz', 'infrarouge', 'ir', 'détecteur', 'mouvement', 'fumée',
            'alarme', 'centrale', 'déclencheur', 'sirène', 'badge',
            'contrôle d\'accès', 'biométrie', 'digicode', 'interphone'
        },
        
        # Réseau & Informatique
        'reseau': {
            'switch', 'commutateur', 'routeur', 'firewall', 'pare-feu',
            'point d\'accès', 'ap', 'wifi', 'antenne', 'répéteur',
            'modem', 'media converter', 'serveur', 'nas', 'san',
            'rack', 'baie', 'unité', 'blade', 'modulaire'
        },
        
        # Climatisation & Ventilation
        'climatisation': {
            'climatiseur', 'split', 'mobile', 'fixe', 'inverter',
            'ventilateur', 'extracteur', 'ventilation', 'air conditionné',
            'pompe à chaleur', 'chauffage', 'radiateur', 'convecteur'
        },
        
        # Éclairage
        'eclairage': {
            'ampoule', 'led', 'fluorescente', 'halogène', 'néon',
            'projecteur', 'spot', 'plafonnier', 'suspension', 'applique',
            'baladeuse', 'torche', 'lampe', 'tube', 'culot', 'douille'
        }
    }
    
    # Mots-clés pour catégorisation automatique
    CATEGORY_KEYWORDS = {
        'fibre_optique': {
            'fibre', 'optique', 'ftth', 'pon', 'splice', 'connecteur',
            'sc', 'lc', 'fc', 'mpo', 'cassette', 'épissure', 'pigtail'
        },
        'generateurs': {
            'générateur', 'groupe', 'electrogène', 'alternateur', 'diesel',
            'essence', 'kva', 'kw', 'onduleur', 'ups', 'batterie'
        },
        'cables': {
            'câble', 'cordon', 'fil', 'rj45', 'cat5', 'cat6', 'coaxial',
            'usb', 'hdmi', 'vga', 'alimentation', 'puissance'
        },
        'connecteurs': {
            'connecteur', 'prise', 'fiche', 'jack', 'terminal', 'borne',
            'domino', 'wago', 'multiprise', 'parafoudre'
        },
        'outillage': {
            'outil', 'pince', 'tournevis', 'perceuse', 'marteau', 'clé',
            'multimètre', 'testeur', 'soudure'
        },
        'securite': {
            'caméra', 'cctv', 'détecteur', 'alarme', 'sirène', 'badge',
            'contrôle d\'accès', 'surveillance'
        },
        'reseau': {
            'switch', 'routeur', 'firewall', 'point d\'accès', 'wifi',
            'modem', 'serveur', 'rack', 'baie'
        },
        'climatisation': {
            'climatiseur', 'split', 'ventilateur', 'air conditionné',
            'chauffage', 'radiateur'
        },
        'eclairage': {
            'ampoule', 'led', 'projecteur', 'spot', 'lampe', 'néon'
        }
    }
    
    # Abréviations courantes dans votre domaine
    ABBREVIATIONS = {
        'ftth': 'fibre to the home',
        'pon': 'passive optical network',
        'olt': 'optical line terminal',
        'onu': 'optical network unit',
        'sc': 'subscriber connector',
        'lc': 'lucent connector',
        'fc': 'ferrule connector',
        'mpo': 'multi-fiber push on',
        'ups': 'uninterruptible power supply',
        'kva': 'kilovolt-ampere',
        'kw': 'kilowatt',
        'cctv': 'closed-circuit television',
        'ip': 'internet protocol',
        'ptz': 'pan tilt zoom',
        'ap': 'access point',
        'nas': 'network attached storage',
        'san': 'storage area network',
        'led': 'light emitting diode',
        'rj45': 'registered jack 45',
        'utp': 'unshielded twisted pair',
        'stp': 'shielded twisted pair',
        'ftp': 'foiled twisted pair',
        'vga': 'video graphics array',
        'dvi': 'digital visual interface',
        'hdmi': 'high definition multimedia interface',
        'usb': 'universal serial bus'
    }
    
    # Synonymes français-anglais
    TRANSLATIONS = {
        # Anglais -> Français
        'fiber': 'fibre',
        'cable': 'câble',
        'connector': 'connecteur',
        'generator': 'générateur',
        'tool': 'outil',
        'camera': 'caméra',
        'switch': 'commutateur',
        'router': 'routeur',
        'firewall': 'pare-feu',
        'air conditioner': 'climatiseur',
        'light': 'éclairage',
        'bulb': 'ampoule',
        'wire': 'fil',
        'plug': 'prise',
        'socket': 'prise',
        'terminal': 'borne',
        'battery': 'batterie',
        'charger': 'chargeur',
        'transformer': 'transformateur',
        'converter': 'convertisseur',
        'tester': 'testeur',
        'meter': 'compteur',
        'sensor': 'capteur',
        'detector': 'détecteur',
        'alarm': 'alarme',
        'server': 'serveur',
        'rack': 'baie',
        'blade': 'lame',
        'module': 'module',
        'unit': 'unité'
    }
    
    @classmethod
    def get_domain_group(cls, text: str) -> Optional[str]:
        """Identifie le groupe de domaine d'un texte"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Vérifier les mots complets
        for domain, keywords in cls.DOMAIN_GROUPS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain
        
        # Vérifier les mots-clés par catégorie
        for domain, keywords in cls.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain
        
        return None
    
    @classmethod
    def expand_abbreviations(cls, text: str) -> str:
        """Développe les abréviations techniques"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in cls.ABBREVIATIONS:
                expanded_words.append(cls.ABBREVIATIONS[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    @classmethod
    def translate_technical_terms(cls, text: str) -> str:
        """Traduit les termes techniques anglais vers français"""
        words = text.split()
        translated_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in cls.TRANSLATIONS:
                translated_words.append(cls.TRANSLATIONS[word_lower])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    @classmethod
    def detect_material_type(cls, text: str) -> Dict[str, float]:
        """Détecte le type de matériel avec un score de confiance"""
        text_lower = text.lower()
        scores = {}
        
        for material_type, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                scores[material_type] = score / total_keywords
        
        return scores

# -------- Processeur de texte avancé --------
class AdvancedTextProcessor:
    """Processeur de texte spécialisé pour les noms techniques"""
    
    # Stopwords techniques à ignorer
    TECHNICAL_STOPWORDS = {
        'de', 'à', 'et', 'en', 'pour', 'avec', 'sans', 'sur', 'dans',
        'par', 'au', 'aux', 'le', 'la', 'les', 'un', 'une', 'des',
        'du', 'd\'', 'l\'', 'est', 'son', 'sa', 'ses', 'ce', 'cette',
        'ces', 'cet', 'cela', 'celui', 'celle', 'ceux', 'celles'
    }
    
    # Unités de mesure (à garder)
    MEASUREMENT_UNITS = {
        'mm', 'cm', 'm', 'km', 'mg', 'g', 'kg', 'ml', 'l', 'w', 'kw',
        'v', 'mv', 'kv', 'a', 'ma', 'ka', 'hz', 'mhz', 'ghz', 'db',
        'kva', 'va', 'wh', 'kwh', 'lux', 'c', 'f', 'k', 'pa', 'kpa',
        'psi', 'bar', 'mbar', 'rpm', 'tr/min', 'm/s', 'km/h', 'mph'
    }
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def normalize_technical_text(text: str) -> str:
        """Normalisation avancée pour textes techniques"""
        if pd.isna(text) or not text:
            return ""
        
        # Conversion en chaîne
        text = str(text)
        
        # Minuscules
        text = text.lower()
        
        # Suppression accents
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Nettoyage des caractères spéciaux
        text = re.sub(r'[^\w\s\-\.\/]', ' ', text)
        
        # Normalisation des séparateurs
        text = re.sub(r'[_\-\/\\]+', ' ', text)
        
        # Suppression espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_technical_keywords(text: str, keep_units: bool = True) -> List[str]:
        """Extrait les mots-clés techniques significatifs"""
        normalized = AdvancedTextProcessor.normalize_technical_text(text)
        words = normalized.split()
        
        keywords = []
        for word in words:
            # Ignorer les stopwords
            if word in AdvancedTextProcessor.TECHNICAL_STOPWORDS:
                continue
            
            # Garder les unités de mesure si demandé
            if keep_units and word in AdvancedTextProcessor.MEASUREMENT_UNITS:
                keywords.append(word)
                continue
            
            # Filtrer les mots trop courts (sauf acronymes)
            if len(word) < 2 and not word.isdigit():
                continue
            
            # Garder les acronymes en majuscules
            if word.isupper() and len(word) <= 5:
                keywords.append(word)
                continue
            
            # Garder les mots significatifs
            if len(word) >= 3:
                keywords.append(word)
        
        return keywords
    
    @staticmethod
    def get_technical_ngrams(text: str, n: int = 2) -> List[str]:
        """Génère des n-grammes techniques"""
        keywords = AdvancedTextProcessor.extract_technical_keywords(text)
        
        if len(keywords) < n:
            return [' '.join(keywords)]
        
        ngrams = []
        for i in range(len(keywords) - n + 1):
            ngram = ' '.join(keywords[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    @staticmethod
    def enrich_technical_text(text: str) -> str:
        """Enrichit le texte technique avec expansions et traductions"""
        # Expansion des abréviations
        expanded = DomainKnowledgeBase.expand_abbreviations(text)
        
        # Traduction des termes anglais
        translated = DomainKnowledgeBase.translate_technical_terms(expanded)
        
        return translated

# -------- Similarité technique avancée --------
class TechnicalSimilarity:
    """Calcul de similarité pour textes techniques"""
    
    @staticmethod
    def weighted_jaccard_similarity(set1: Set, set2: Set, weights: Dict = None) -> float:
        """Similarité de Jaccard avec poids pour termes techniques"""
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        # Calcul avec poids si fournis
        if weights:
            intersection_weight = sum(weights.get(word, 1.0) for word in intersection)
            union_weight = sum(weights.get(word, 1.0) for word in union)
            return intersection_weight / union_weight if union_weight > 0 else 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def technical_text_similarity(text1: str, text2: str) -> Dict[str, float]:
        """Calcule plusieurs métriques de similarité pour textes techniques"""
        
        # Enrichissement des textes
        enriched1 = AdvancedTextProcessor.enrich_technical_text(text1)
        enriched2 = AdvancedTextProcessor.enrich_technical_text(text2)
        
        # Extraction des mots-clés
        keywords1 = set(AdvancedTextProcessor.extract_technical_keywords(enriched1))
        keywords2 = set(AdvancedTextProcessor.extract_technical_keywords(enriched2))
        
        # Similarité de base
        jaccard_sim = TechnicalSimilarity.weighted_jaccard_similarity(keywords1, keywords2)
        
        # Similarité des n-grammes
        ngram_sims = []
        for n in [2, 3]:
            ngrams1 = set(AdvancedTextProcessor.get_technical_ngrams(enriched1, n))
            ngrams2 = set(AdvancedTextProcessor.get_technical_ngrams(enriched2, n))
            
            if ngrams1 and ngrams2:
                ngram_sim = len(ngrams1.intersection(ngrams2)) / max(len(ngrams1), len(ngrams2))
                ngram_sims.append(ngram_sim)
        
        avg_ngram_sim = np.mean(ngram_sims) if ngram_sims else 0.0
        
        # Similarité de domaine
        domain1 = DomainKnowledgeBase.get_domain_group(text1)
        domain2 = DomainKnowledgeBase.get_domain_group(text2)
        
        domain_sim = 0.3 if domain1 and domain2 and domain1 == domain2 else 0.0
        
        # Score composite
        composite_score = (
            jaccard_sim * 0.5 +
            avg_ngram_sim * 0.3 +
            domain_sim * 0.2
        )
        
        return {
            'jaccard': jaccard_sim,
            'ngram': avg_ngram_sim,
            'domain': domain_sim,
            'composite': composite_score,
            'domain1': domain1,
            'domain2': domain2
        }
    
    @staticmethod
    def reference_similarity(ref1: str, ref2: str) -> float:
        """Similarité entre références techniques"""
        if not ref1 or not ref2:
            return 0.0
        
        # Nettoyage des références
        clean_ref1 = re.sub(r'[^\w]', '', str(ref1).lower())
        clean_ref2 = re.sub(r'[^\w]', '', str(ref2).lower())
        
        if clean_ref1 == clean_ref2:
            return 1.0
        
        # Vérifier les préfixes communs
        if clean_ref1[:3] == clean_ref2[:3]:
            return 0.3
        
        return 0.0

# -------- Item Technique --------
class TechnicalItem:
    """Représente un item technique avec toutes ses propriétés"""
    
    def __init__(self, **kwargs):
        # Attributs de base
        self.id = kwargs.get('id', '')
        self.reference = kwargs.get('reference', '')
        self.name = kwargs.get('name', '')  # Anciennement item_name
        self.french_name = kwargs.get('french_name', '')
        self.status = kwargs.get('status', '')
        self.uom_name = kwargs.get('uom_name', '')
        self.type_name = kwargs.get('type_name', '')
        self.sub_category_name = kwargs.get('sub_category_name', '')
        self.category_name = kwargs.get('category_name', '')
        self.last_use = kwargs.get('last_use', '')
        self.last_price = kwargs.get('last_price', '')
        self.requestor_name = kwargs.get('requestor_name', '')
        self.company_name = kwargs.get('company_name', '')
        self.department_name = kwargs.get('department_name', '')
        self.created_at = kwargs.get('created_at', '')
        self.updated_at = kwargs.get('updated_at', '')
        
        # Attributs calculés
        self._calculate_derived_fields()
    
    def _calculate_derived_fields(self):
        """Calcule les champs dérivés"""
        # Texte principal pour comparaison
        self.main_text = f"{self.name} {self.french_name}".strip()
        
        # Normalisation
        self.normalized_text = AdvancedTextProcessor.normalize_technical_text(self.main_text)
        
        # Mots-clés techniques
        self.technical_keywords = AdvancedTextProcessor.extract_technical_keywords(self.main_text)
        
        # Domaine détecté
        self.domain = DomainKnowledgeBase.get_domain_group(self.main_text)
        
        # Type de matériel
        self.material_scores = DomainKnowledgeBase.detect_material_type(self.main_text)
        
        # Texte enrichi
        self.enriched_text = AdvancedTextProcessor.enrich_technical_text(self.main_text)
    
    def get_similarity_fingerprint(self) -> str:
        """Crée une empreinte pour comparaison rapide"""
        components = [
            self.domain or 'unknown',
            '|'.join(sorted(self.technical_keywords[:5])),
            self.type_name or '',
            self.category_name or ''
        ]
        return '#'.join(filter(None, components))
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour affichage"""
        return {
            'id': self.id,
            'reference': self.reference,
            'name': self.name,
            'french_name': self.french_name,
            'type_name': self.type_name,
            'category_name': self.category_name,
            'company_name': self.company_name,
            'domain': self.domain,
            'keywords': ', '.join(self.technical_keywords[:5])
        }

# -------- Détecteur de Doublons Technique --------
class TechnicalDuplicateDetector:
    """Détecteur spécialisé pour items techniques"""
    
    def __init__(self, items: List[TechnicalItem]):
        self.items = items
        self._build_indexes()
    
    def _build_indexes(self):
        """Construit les index pour recherche rapide"""
        self.by_domain = {}
        self.by_type = {}
        self.by_category = {}
        self.by_company = {}
        self.by_keyword = {}
        
        for item in self.items:
            # Index par domaine
            if item.domain:
                self.by_domain.setdefault(item.domain, []).append(item)
            
            # Index par type
            if item.type_name:
                self.by_type.setdefault(item.type_name, []).append(item)
            
            # Index par catégorie
            if item.category_name:
                self.by_category.setdefault(item.category_name, []).append(item)
            
            # Index par entreprise
            if item.company_name:
                self.by_company.setdefault(item.company_name, []).append(item)
            
            # Index par mot-clé
            for keyword in item.technical_keywords[:5]:  # 5 premiers mots-clés
                self.by_keyword.setdefault(keyword, []).append(item)
    
    def find_technical_duplicates(self, target_item: TechnicalItem, 
                                threshold: float = 0.6) -> List[Tuple[TechnicalItem, Dict]]:
        """Trouve les doublons techniques"""
        results = []
        
        # 1. Recherche par empreinte exacte
        target_fingerprint = target_item.get_similarity_fingerprint()
        for item in self.items:
            if item.id == target_item.id:
                continue
            
            if item.get_similarity_fingerprint() == target_fingerprint:
                similarity = {
                    'composite': 1.0,
                    'method': 'fingerprint_exact',
                    'domain_match': True if item.domain == target_item.domain else False
                }
                results.append((item, similarity))
        
        if len(results) >= 10:
            return results[:10]
        
        # 2. Collecte des candidats potentiels
        candidates = []
        
        # Par domaine
        if target_item.domain in self.by_domain:
            candidates.extend(self.by_domain[target_item.domain])
        
        # Par type
        if target_item.type_name in self.by_type:
            candidates.extend(self.by_type[target_item.type_name])
        
        # Par catégorie
        if target_item.category_name in self.by_category:
            candidates.extend(self.by_category[target_item.category_name])
        
        # Par mot-clé commun
        for keyword in target_item.technical_keywords[:3]:
            if keyword in self.by_keyword:
                candidates.extend(self.by_keyword[keyword])
        
        # Déduplication
        candidates = list({item.id: item for item in candidates if item.id != target_item.id}.values())
        
        # 3. Calcul de similarité détaillé
        for candidate in candidates[:50]:  # Limiter pour performance
            similarity = TechnicalSimilarity.technical_text_similarity(
                target_item.main_text,
                candidate.main_text
            )
            
            # Bonus pour même référence
            ref_sim = TechnicalSimilarity.reference_similarity(
                target_item.reference,
                candidate.reference
            )
            
            similarity['reference'] = ref_sim
            
            # Score final pondéré
            final_score = (
                similarity['composite'] * 0.7 +
                ref_sim * 0.3
            )
            
            similarity['final'] = final_score
            
            if final_score >= threshold:
                results.append((candidate, similarity))
        
        # 4. Tri par score final
        results.sort(key=lambda x: x[1]['final'], reverse=True)
        return results[:10]
    
    def analyze_domains_distribution(self) -> Dict:
        """Analyse la distribution par domaine"""
        distribution = {}
        for item in self.items:
            domain = item.domain or 'non_classé'
            distribution[domain] = distribution.get(domain, 0) + 1
        
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    def find_similar_by_domain(self, domain: str, limit: int = 20) -> List[TechnicalItem]:
        """Trouve des items similaires par domaine"""
        if domain in self.by_domain:
            return self.by_domain[domain][:limit]
        return []

# -------- Interface Streamlit --------
def main():
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .domain-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.1rem;
        }
        
        .fibre-badge { background: #8B5CF6; color: white; }
        .generateur-badge { background: #10B981; color: white; }
        .cable-badge { background: #F59E0B; color: white; }
        .connecteur-badge { background: #EF4444; color: white; }
        .outil-badge { background: #3B82F6; color: white; }
        .securite-badge { background: #EC4899; color: white; }
        .reseau-badge { background: #6366F1; color: white; }
        .other-badge { background: #6B7280; color: white; }
        
        .duplicate-card {
            border-left: 4px solid #8B5CF6;
            padding: 1rem;
            margin: 0.5rem 0;
            background: #F8FAFC;
            border-radius: 8px;
        }
        
        .similarity-bar {
            height: 8px;
            background: #E5E7EB;
            border-radius: 4px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        
        .similarity-fill {
            height: 100%;
            background: linear-gradient(90deg, #10B981, #3B82F6);
            border-radius: 4px;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        .mapping-table {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">🔍 Détection Intelligente de Doublons Techniques</h1>', unsafe_allow_html=True)
    st.caption("Optimisé pour vos items techniques : fibre optique, générateurs, câbles, connecteurs, etc.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📤 Chargement des données")
        
        uploaded_file = st.file_uploader(
            "Téléversez votre fichier CSV",
            type=['csv'],
            help="Peut contenir des colonnes génériques (Colonne1, Colonne2...)"
        )
        
        if uploaded_file:
            try:
                # Essayer différents séparateurs
                try:
                    df = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='utf-8', on_bad_lines='skip')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, sep=',', dtype=str, encoding='utf-8', on_bad_lines='skip')
                    except:
                        # Lire comme texte et essayer de parser
                        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                        lines = content.split('\n')
                        
                        # Détecter le séparateur
                        first_line = lines[0] if lines else ''
                        if ';' in first_line and first_line.count(';') > first_line.count(','):
                            sep = ';'
                        else:
                            sep = ','
                        
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), sep=sep, dtype=str, on_bad_lines='skip')
                
                # Stocker le dataframe brut
                st.session_state.raw_df = df
                st.session_state.filename = uploaded_file.name
                
                st.success(f"✅ Fichier chargé : {uploaded_file.name}")
                st.metric("Lignes", len(df))
                st.metric("Colonnes", len(df.columns))
                
                # Afficher un aperçu
                with st.expander("📋 Aperçu des données"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Vérifier si les colonnes attendues sont présentes
                if 'name' in df.columns and 'reference' in df.columns:
                    # Les colonnes ont déjà les bons noms
                    st.session_state.column_mapping = {col: col for col in EXPECTED_COLUMNS if col in df.columns}
                    st.info("✅ Colonnes détectées automatiquement")
                else:
                    # Demander le mapping
                    st.info("🔧 Mapping des colonnes nécessaire")
                    if st.button("🗺️ Configurer le mapping des colonnes"):
                        st.session_state.need_mapping = True
                
            except Exception as e:
                st.error(f"❌ Erreur de chargement : {str(e)}")
                return
    
    # Contenu principal
    if 'raw_df' not in st.session_state:
        st.info("👈 Veuillez téléverser un fichier CSV dans la barre latérale")
        
        with st.expander("📋 Format attendu du CSV"):
            st.markdown("""
            ### Votre fichier peut avoir :
            
            **Soit les noms de colonnes exacts :**
            - `name` : Nom principal de l'item
            - `reference` : Référence unique
            - `category_name` : Catégorie
            - `type_name` : Type
            
            **Soit des noms génériques (Colonne1, Colonne2...) :**
            - Nous vous aiderons à mapper les colonnes
            - Analyse automatique du contenu
            - Interface interactive de mapping
            """)
        
        return
    
    # Étape 1 : Mapping des colonnes si nécessaire
    if hasattr(st.session_state, 'need_mapping') and st.session_state.need_mapping:
        st.header("🗺️ Mapping des colonnes")
        
        df = st.session_state.raw_df
        
        # Détection automatique
        auto_mapping = detect_columns_automatically(df)
        
        st.markdown("### Détection automatique")
        if auto_mapping:
            st.success(f"✅ {len(auto_mapping)} colonnes détectées automatiquement")
            
            # Afficher le mapping détecté
            mapping_df = pd.DataFrame([
                {"Champ attendu": field, "Colonne détectée": col, "Confiance": "Haute"}
                for field, col in auto_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True)
        else:
            st.warning("⚠️ Aucune colonne détectée automatiquement")
        
        # Mapping interactif
        st.markdown("### 📝 Mapping manuel")
        st.info("Sélectionnez pour chaque champ important la colonne correspondante dans votre fichier")
        
        column_mapping = {}
        
        # Champs critiques
        critical_fields = ['name', 'reference']
        
        for field in critical_fields:
            st.markdown(f"**{field.upper()}**")
            
            # Suggestions
            suggestions = []
            if field in auto_mapping:
                suggestions.append(auto_mapping[field])
            
            # Ajouter d'autres colonnes
            for col in df.columns:
                if col not in suggestions:
                    suggestions.append(col)
            
            selected_col = st.selectbox(
                f"Colonne pour '{field}' :",
                options=suggestions,
                key=f"map_{field}"
            )
            
            if selected_col:
                column_mapping[field] = selected_col
        
        # Champs optionnels
        optional_fields = ['id', 'category_name', 'type_name', 'company_name', 'french_name']
        
        with st.expander("Champs optionnels"):
            for field in optional_fields:
                suggestions = []
                if field in auto_mapping:
                    suggestions.append(auto_mapping[field])
                
                for col in df.columns:
                    if col not in suggestions and col not in column_mapping.values():
                        suggestions.append(col)
                
                suggestions.append("(Ignorer)")
                
                selected_col = st.selectbox(
                    f"Colonne pour '{field}' (optionnel) :",
                    options=suggestions,
                    key=f"map_opt_{field}"
                )
                
                if selected_col and selected_col != "(Ignorer)":
                    column_mapping[field] = selected_col
        
        # Validation
        if st.button("✅ Valider le mapping", type="primary"):
            if 'name' in column_mapping and 'reference' in column_mapping:
                st.session_state.column_mapping = column_mapping
                st.session_state.need_mapping = False
                st.success("Mapping validé !")
                st.rerun()
            else:
                st.error("Veuillez mapper au moins les champs 'name' et 'reference'")
    
    # Étape 2 : Traitement des données
    elif 'column_mapping' in st.session_state:
        df = st.session_state.raw_df
        
        # Appliquer le mapping
        processed_df = pd.DataFrame()
        
        for field, source_col in st.session_state.column_mapping.items():
            if source_col in df.columns:
                processed_df[field] = df[source_col]
            else:
                processed_df[field] = ''
        
        # Remplir les colonnes manquantes
        for col in EXPECTED_COLUMNS:
            if col not in processed_df.columns:
                processed_df[col] = ''
        
        # Conversion en TechnicalItem
        if 'technical_items' not in st.session_state:
            with st.spinner("Traitement des données..."):
                technical_items = []
                
                for idx, row in processed_df.iterrows():
                    item_data = {col: row.get(col, '') for col in EXPECTED_COLUMNS}
                    technical_item = TechnicalItem(**item_data)
                    technical_items.append(technical_item)
                
                # Création du détecteur
                detector = TechnicalDuplicateDetector(technical_items)
                
                # Stockage en session
                st.session_state.df = processed_df
                st.session_state.detector = detector
                st.session_state.technical_items = technical_items
        
        # Affichage du mapping
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📊 Mapping appliqué")
            
            mapping_list = []
            for field, source in st.session_state.column_mapping.items():
                mapping_list.append(f"**{field}** ← {source}")
            
            st.markdown("\n".join(mapping_list[:5]), unsafe_allow_html=True)
            
            if len(st.session_state.column_mapping) > 5:
                with st.expander("Voir tout le mapping"):
                    st.markdown("\n".join(mapping_list[5:]), unsafe_allow_html=True)
            
            # Bouton de réinitialisation
            if st.button("🔄 Réinitialiser l'analyse", use_container_width=True):
                for key in ['df', 'detector', 'technical_items', 'column_mapping', 'raw_df']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Si les données sont prêtes, afficher l'interface principale
    if 'detector' in st.session_state:
        # Onglets principaux
        tab1, tab2, tab3 = st.tabs([
            "🔍 Recherche de Doublons", 
            "📊 Analyse par Domaine", 
            "⚙️ Paramètres Avancés"
        ])
        
        # Tab 1 - Recherche de Doublons
        with tab1:
            st.header("🔍 Recherche de doublons techniques")
            
            # Mode de recherche
            search_mode = st.radio(
                "Mode de recherche :",
                ["Sélectionner un item existant", "Saisir un nouvel item"],
                horizontal=True,
                key="search_mode"
            )
            
            if search_mode == "Sélectionner un item existant":
                # Liste des items disponibles
                item_options = [f"{item.name} | {item.reference} | {item.category_name}" 
                              for item in st.session_state.technical_items[:200]]
                
                selected_item_str = st.selectbox(
                    "Sélectionnez un item à analyser :",
                    options=item_options,
                    index=0,
                    help="Affiche : Nom | Référence | Catégorie"
                )
                
                # Extraire l'ID ou le nom de l'item sélectionné
                selected_name = selected_item_str.split(' | ')[0]
                target_item = None
                
                for item in st.session_state.technical_items:
                    if item.name == selected_name:
                        target_item = item
                        break
            
            else:
                # Saisie manuelle
                st.markdown("### 📝 Saisie d'un nouvel item")
                
                col_name, col_ref = st.columns(2)
                with col_name:
                    item_name = st.text_input("Nom de l'item *", "", 
                                            placeholder="Ex: Câble fibre optique 50m")
                with col_ref:
                    item_reference = st.text_input("Référence", "", 
                                                 placeholder="Ex: FIB-50M")
                
                col_cat, col_type = st.columns(2)
                with col_cat:
                    item_category = st.text_input("Catégorie", "", 
                                                placeholder="Ex: Fibre optique")
                with col_type:
                    item_type = st.text_input("Type", "", 
                                            placeholder="Ex: Câble")
                
                if st.button("Analyser cet item", key="analyze_manual"):
                    if item_name:
                        target_item = TechnicalItem(
                            name=item_name,
                            reference=item_reference,
                            category_name=item_category,
                            type_name=item_type
                        )
                    else:
                        st.warning("Veuillez saisir au moins le nom de l'item")
                        target_item = None
                else:
                    target_item = None
            
            # Affichage et analyse de l'item cible
            if target_item:
                st.markdown("### 📋 Analyse de l'item")
                
                # Métriques de l'item
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Nom", target_item.name[:30] + "..." if len(target_item.name) > 30 else target_item.name)
                with col_info2:
                    badge_class = {
                        'fibre_optique': 'fibre-badge',
                        'generateurs': 'generateur-badge',
                        'cables': 'cable-badge',
                        'connecteurs': 'connecteur-badge',
                        'outillage': 'outil-badge',
                        'securite': 'securite-badge',
                        'reseau': 'reseau-badge'
                    }.get(target_item.domain, 'other-badge')
                    
                    domain_display = target_item.domain or "Non classé"
                    st.markdown(f'<span class="domain-badge {badge_class}">{domain_display}</span>', 
                              unsafe_allow_html=True)
                with col_info3:
                    st.metric("Mots-clés", len(target_item.technical_keywords))
                
                # Détails supplémentaires
                with st.expander("📖 Détails techniques"):
                    col_detail1, col_detail2 = st.columns(2)
                    with col_detail1:
                        st.write("**Mots-clés extraits :**")
                        keywords_html = ""
                        for keyword in target_item.technical_keywords[:10]:
                            keywords_html += f'<span style="background:#E5E7EB; padding:2px 8px; margin:2px; border-radius:10px; font-size:0.8rem;">{keyword}</span> '
                        st.markdown(keywords_html, unsafe_allow_html=True)
                    
                    with col_detail2:
                        if target_item.material_scores:
                            st.write("**Scores de matériel :**")
                            for material, score in list(target_item.material_scores.items())[:5]:
                                st.progress(score, text=f"{material}: {score:.1%}")
                
                # Recherche de doublons
                st.markdown("---")
                st.markdown("### 🎯 Recherche de doublons techniques")
                
                # Paramètres de recherche
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    similarity_threshold = st.slider(
                        "Seuil de similarité",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.6,
                        step=0.05,
                        help="Plus bas = plus de résultats (peut inclure des faux positifs)"
                    )
                
                with col_param2:
                    max_results = st.slider(
                        "Nombre max de résultats",
                        min_value=5,
                        max_value=20,
                        value=10,
                        step=1
                    )
                
                if st.button("🔍 Lancer la recherche", type="primary", key="search_duplicates"):
                    with st.spinner("Recherche de doublons en cours..."):
                        duplicates = st.session_state.detector.find_technical_duplicates(
                            target_item, 
                            threshold=similarity_threshold
                        )
                        
                        if not duplicates:
                            st.success("✅ Aucun doublon technique détecté !")
                        else:
                            st.warning(f"⚠️ {len(duplicates)} doublon(s) potentiel(s) détecté(s)")
                            
                            # Calcul du score moyen
                            avg_score = np.mean([sim['final'] for _, sim in duplicates])
                            st.metric("Similarité moyenne", f"{avg_score:.1%}")
                            
                            # Affichage des résultats
                            for idx, (item, similarity) in enumerate(duplicates[:max_results]):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="duplicate-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <div>
                                                <strong>{item.name}</strong><br>
                                                <small>
                                                    Réf: {item.reference} | 
                                                    Cat: {item.category_name} | 
                                                    Type: {item.type_name}
                                                </small>
                                            </div>
                                            <div style="text-align: right;">
                                                <span style="font-size: 1.2rem; font-weight: bold; color: #8B5CF6;">
                                                    {similarity['final']:.1%}
                                                </span><br>
                                                <small style="color: #6B7280;">score</small>
                                            </div>
                                        </div>
                                        
                                        <div class="similarity-bar">
                                            <div class="similarity-fill" style="width: {similarity['final'] * 100}%"></div>
                                        </div>
                                        
                                        <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;">
                                            <strong>Détail :</strong><br>
                                            • Similarité textuelle : {similarity['jaccard']:.1%}<br>
                                            • Domaine : {similarity.get('domain1', 'N/A')} → {similarity.get('domain2', 'N/A')}<br>
                                            • Référence : {similarity['reference']:.1%}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Option d'export
                            if duplicates:
                                export_data = []
                                for item, similarity in duplicates[:max_results]:
                                    export_data.append({
                                        'nom_item_cible': target_item.name,
                                        'nom_item_similaire': item.name,
                                        'reference_similaire': item.reference,
                                        'categorie_similaire': item.category_name,
                                        'score_final': similarity['final'],
                                        'score_textuel': similarity['jaccard'],
                                        'score_domaine': similarity['domain'],
                                        'meme_domaine': similarity.get('domain1') == similarity.get('domain2')
                                    })
                                
                                export_df = pd.DataFrame(export_data)
                                csv_data = export_df.to_csv(index=False).encode('utf-8')
                                
                                st.download_button(
                                    "📥 Exporter les résultats",
                                    data=csv_data,
                                    file_name="doublons_detectes.csv",
                                    mime="text/csv"
                                )
        
        # Tab 2 - Analyse par Domaine
        with tab2:
            st.header("📊 Analyse par domaine technique")
            
            if 'detector' in st.session_state:
                # Distribution par domaine
                distribution = st.session_state.detector.analyze_domains_distribution()
                
                col_chart, col_stats = st.columns([3, 2])
                
                with col_chart:
                    st.markdown("### 🗂️ Répartition des items par domaine")
                    
                    # Préparation des données pour le graphique
                    dist_df = pd.DataFrame({
                        'Domaine': list(distribution.keys()),
                        'Nombre': list(distribution.values())
                    }).sort_values('Nombre', ascending=False)
                    
                    # Graphique à barres
                    st.bar_chart(dist_df.set_index('Domaine'))
                
                with col_stats:
                    st.markdown("### 📈 Statistiques")
                    
                    total_items = len(st.session_state.technical_items)
                    classified_items = sum(count for domain, count in distribution.items() 
                                         if domain != 'non_classé')
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Total items", total_items)
                    with col_stat2:
                        st.metric("Items classés", classified_items)
                    
                    # Taux de classification
                    classification_rate = classified_items / total_items if total_items > 0 else 0
                    st.metric("Taux de classification", f"{classification_rate:.1%}")
                
                # Analyse détaillée par domaine
                st.markdown("### 🔍 Analyse par domaine")
                
                selected_domain = st.selectbox(
                    "Sélectionnez un domaine pour analyse détaillée :",
                    options=['Tous'] + list(distribution.keys())
                )
                
                if selected_domain != 'Tous':
                    # Filtrer les items du domaine sélectionné
                    domain_items = st.session_state.detector.find_similar_by_domain(selected_domain, 50)
                    
                    if domain_items:
                        st.markdown(f"**{len(domain_items)} items dans le domaine '{selected_domain}'**")
                        
                        # Afficher un échantillon
                        sample_data = []
                        for item in domain_items[:20]:
                            sample_data.append({
                                'Nom': item.name,
                                'Référence': item.reference,
                                'Catégorie': item.category_name,
                                'Type': item.type_name,
                                'Mots-clés': ', '.join(item.technical_keywords[:3])
                            })
                        
                        sample_df = pd.DataFrame(sample_data)
                        st.dataframe(sample_df, use_container_width=True)
        
        # Tab 3 - Paramètres Avancés
        with tab3:
            st.header("⚙️ Paramètres avancés")
            
            st.markdown("### 🎛️ Configuration de la détection")
            
            # Configuration de la base de connaissances
            with st.expander("🧠 Configuration de la base de connaissances"):
                st.markdown("**Domaines techniques reconnus :**")
                
                for domain, keywords in DomainKnowledgeBase.CATEGORY_KEYWORDS.items():
                    with st.expander(f"Domaine : {domain}"):
                        st.write(f"Mots-clés : {', '.join(list(keywords)[:10])}")
                
                # Option pour ajouter des mots-clés personnalisés
                st.markdown("**Ajouter des mots-clés personnalisés :**")
                custom_domain = st.text_input("Nom du domaine", placeholder="Ex: climatisation")
                custom_keywords = st.text_area(
                    "Mots-clés (séparés par des virgules)",
                    placeholder="climatiseur, split, inverter, ventilation"
                )
                
                if st.button("Ajouter aux domaines reconnus"):
                    if custom_domain and custom_keywords:
                        keywords_list = [k.strip() for k in custom_keywords.split(',')]
                        DomainKnowledgeBase.CATEGORY_KEYWORDS[custom_domain] = set(keywords_list)
                        st.success(f"✅ Domaine '{custom_domain}' ajouté avec {len(keywords_list)} mots-clés")
            
            # Export des données analysées
            st.markdown("---")
            st.markdown("### 💾 Export des données")
            
            if 'df' in st.session_state:
                # Préparation des données enrichies
                enriched_data = []
                for item in st.session_state.technical_items:
                    enriched_data.append({
                        **item.to_dict(),
                        'domain_detected': item.domain,
                        'technical_keywords': ', '.join(item.technical_keywords),
                        'material_scores': json.dumps(item.material_scores)
                    })
                
                enriched_df = pd.DataFrame(enriched_data)
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # Export CSV enrichi
                    csv_enriched = enriched_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Exporter données enrichies (CSV)",
                        data=csv_enriched,
                        file_name="donnees_enrichies.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export2:
                    # Export des domaines détectés
                    domain_summary = pd.DataFrame({
                        'Domaine': list(DomainKnowledgeBase.CATEGORY_KEYWORDS.keys()),
                        'Description': ["Matériel fibre optique et télécom"] * len(DomainKnowledgeBase.CATEGORY_KEYWORDS)
                    })
                    
                    csv_domains = domain_summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Exporter les domaines reconnus",
                        data=csv_domains,
                        file_name="domaines_techniques.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Réinitialisation
            st.markdown("---")
            st.markdown("### 🔄 Maintenance")
            
            if st.button("🗑️ Vider le cache et réinitialiser", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_data.clear()
                st.rerun()

if __name__ == "__main__":
    main()