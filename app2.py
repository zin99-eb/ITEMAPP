# ================================================================
# Détection Intelligente de Doublons - CORRIGÉ V2
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
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
    'id': 'id',
    'reference': 'reference',
    'name': 'name',
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
    """Tente de détecter automatiquement quelle colonne correspond à quel champ"""
    column_mapping = {}
    
    df_str = df.astype(str)
    
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
        }
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        col_data = df_str[col]
        
        best_match = None
        best_score = 0
        
        for field, rules in detection_rules.items():
            score = 0
            
            for keyword in rules['keywords']:
                if keyword in col_lower:
                    score += 3
            
            try:
                if rules['type_check'](col_data):
                    score += 2
            except:
                pass
            
            if score > best_score:
                best_score = score
                best_match = field
        
        if best_match and best_score >= 2:
            column_mapping[best_match] = col
    
    return column_mapping

# -------- Base de Connaissance --------
class DomainKnowledgeBase:
    """Base de connaissance spécialisée pour vos domaines"""
    
    CATEGORY_KEYWORDS = {
        'fibre_optique': {'fibre', 'optique', 'ftth', 'pon', 'splice', 'connecteur'},
        'generateurs': {'générateur', 'groupe', 'electrogène', 'alternateur', 'diesel'},
        'cables': {'câble', 'cordon', 'fil', 'rj45', 'cat5', 'cat6', 'coaxial'},
        'connecteurs': {'connecteur', 'prise', 'fiche', 'jack', 'terminal', 'borne'},
        'outillage': {'outil', 'pince', 'tournevis', 'perceuse', 'marteau', 'clé'},
        'securite': {'caméra', 'cctv', 'détecteur', 'alarme', 'sirène', 'badge'},
        'reseau': {'switch', 'routeur', 'firewall', 'point d\'accès', 'wifi'},
    }
    
    @classmethod
    def get_domain_group(cls, text: str) -> Optional[str]:
        """Identifie le groupe de domaine d'un texte"""
        if not text or pd.isna(text):
            return None
            
        text_lower = str(text).lower()
        
        for domain, keywords in cls.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain  
        return None

# -------- Processeur de texte --------
class AdvancedTextProcessor:
    """Processeur de texte spécialisé pour les noms techniques"""
    
    TECHNICAL_STOPWORDS = {
        'de', 'à', 'et', 'en', 'pour', 'avec', 'sans', 'sur', 'dans',
        'par', 'au', 'aux', 'le', 'la', 'les', 'un', 'une', 'des',
        'du', 'd\'', 'l\'', 'est', 'son', 'sa', 'ses'
    }
    
    @staticmethod
    def normalize_technical_text(text: str) -> str:
        """Normalisation pour textes techniques"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s\-\.\/]', ' ', text)
        text = re.sub(r'[_\-\/\\]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_technical_keywords(text: str) -> List[str]:
        """Extrait les mots-clés techniques significatifs"""
        normalized = AdvancedTextProcessor.normalize_technical_text(text)
        words = normalized.split()
        
        keywords = []
        for word in words:
            if word in AdvancedTextProcessor.TECHNICAL_STOPWORDS:
                continue
            
            if len(word) < 2 and not word.isdigit():
                continue
            
            if word.isupper() and len(word) <= 5:
                keywords.append(word)
                continue
            
            if len(word) >= 3:
                keywords.append(word)
        
        return keywords

# -------- Similarité technique --------
class TechnicalSimilarity:
    """Calcul de similarité pour textes techniques"""
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Similarité de Jaccard"""
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def technical_text_similarity(text1: str, text2: str) -> Dict[str, float]:
        """Calcule la similarité entre deux textes techniques"""
        
        keywords1 = set(AdvancedTextProcessor.extract_technical_keywords(text1))
        keywords2 = set(AdvancedTextProcessor.extract_technical_keywords(text2))
        
        jaccard_sim = TechnicalSimilarity.jaccard_similarity(keywords1, keywords2)
        
        domain1 = DomainKnowledgeBase.get_domain_group(text1)
        domain2 = DomainKnowledgeBase.get_domain_group(text2)
        
        domain_sim = 0.3 if domain1 and domain2 and domain1 == domain2 else 0.0
        
        composite_score = (
            jaccard_sim * 0.8 +
            domain_sim * 0.2
        )
        
        return {
            'jaccard': jaccard_sim,
            'domain': domain_sim,
            'composite': composite_score,
            'domain1': domain1,
            'domain2': domain2
        }

# -------- Item Technique --------
class TechnicalItem:
    """Représente un item technique avec toutes ses propriétés"""
    
    def __init__(self, **kwargs):
        # S'assurer que tous les attributs sont des strings
        for key in kwargs:
            if kwargs[key] is None or pd.isna(kwargs[key]):
                kwargs[key] = ''
            else:
                kwargs[key] = str(kwargs[key])
        
        # Attributs de base
        self.id = kwargs.get('id', '')
        self.reference = kwargs.get('reference', '')
        self.name = kwargs.get('name', '')
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
    
    def get_similarity_fingerprint(self) -> str:
        """Crée une empreinte pour comparaison rapide"""
        # S'assurer que tous les éléments sont des strings
        domain_str = str(self.domain) if self.domain else 'unknown'
        keywords_str = '|'.join(sorted([str(k) for k in self.technical_keywords[:5]]))
        type_str = str(self.type_name) if self.type_name else ''
        category_str = str(self.category_name) if self.category_name else ''
        
        components = [domain_str, keywords_str, type_str, category_str]
        
        # Filtrer les chaînes vides et joindre
        filtered = [c for c in components if c and str(c).strip()]
        return '#'.join(filtered) if filtered else 'empty'
    
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
        
        for item in self.items:
            # Index par domaine
            if item.domain:
                self.by_domain.setdefault(item.domain, []).append(item)
    
    def find_technical_duplicates(self, target_item: TechnicalItem, 
                                threshold: float = 0.6) -> List[Tuple[TechnicalItem, Dict]]:
        """Trouve les doublons techniques"""
        results = []
        
        # 1. Recherche par empreinte exacte
        target_fingerprint = target_item.get_similarity_fingerprint()
        for item in self.items:
            if item.id == target_item.id:
                continue
            
            item_fingerprint = item.get_similarity_fingerprint()
            if item_fingerprint == target_fingerprint and item_fingerprint != 'empty':
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
        
        # Déduplication
        candidates = list({item.id: item for item in candidates if item.id != target_item.id}.values())
        
        # 3. Calcul de similarité détaillé
        for candidate in candidates[:50]:
            similarity = TechnicalSimilarity.technical_text_similarity(
                target_item.main_text,
                candidate.main_text
            )
            
            similarity['final'] = similarity['composite']
            
            if similarity['final'] >= threshold:
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">🔍 Détection Intelligente de Doublons Techniques</h1>', unsafe_allow_html=True)
    st.caption("Optimisé pour vos items techniques")
    
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
                    df = pd.read_csv(uploaded_file, sep=',', dtype=str, encoding='utf-8', on_bad_lines='skip')
                
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
                    st.session_state.column_mapping = {col: col for col in EXPECTED_COLUMNS if col in df.columns}
                    st.info("✅ Colonnes détectées automatiquement")
                else:
                    st.info("🔧 Mapping des colonnes nécessaire")
                    if st.button("🗺️ Configurer le mapping"):
                        st.session_state.need_mapping = True
                
            except Exception as e:
                st.error(f"❌ Erreur de chargement : {str(e)}")
                return
    
    # Contenu principal
    if 'raw_df' not in st.session_state:
        st.info("👈 Veuillez téléverser un fichier CSV dans la barre latérale")
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
        
        # Mapping interactif
        st.markdown("### 📝 Mapping manuel")
        
        column_mapping = {}
        critical_fields = ['name', 'reference']
        
        for field in critical_fields:
            st.markdown(f"**{field.upper()}**")
            
            # Suggestions
            suggestions = []
            if field in auto_mapping:
                suggestions.append(auto_mapping[field])
            
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
                processed_df[field] = df[source_col].astype(str)
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
        
        # Affichage principal
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
            item_options = [f"{item.name[:50]}... | {item.reference}" 
                          if len(item.name) > 50 else f"{item.name} | {item.reference}"
                          for item in st.session_state.technical_items[:100]]
            
            if not item_options:
                st.warning("Aucun item trouvé dans les données")
                return
                
            selected_item_str = st.selectbox(
                "Sélectionnez un item à analyser :",
                options=item_options,
                index=0
            )
            
            # Extraire le nom de l'item sélectionné
            selected_name = selected_item_str.split(' | ')[0]
            target_item = None
            
            for item in st.session_state.technical_items:
                if item.name.startswith(selected_name.replace('...', '')):
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
            
            if st.button("Analyser cet item", key="analyze_manual"):
                if item_name:
                    target_item = TechnicalItem(
                        name=item_name,
                        reference=item_reference
                    )
                else:
                    st.warning("Veuillez saisir au moins le nom de l'item")
                    target_item = None
            else:
                target_item = None
        
        # Affichage et analyse de l'item cible
        if target_item:
            st.markdown("### 📋 Analyse de l'item")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Nom", target_item.name[:50] + "..." if len(target_item.name) > 50 else target_item.name)
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
            
            # Recherche de doublons
            st.markdown("---")
            st.markdown("### 🎯 Recherche de doublons techniques")
            
            similarity_threshold = st.slider(
                "Seuil de similarité",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                step=0.05
            )
            
            max_results = st.slider(
                "Nombre max de résultats",
                min_value=5,
                max_value=20,
                value=10,
                step=1
            )
            
            if st.button("🔍 Lancer la recherche", type="primary"):
                with st.spinner("Recherche de doublons en cours..."):
                    duplicates = st.session_state.detector.find_technical_duplicates(
                        target_item, 
                        threshold=similarity_threshold
                    )
                    
                    if not duplicates:
                        st.success("✅ Aucun doublon technique détecté !")
                    else:
                        st.warning(f"⚠️ {len(duplicates)} doublon(s) potentiel(s) détecté(s)")
                        
                        # Affichage des résultats
                        for idx, (item, similarity) in enumerate(duplicates[:max_results]):
                            with st.container():
                                st.markdown(f"""
                                <div class="duplicate-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <strong>{item.name[:100]}{'...' if len(item.name) > 100 else ''}</strong><br>
                                            <small>
                                                Réf: {item.reference} | 
                                                Cat: {item.category_name}
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
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Option pour voir plus de détails
                                with st.expander("📊 Détails de la similarité"):
                                    st.write(f"**Similarité Jaccard:** {similarity['jaccard']:.1%}")
                                    st.write(f"**Domaine 1:** {similarity.get('domain1', 'N/A')}")
                                    st.write(f"**Domaine 2:** {similarity.get('domain2', 'N/A')}")
                                    st.write(f"**Score composite:** {similarity['composite']:.1%}")

if __name__ == "__main__":
    main()