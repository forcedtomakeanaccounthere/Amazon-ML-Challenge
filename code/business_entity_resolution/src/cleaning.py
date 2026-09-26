"""
Data cleaning and normalization module for Business Entity Resolution.
Handles text normalization, legal suffix extraction, and address parsing.
"""

import re
import unicodedata
from typing import Dict, Tuple, Optional


class BusinessDataCleaner:
    """
    Comprehensive data cleaner for business entity records.
    Handles name normalization, legal suffix extraction, and address parsing.
    """
    
    def __init__(self):
        # Legal suffix dictionaries by country
        self.legal_suffixes = {
            'US': [
                'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
                'co', 'company', 'llp', 'lp', 'plc', 'pa', 'pc',
            ],
            'India': [
                'pvt', 'private', 'ltd', 'limited', 'llp', 'llc',
                'proprietorship', 'partnership', 'opc', 'one person company',
            ],
            'France': [
                'sarl', 'sas', 'sa', 'eurl', 'sci', 'sasu', 'snc', 'scs',
                'gie', 'sep', 'ei', 'societe',
            ],
        }
        
        # Street type abbreviations (cross-country)
        self.street_types = {
            'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
            'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
            'pkwy': 'parkway', 'hwy': 'highway', 'sq': 'square',
            'rue': 'rue', 'avenue': 'avenue', 'boulevard': 'boulevard',
        }
        
        # Landmark phrases to detect and flag
        self.landmark_patterns = [
            r'\bnear\s+', r'\bopp(?:osite)?\s+', r'\bbeside\s+',
            r'\badjacent\s+to\s+', r'\bnext\s+to\s+', r'\bproche\s+',
        ]
    
    def normalize_unicode(self, text: str) -> str:
        """Apply NFKC normalization and casefold."""
        if not isinstance(text, str) or not text:
            return ""
        return unicodedata.normalize('NFKC', text).casefold()
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize common punctuation variations."""
        if not text:
            return ""
        # Replace & with 'and'
        text = re.sub(r'\s*&\s*', ' and ', text)
        # Remove special punctuation but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_legal_suffix(self, business_name: str, country: str = None) -> Tuple[str, str]:
        """
        Extract legal suffix from business name.
        Returns (name_without_suffix, extracted_suffix).
        """
        if not business_name:
            return "", ""
        
        normalized = self.normalize_unicode(business_name)
        tokens = normalized.split()
        
        # Get relevant suffixes (all if country unknown)
        suffixes_to_check = []
        if country and country in self.legal_suffixes:
            suffixes_to_check = self.legal_suffixes[country]
        else:
            # Check all known suffixes
            for country_suffixes in self.legal_suffixes.values():
                suffixes_to_check.extend(country_suffixes)
        
        # Check last 1-3 tokens for legal suffixes
        for num_tokens in range(min(3, len(tokens)), 0, -1):
            last_tokens = ' '.join(tokens[-num_tokens:])
            if last_tokens in suffixes_to_check:
                name_without_suffix = ' '.join(tokens[:-num_tokens])
                return name_without_suffix, last_tokens
        
        return normalized, ""
    
    def extract_address_components(self, address: str) -> Dict[str, str]:
        """
        Extract structured components from address.
        Returns dict with: building_number, street, postal_code, has_landmark.
        """
        if not address:
            return {
                'building_number': '',
                'street': '',
                'postal_code': '',
                'has_landmark': False,
                'normalized_full': ''
            }
        
        normalized = self.normalize_unicode(address)
        
        # Check for landmark phrases
        has_landmark = any(re.search(pattern, normalized) for pattern in self.landmark_patterns)
        
        # Extract leading building/house number (1-5 digits at start)
        building_match = re.match(r'^(\d{1,5})\s+', normalized)
        building_number = building_match.group(1) if building_match else ''
        
        # Extract postal code (various formats)
        # US: 5 digits or 5-4 digits
        # India: 6 digits
        # France: 5 digits
        postal_match = re.search(r'\b(\d{5,6}(?:-\d{4})?)\b', normalized)
        postal_code = postal_match.group(1) if postal_match else ''
        
        # Normalize street types
        street = normalized
        for abbr, full in self.street_types.items():
            street = re.sub(r'\b' + abbr + r'\b', full, street)
        
        return {
            'building_number': building_number,
            'street': street,
            'postal_code': postal_code,
            'has_landmark': has_landmark,
            'normalized_full': normalized
        }
    
    def get_phonetic_code(self, text: str, algorithm: str = 'soundex') -> str:
        """
        Generate phonetic code for text.
        Uses simple soundex-like implementation.
        """
        if not text:
            return ""
        
        # Simple soundex implementation
        text = self.normalize_unicode(text)
        if not text:
            return ""
        
        # Keep first letter
        code = text[0]
        
        # Mapping of consonants to digits
        soundex_map = {
            'b': '1', 'f': '1', 'p': '1', 'v': '1',
            'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
            'd': '3', 't': '3',
            'l': '4',
            'm': '5', 'n': '5',
            'r': '6'
        }
        
        prev = soundex_map.get(text[0], '0')
        for char in text[1:]:
            digit = soundex_map.get(char, '0')
            if digit != '0' and digit != prev:
                code += digit
            prev = digit
        
        # Pad or truncate to 4 characters
        code = (code + '000')[:4]
        return code
    
    def clean_business_name(self, business_name: str, country: str = None) -> Dict[str, str]:
        """
        Comprehensive business name cleaning.
        Returns dict with multiple representations.
        """
        if not business_name:
            return {
                'original': '',
                'normalized': '',
                'without_suffix': '',
                'suffix': '',
                'tokens': [],
                'phonetic': ''
            }
        
        # Basic normalization
        normalized = self.normalize_unicode(business_name)
        normalized = self.normalize_punctuation(normalized)
        
        # Extract legal suffix
        without_suffix, suffix = self.extract_legal_suffix(normalized, country)
        
        # Tokenize
        tokens = [t for t in without_suffix.split() if len(t) > 1]
        
        # Phonetic encoding
        phonetic = self.get_phonetic_code(without_suffix)
        
        return {
            'original': business_name,
            'normalized': normalized,
            'without_suffix': without_suffix,
            'suffix': suffix,
            'tokens': tokens,
            'phonetic': phonetic
        }
    
    def clean_record(self, entity_id: str, business_name: str, 
                    business_address: str, country: str) -> Dict[str, any]:
        """
        Clean a complete entity record.
        Returns dict with all cleaned fields.
        """
        name_features = self.clean_business_name(business_name, country)
        address_features = self.extract_address_components(business_address)
        
        return {
            'entity_id': entity_id,
            'country': country if country else 'UNKNOWN',
            
            # Name fields
            'name_original': business_name,
            'name_normalized': name_features['normalized'],
            'name_without_suffix': name_features['without_suffix'],
            'name_suffix': name_features['suffix'],
            'name_tokens': name_features['tokens'],
            'name_phonetic': name_features['phonetic'],
            
            # Address fields
            'address_original': business_address,
            'address_normalized': address_features['normalized_full'],
            'address_building_number': address_features['building_number'],
            'address_postal_code': address_features['postal_code'],
            'address_has_landmark': address_features['has_landmark'],
        }


def clean_dataframe(df, cleaner: BusinessDataCleaner = None):
    """
    Apply cleaning to a pandas DataFrame with entity records.
    """
    import pandas as pd
    
    if cleaner is None:
        cleaner = BusinessDataCleaner()
    
    cleaned_records = []
    for _, row in df.iterrows():
        cleaned = cleaner.clean_record(
            entity_id=row.get('entity_id', ''),
            business_name=row.get('business_name', ''),
            business_address=row.get('business_address', ''),
            country=row.get('country', '')
        )
        cleaned_records.append(cleaned)
    
    return pd.DataFrame(cleaned_records)
