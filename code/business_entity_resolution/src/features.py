"""
Feature engineering module for entity matching.
Generates pairwise similarity features for candidate pairs.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class FeatureExtractor:
    """
    Extract pairwise features for entity matching.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.tfidf_name = None
        self.tfidf_address = None
        
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _token_sort_ratio(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate token sort ratio similarity."""
        str1 = ' '.join(sorted(tokens1))
        str2 = ' '.join(sorted(tokens2))
        return fuzz.ratio(str1, str2) / 100.0
    
    def _levenshtein_similarity(self, str1: str, str2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        if not str1 or not str2:
            return 0.0
        return fuzz.ratio(str1, str2) / 100.0
    
    def _jaro_winkler_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaro-Winkler similarity."""
        if not str1 or not str2:
            return 0.0
        return fuzz.WRatio(str1, str2) / 100.0
    
    def _monge_elkan_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate Monge-Elkan similarity.
        For each token in set1, find best match in set2, then average.
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        total_sim = 0.0
        for t1 in tokens1:
            max_sim = max([fuzz.ratio(t1, t2) / 100.0 for t2 in tokens2])
            total_sim += max_sim
        
        return total_sim / len(tokens1)
    
    def _weighted_jaccard(self, tokens1: List[str], tokens2: List[str], idf_weights: Dict[str, float]) -> float:
        """Calculate IDF-weighted Jaccard similarity."""
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection_weight = sum([idf_weights.get(t, 1.0) for t in set1 & set2])
        union_weight = sum([idf_weights.get(t, 1.0) for t in set1 | set2])
        
        return intersection_weight / union_weight if union_weight > 0 else 0.0
    
    def extract_name_features(self, s1_record: pd.Series, cand_record: pd.Series, 
                             idf_weights: Dict[str, float] = None) -> Dict[str, float]:
        """Extract name similarity features."""
        features = {}
        
        # Exact match features
        features['name_exact_match'] = float(s1_record['name_normalized'] == cand_record['name_normalized'])
        features['name_exact_match_no_suffix'] = float(s1_record['name_without_suffix'] == cand_record['name_without_suffix'])
        features['suffix_match'] = float(s1_record['name_suffix'] == cand_record['name_suffix'])
        features['suffix_both_present'] = float(bool(s1_record['name_suffix']) and bool(cand_record['name_suffix']))
        
        # Token-based features
        tokens1 = set(s1_record['name_tokens'])
        tokens2 = set(cand_record['name_tokens'])
        
        features['name_token_jaccard'] = self._jaccard_similarity(tokens1, tokens2)
        features['name_token_sort_ratio'] = self._token_sort_ratio(s1_record['name_tokens'], cand_record['name_tokens'])
        features['name_token_count_diff'] = abs(len(tokens1) - len(tokens2))
        features['name_token_count_ratio'] = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0.0
        
        # String similarity features
        features['name_levenshtein'] = self._levenshtein_similarity(s1_record['name_normalized'], cand_record['name_normalized'])
        features['name_jaro_winkler'] = self._jaro_winkler_similarity(s1_record['name_normalized'], cand_record['name_normalized'])
        features['name_no_suffix_levenshtein'] = self._levenshtein_similarity(s1_record['name_without_suffix'], cand_record['name_without_suffix'])
        
        # Monge-Elkan (token-level best match)
        if s1_record['name_tokens'] and cand_record['name_tokens']:
            features['name_monge_elkan'] = self._monge_elkan_similarity(s1_record['name_tokens'], cand_record['name_tokens'])
        else:
            features['name_monge_elkan'] = 0.0
        
        # IDF-weighted similarity
        if idf_weights:
            features['name_weighted_jaccard'] = self._weighted_jaccard(s1_record['name_tokens'], cand_record['name_tokens'], idf_weights)
        else:
            features['name_weighted_jaccard'] = 0.0
        
        # Phonetic match
        features['phonetic_match'] = float(s1_record['name_phonetic'] == cand_record['name_phonetic'])
        
        # First token match (company name core)
        if s1_record['name_tokens'] and cand_record['name_tokens']:
            features['first_token_match'] = float(s1_record['name_tokens'][0] == cand_record['name_tokens'][0])
        else:
            features['first_token_match'] = 0.0
        
        return features
    
    def extract_address_features(self, s1_record: pd.Series, cand_record: pd.Series) -> Dict[str, float]:
        """Extract address similarity features."""
        features = {}
        
        # Exact component matches
        features['address_building_match'] = float(
            s1_record['address_building_number'] and 
            cand_record['address_building_number'] and
            s1_record['address_building_number'] == cand_record['address_building_number']
        )
        
        features['address_postal_match'] = float(
            s1_record['address_postal_code'] and 
            cand_record['address_postal_code'] and
            s1_record['address_postal_code'] == cand_record['address_postal_code']
        )
        
        # Postal code partial match (first 3-5 digits)
        if s1_record['address_postal_code'] and cand_record['address_postal_code']:
            postal1 = s1_record['address_postal_code'][:5]
            postal2 = cand_record['address_postal_code'][:5]
            features['address_postal_partial_match'] = float(postal1 == postal2)
        else:
            features['address_postal_partial_match'] = 0.0
        
        # Landmark flags
        features['address_has_landmark_s1'] = float(s1_record['address_has_landmark'])
        features['address_has_landmark_cand'] = float(cand_record['address_has_landmark'])
        features['address_both_have_landmark'] = float(s1_record['address_has_landmark'] and cand_record['address_has_landmark'])
        
        # Token-based address similarity
        addr1_tokens = set(s1_record['address_normalized'].split())
        addr2_tokens = set(cand_record['address_normalized'].split())
        
        features['address_token_jaccard'] = self._jaccard_similarity(addr1_tokens, addr2_tokens)
        
        # String similarity on normalized address
        features['address_levenshtein'] = self._levenshtein_similarity(
            s1_record['address_normalized'], 
            cand_record['address_normalized']
        )
        
        features['address_jaro_winkler'] = self._jaro_winkler_similarity(
            s1_record['address_normalized'],
            cand_record['address_normalized']
        )
        
        return features
    
    def extract_country_features(self, s1_record: pd.Series, cand_record: pd.Series) -> Dict[str, float]:
        """Extract country-related features."""
        features = {}
        
        features['country_match'] = float(s1_record['country'] == cand_record['country'])
        features['country_is_us'] = float(s1_record['country'] == 'US')
        features['country_is_india'] = float(s1_record['country'] == 'India')
        features['country_is_other'] = float(s1_record['country'] not in ['US', 'India'])
        
        return features
    
    def extract_pair_features(self, s1_record: pd.Series, cand_record: pd.Series,
                            idf_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Extract all features for a candidate pair.
        
        Args:
            s1_record: Source 1 cleaned record
            cand_record: Candidate cleaned record
            idf_weights: Optional IDF weights for tokens
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Name features
        features.update(self.extract_name_features(s1_record, cand_record, idf_weights))
        
        # Address features
        features.update(self.extract_address_features(s1_record, cand_record))
        
        # Country features
        features.update(self.extract_country_features(s1_record, cand_record))
        
        # Combined features
        features['name_and_address_match'] = features['name_exact_match'] * features['address_postal_match']
        features['high_name_sim'] = float(features['name_token_jaccard'] > 0.8)
        features['high_address_sim'] = float(features['address_token_jaccard'] > 0.8)
        
        return features
    
    def compute_idf_weights(self, all_records_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute IDF weights for tokens across all records.
        
        Args:
            all_records_df: DataFrame with all cleaned records (S1, S2, S3)
            
        Returns:
            Dictionary mapping tokens to IDF weights
        """
        # Count document frequency
        token_doc_freq = Counter()
        total_docs = len(all_records_df)
        
        for tokens in all_records_df['name_tokens']:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                token_doc_freq[token] += 1
        
        # Compute IDF
        idf_weights = {}
        for token, freq in token_doc_freq.items():
            idf_weights[token] = np.log(total_docs / (1 + freq))
        
        return idf_weights
    
    def create_feature_matrix(self, candidate_pairs_df: pd.DataFrame,
                            source1_df: pd.DataFrame, 
                            candidates_df: pd.DataFrame,
                            idf_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Create feature matrix for all candidate pairs.
        
        Args:
            candidate_pairs_df: DataFrame with candidate pairs
            source1_df: DataFrame with Source 1 cleaned records
            candidates_df: DataFrame with candidate cleaned records
            idf_weights: Optional IDF weights
            
        Returns:
            DataFrame with features for each pair
        """
        # Create lookup dictionaries
        s1_dict = {row['entity_id']: row for _, row in source1_df.iterrows()}
        cand_dict = {row['entity_id']: row for _, row in candidates_df.iterrows()}
        
        print(f"\\nExtracting features for candidate pairs...")
        
        all_features = []
        pair_count = 0
        
        for idx, row in candidate_pairs_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_ids_str = row['candidate_entity_ids']
            
            if not cand_ids_str or pd.isna(cand_ids_str):
                continue
            
            cand_ids = [c.strip() for c in str(cand_ids_str).split(',') if c.strip()]
            s1_record = s1_dict.get(s1_id)
            
            if s1_record is None:
                continue
            
            for cand_id in cand_ids:
                cand_record = cand_dict.get(cand_id)
                if cand_record is None:
                    continue
                
                # Extract features
                features = self.extract_pair_features(s1_record, cand_record, idf_weights)
                features['source1_entity_id'] = s1_id
                features['candidate_entity_id'] = cand_id
                
                all_features.append(features)
                pair_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1:,} / {len(candidate_pairs_df):,} S1 entities...")
        
        print(f"\\n✓ Feature extraction complete: {pair_count:,} pairs")
        
        return pd.DataFrame(all_features)
