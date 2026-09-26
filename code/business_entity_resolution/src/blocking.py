"""
Blocking / Candidate Generation module for Business Entity Resolution.
Implements multiple blocking strategies to create high-recall candidate sets.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class MultiBlocker:
    """
    Multi-strategy blocking for entity resolution.
    Combines multiple blocking keys to maximize recall.
    """
    
    def __init__(self, max_candidates_per_entity: int = 100):
        """
        Initialize the blocker.
        
        Args:
            max_candidates_per_entity: Maximum candidates to keep per Source 1 entity
        """
        self.max_candidates_per_entity = max_candidates_per_entity
        self.blocking_index = defaultdict(set)
        
    def _create_sorted_token_key(self, tokens: List[str]) -> str:
        """Create a blocking key from sorted tokens."""
        if not tokens:
            return ""
        # Sort tokens alphabetically and join
        return " ".join(sorted([t.lower() for t in tokens if len(t) > 1]))
    
    def _create_phonetic_key(self, phonetic_code: str) -> str:
        """Create a blocking key from phonetic code."""
        return phonetic_code if phonetic_code else ""
    
    def _create_address_number_key(self, building_num: str, postal_code: str) -> str:
        """Create a blocking key from address numeric components."""
        parts = []
        if building_num:
            parts.append(building_num)
        if postal_code:
            # Use first part of postal code (e.g., first 5 digits)
            parts.append(postal_code[:5])
        return "_".join(parts) if parts else ""
    
    def _create_country_key(self, country: str) -> str:
        """Create a blocking key from country."""
        return country.lower() if country else "unknown"
    
    def _create_first_letter_key(self, tokens: List[str]) -> str:
        """Create a blocking key from first letters of tokens."""
        if not tokens:
            return ""
        return "".join([t[0].lower() for t in tokens if len(t) > 0])[:4]
    
    def build_blocking_index(self, candidates_df: pd.DataFrame):
        """
        Build blocking index from candidate records (Source 2 and Source 3).
        
        Args:
            candidates_df: DataFrame with cleaned candidate records
        """
        self.blocking_index = defaultdict(set)
        self.candidate_records = {}
        
        print(f"Building blocking index for {len(candidates_df):,} candidates...")
        
        for idx, row in candidates_df.iterrows():
            entity_id = row['entity_id']
            self.candidate_records[entity_id] = row
            
            # Strategy 1: Sorted name tokens
            key1 = self._create_sorted_token_key(row['name_tokens'])
            if key1:
                self.blocking_index[('sorted_tokens', key1)].add(entity_id)
            
            # Strategy 2: Phonetic code
            key2 = self._create_phonetic_key(row['name_phonetic'])
            if key2:
                self.blocking_index[('phonetic', key2)].add(entity_id)
            
            # Strategy 3: Address numeric components
            key3 = self._create_address_number_key(
                row['address_building_number'], 
                row['address_postal_code']
            )
            if key3:
                self.blocking_index[('address_number', key3)].add(entity_id)
            
            # Strategy 4: Country
            key4 = self._create_country_key(row['country'])
            if key4:
                self.blocking_index[('country', key4)].add(entity_id)
            
            # Strategy 5: First letter key
            key5 = self._create_first_letter_key(row['name_tokens'])
            if key5:
                self.blocking_index[('first_letters', key5)].add(entity_id)
            
            # Strategy 6: Exact normalized name (for high precision)
            key6 = row['name_normalized']
            if key6:
                self.blocking_index[('exact_name', key6)].add(entity_id)
        
        print(f"✓ Built {len(self.blocking_index):,} blocking keys")
    
    def get_candidates(self, source1_record: pd.Series) -> Set[str]:
        """
        Get candidate entity IDs for a Source 1 record using blocking.
        
        Args:
            source1_record: Cleaned Source 1 record
            
        Returns:
            Set of candidate entity IDs
        """
        candidates = set()
        
        # Apply all blocking strategies and union results
        
        # Strategy 1: Sorted tokens
        key1 = self._create_sorted_token_key(source1_record['name_tokens'])
        if key1:
            candidates.update(self.blocking_index.get(('sorted_tokens', key1), set()))
        
        # Strategy 2: Phonetic
        key2 = self._create_phonetic_key(source1_record['name_phonetic'])
        if key2:
            candidates.update(self.blocking_index.get(('phonetic', key2), set()))
        
        # Strategy 3: Address numbers
        key3 = self._create_address_number_key(
            source1_record['address_building_number'],
            source1_record['address_postal_code']
        )
        if key3:
            candidates.update(self.blocking_index.get(('address_number', key3), set()))
        
        # Strategy 4: Country (always include same country)
        key4 = self._create_country_key(source1_record['country'])
        if key4:
            candidates.update(self.blocking_index.get(('country', key4), set()))
        
        # Strategy 5: First letters
        key5 = self._create_first_letter_key(source1_record['name_tokens'])
        if key5:
            candidates.update(self.blocking_index.get(('first_letters', key5), set()))
        
        # Strategy 6: Exact name
        key6 = source1_record['name_normalized']
        if key6:
            candidates.update(self.blocking_index.get(('exact_name', key6), set()))
        
        # Limit candidates if too many
        if len(candidates) > self.max_candidates_per_entity:
            # Score candidates by simple similarity and keep top K
            scored_candidates = []
            for cand_id in candidates:
                score = self._quick_similarity_score(source1_record, self.candidate_records[cand_id])
                scored_candidates.append((score, cand_id))
            
            scored_candidates.sort(reverse=True)
            candidates = set([cid for _, cid in scored_candidates[:self.max_candidates_per_entity]])
        
        return candidates
    
    def _quick_similarity_score(self, s1_record: pd.Series, cand_record: pd.Series) -> float:
        """
        Quick similarity score for ranking candidates.
        Used for pruning when there are too many candidates.
        """
        score = 0.0
        
        # Exact name match
        if s1_record['name_normalized'] == cand_record['name_normalized']:
            score += 10.0
        
        # Token overlap (Jaccard)
        s1_tokens = set(s1_record['name_tokens'])
        cand_tokens = set(cand_record['name_tokens'])
        if s1_tokens and cand_tokens:
            jaccard = len(s1_tokens & cand_tokens) / len(s1_tokens | cand_tokens)
            score += jaccard * 5.0
        
        # Same country
        if s1_record['country'] == cand_record['country']:
            score += 2.0
        
        # Postal code match
        if (s1_record['address_postal_code'] and cand_record['address_postal_code'] and
            s1_record['address_postal_code'] == cand_record['address_postal_code']):
            score += 3.0
        
        # Phonetic match
        if (s1_record['name_phonetic'] and cand_record['name_phonetic'] and
            s1_record['name_phonetic'] == cand_record['name_phonetic']):
            score += 1.0
        
        return score
    
    def generate_candidate_pairs(self, source1_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate candidate pairs for all Source 1 entities.
        
        Args:
            source1_df: DataFrame with cleaned Source 1 records
            
        Returns:
            DataFrame with columns: source1_entity_id, candidate_entity_ids
        """
        print(f"\\nGenerating candidates for {len(source1_df):,} Source 1 entities...")
        
        results = []
        total_candidates = 0
        singleton_count = 0
        
        for idx, row in source1_df.iterrows():
            s1_id = row['entity_id']
            candidates = self.get_candidates(row)
            
            # Remove Source 1 IDs if any accidentally got in
            candidates = {c for c in candidates if not c.startswith('S1-')}
            
            candidate_list = sorted(list(candidates))
            total_candidates += len(candidate_list)
            
            if len(candidate_list) == 0:
                singleton_count += 1
            
            results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ','.join(candidate_list) if candidate_list else ''
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1:,} / {len(source1_df):,} entities...")
        
        print(f"\\n✓ Candidate generation complete:")
        print(f"  Total S1 entities: {len(source1_df):,}")
        print(f"  Entities with 0 candidates: {singleton_count:,} ({100*singleton_count/len(source1_df):.1f}%)") 
        print(f"  Average candidates per entity: {total_candidates/len(source1_df):.1f}")
        print(f"  Total candidate pairs: {total_candidates:,}")
        
        return pd.DataFrame(results)


def calculate_candidate_recall(candidate_pairs_df: pd.DataFrame, 
                               ground_truth_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate recall of the blocking stage.
    
    Args:
        candidate_pairs_df: DataFrame with candidate pairs
        ground_truth_df: DataFrame with ground truth matches
        
    Returns:
        Dictionary with recall metrics
    """
    # Parse ground truth
    def parse_matches(match_str):
        if pd.isna(match_str) or str(match_str).strip() == '':
            return set()
        return set([s.strip() for s in str(match_str).split(',') if s.strip()])
    
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        gt_dict[row['source1_entity_id']] = parse_matches(row['matched_entity_ids'])
    
    # Parse candidate pairs
    cand_dict = {}
    for _, row in candidate_pairs_df.iterrows():
        cand_dict[row['source1_entity_id']] = parse_matches(row['candidate_entity_ids'])
    
    # Calculate recall
    total_true_matches = 0
    found_matches = 0
    entities_with_matches = 0
    entities_found_all = 0
    
    for s1_id, true_matches in gt_dict.items():
        if len(true_matches) > 0:
            entities_with_matches += 1
            total_true_matches += len(true_matches)
            
            candidates = cand_dict.get(s1_id, set())
            found = len(true_matches & candidates)
            found_matches += found
            
            if found == len(true_matches):
                entities_found_all += 1
    
    recall = found_matches / total_true_matches if total_true_matches > 0 else 0.0
    entity_recall = entities_found_all / entities_with_matches if entities_with_matches > 0 else 0.0
    
    return {
        'pair_recall': recall,
        'entity_recall': entity_recall,
        'total_true_matches': total_true_matches,
        'found_matches': found_matches,
        'entities_with_matches': entities_with_matches,
        'entities_found_all': entities_found_all
    }
