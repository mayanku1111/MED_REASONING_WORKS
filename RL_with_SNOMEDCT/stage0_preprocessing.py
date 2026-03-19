
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PreprocessedSentence:
    """Structured sentence with metadata"""
    text: str
    index: int
    is_negated: bool = False
    temporal_info: str = None
    sentence_type: str = "statement"  # statement, negation, temporal

class ClinicalPreprocessor:
    """Stage 0: Cheap deterministic preprocessing"""
    
    # Negation triggers
    NEGATION_PATTERNS = [
        r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bdenies\b',
        r'\bwithout\b', r'\babsent\b', r'\bnegative for\b'
    ]
    
    # Temporal patterns
    TEMPORAL_PATTERNS = {
        'acute': r'\b(\d+\s*(days?|hours?|weeks?))\b',
        'chronic': r'\b(months?|years?)\b',
        'recent': r'\b(recently|lately|just started)\b',
        'past': r'\b(history of|previous|prior)\b'
    }
    
    def __init__(self):
        self.negation_regex = re.compile('|'.join(self.NEGATION_PATTERNS), re.IGNORECASE)
    
    def split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Split on periods, but preserve decimal numbers
        sentences = re.split(r'(?<!\d)\.(?!\d)\s+', text)
        # Also split on newlines
        result = []
        for s in sentences:
            result.extend([x.strip() for x in s.split('\n') if x.strip()])
        return [s for s in result if s]
    
    def detect_negation(self, sentence: str) -> bool:
        """Check if sentence contains negation"""
        return bool(self.negation_regex.search(sentence))
    
    def extract_temporal(self, sentence: str) -> Tuple[str, str]:
        """Extract temporal information"""
        for temp_type, pattern in self.TEMPORAL_PATTERNS.items():
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return temp_type, match.group(0)
        return None, None
    
    def preprocess(self, clinical_text: str) -> List[PreprocessedSentence]:
        """Main preprocessing pipeline"""
        sentences = self.split_sentences(clinical_text)
        processed = []
        
        for idx, sentence in enumerate(sentences):
            is_negated = self.detect_negation(sentence)
            temp_type, temp_value = self.extract_temporal(sentence)
            
            # Determine sentence type
            if is_negated:
                sent_type = "negation"
            elif temp_type:
                sent_type = "temporal"
            else:
                sent_type = "statement"
            
            processed.append(PreprocessedSentence(
                text=sentence,
                index=idx,
                is_negated=is_negated,
                temporal_info=f"{temp_type}: {temp_value}" if temp_type else None,
                sentence_type=sent_type
            ))
        
        return processed

def test_preprocessing():
    """Test the preprocessor"""
    preprocessor = ClinicalPreprocessor()
    
    test_text = """
    Patient presents with tight chest pain for 2 days.
    Pain worsens when walking upstairs.
    Radiates to left arm.
    No fever.
    No cough.
    History of heartburn.
    """
    
    results = preprocessor.preprocess(test_text)
    
    print("=== PREPROCESSING OUTPUT ===\n")
    for s in results:
        print(f"S{s.index}: {s.text}")
        print(f"   Type: {s.sentence_type}")
        if s.is_negated:
            print(f"   ⚠ NEGATED")
        if s.temporal_info:
            print(f"   ⏰ {s.temporal_info}")
        print()

if __name__ == "__main__":
    test_preprocessing()
