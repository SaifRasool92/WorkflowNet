import re
import numpy as np
import pandas as pd
from typing import List, Dict, Set

SKILL_SYNONYM_MAP: Dict[str, str] = {
    "python": "Python",
    "r": "R",
    "sql": "SQL",
    "nosql": "NoSQL",
    "pyspark/spark": "PySpark/Spark",
    "pyspark": "PySpark/Spark",
    "spark": "PySpark/Spark",
    "cloud (aws/gcp/azure)": "Cloud (AWS/GCP/Azure)",
    "aws": "Cloud (AWS/GCP/Azure)",
    "gcp": "Cloud (AWS/GCP/Azure)",
    "azure": "Cloud (AWS/GCP/Azure)",
    "docker/kubernetes": "Docker/Kubernetes",
    "docker": "Docker/Kubernetes",
    "kubernetes": "Docker/Kubernetes",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-Learn",
    "tableau/powerbi": "Tableau/PowerBI",
    "tableau": "Tableau/PowerBI",
    "powerbi": "Tableau/PowerBI",
    "git/ci-cd": "Git/CI-CD",
    "git": "Git/CI-CD",
    "ci/cd": "Git/CI-CD",
    "llms/nlp": "LLMs/NLP",
    "llms": "LLMs/NLP",
    "nlp": "LLMs/NLP",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "data modeling": "Data Modeling",
    "etl pipelines": "ETL Pipelines",
    "excel": "Excel",
    "go": "Go",
    "linux": "Linux",
    "mlops": "MLOps",
    "mathematics": "Mathematics"
}

class SkillExtractor:
    """Normalized, high-precision skill extraction engine."""
    def __init__(self, synonym_map: Dict[str, str] = None):
        self.synonym_map = synonym_map or SKILL_SYNONYM_MAP
        sorted_terms = sorted(self.synonym_map.keys(), key=lambda x: len(x), reverse=True)
        patterns = [r'(?<![a-zA-Z0-9])' + re.escape(term) + r'(?![a-zA-Z0-9])' for term in sorted_terms]
        self.regex = re.compile(r'(' + '|'.join(patterns) + r')', re.IGNORECASE)

    def extract_skills(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        matches = self.regex.findall(text.lower())
        extracted: Set[str] = set()
        for match in matches:
            canonical = self.synonym_map.get(match.lower())
            if canonical:
                extracted.add(canonical)
        return sorted(list(extracted))

    def evaluate_extraction_accuracy(self, postings_df: pd.DataFrame, ground_truth_col: str = 'ground_truth_skills') -> float:
        if ground_truth_col not in postings_df.columns:
            return 1.0
        jaccard_scores = []
        for _, row in postings_df.iterrows():
            extracted = set(self.extract_skills(row['required_skills_text']))
            actual = set([s.strip() for s in row[ground_truth_col].split(',') if s.strip()])
            if actual:
                intersection = len(extracted.intersection(actual))
                union = len(extracted.union(actual))
                jaccard_scores.append(intersection / union)
        return float(np.mean(jaccard_scores)) if jaccard_scores else 1.0
