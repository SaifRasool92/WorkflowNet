import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict

class WorkflowMatchingAgent:
    """Graph-Augmented Matching Agent with topological path distance and IDF weighting."""

    def __init__(self, graph: nx.Graph, job_postings_df: pd.DataFrame):
        self.graph = graph
        self.df = job_postings_df.copy()
        
        # Precalculate degree centralities and shortest paths
        self.degree_cent = nx.degree_centrality(self.graph)
        self.path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
        
        # Pre-index job postings
        self.records = []
        for idx, row in self.df.iterrows():
            skills = set(row['extracted_skills'] if isinstance(row['extracted_skills'], list) else str(row['extracted_skills']).split(','))
            self.records.append({
                'posting_id': row['posting_id'],
                'job_title': row['job_title'],
                'region': row['region'],
                'skill_set': skills,
            })

    def match_candidate_graph(self, candidate_skills: List[str], target_region: str = None, top_k: int = 5) -> List[Dict]:
        """Graph match using shortest path topological distance & inverse skill degree."""
        candidate_set = set(candidate_skills)
        if not candidate_set:
            return []

        scores = []
        for rec in self.records:
            if target_region and rec['region'] != target_region:
                continue

            job_skills = rec['skill_set']
            intersection = candidate_set.intersection(job_skills)
            if not intersection:
                continue

            # Topological path distance from candidate skills to target role node
            role_node = rec['job_title']
            proximity_score = 0.0
            if role_node in self.path_lengths:
                role_paths = self.path_lengths[role_node]
                dists = [role_paths[s] for s in candidate_skills if s in role_paths]
                if dists:
                    proximity_score = 1.0 / (np.mean(dists) + 1.0)

            # Specificity Weighting: Downweight high-degree hub skills (Python/SQL)
            spec_weight = sum(1.0 / (1.0 + self.degree_cent.get(s, 0.5)) for s in intersection)

            final_score = (0.65 * proximity_score) + (0.35 * spec_weight)

            scores.append({
                'posting_id': rec['posting_id'],
                'job_title': rec['job_title'],
                'region': rec['region'],
                'score': float(final_score)
            })

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def match_candidate_baseline(self, candidate_skills: List[str], target_region: str = None, top_k: int = 5) -> List[Dict]:
        """Naive unweighted Jaccard similarity baseline."""
        candidate_set = set(candidate_skills)
        if not candidate_set:
            return []

        scores = []
        for rec in self.records:
            if target_region and rec['region'] != target_region:
                continue

            job_skills = rec['skill_set']
            intersection = candidate_set.intersection(job_skills)
            union = candidate_set.union(job_skills)
            
            jaccard = len(intersection) / float(len(union)) if union else 0.0

            scores.append({
                'posting_id': rec['posting_id'],
                'job_title': rec['job_title'],
                'region': rec['region'],
                'score': float(jaccard)
            })

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
