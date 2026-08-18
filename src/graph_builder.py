import networkx as nx
import pandas as pd
import json
from typing import Dict, List, Tuple

class SkillGraphBuilder:
    """Builds heterogeneous bipartite/tripartite Skill <-> Role <-> Region Knowledge Graphs."""
    
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Constructs graph nodes and weighted edges from job posting dataset."""
        self.graph.clear()
        
        for _, row in df.iterrows():
            posting_id = row['posting_id']
            role = row['job_title']
            region = row['region']
            skills = row['extracted_skills'] if isinstance(row['extracted_skills'], list) else row['extracted_skills'].split(',')

            # Add Nodes with Node Types
            self.graph.add_node(role, node_type='Role')
            self.graph.add_node(region, node_type='Region')
            
            # Connect Role to Region
            if self.graph.has_edge(role, region):
                self.graph[role][region]['weight'] += 1
            else:
                self.graph.add_edge(role, region, weight=1)

            # Connect Skills to Role and Region
            for skill in skills:
                skill = skill.strip()
                if not skill:
                    continue
                self.graph.add_node(skill, node_type='Skill')
                
                # Skill <-> Role Edge
                if self.graph.has_edge(skill, role):
                    self.graph[skill][role]['weight'] += 1
                else:
                    self.graph.add_edge(skill, role, weight=1)

                # Skill <-> Region Edge
                if self.graph.has_edge(skill, region):
                    self.graph[skill][region]['weight'] += 1
                else:
                    self.graph.add_edge(skill, region, weight=1)

        return self.graph

    def compute_bridge_skills(self, top_n: int = 5) -> pd.DataFrame:
        """Calculates degree and betweenness centrality to identify high-transferability Bridge Skills."""
        skill_nodes = [node for node, attr in self.graph.nodes(data=True) if attr.get('node_type') == 'Skill']
        
        degree_cent = nx.degree_centrality(self.graph)
        between_cent = nx.betweenness_centrality(self.graph)
        
        results = []
        for skill in skill_nodes:
            results.append({
                'skill': skill,
                'degree_centrality': degree_cent[skill],
                'betweenness_centrality': between_cent[skill],
                'composite_bridge_score': (degree_cent[skill] + 2 * between_cent[skill]) / 3
            })
            
        res_df = pd.DataFrame(results).sort_values(by='composite_bridge_score', ascending=False)
        return res_df.head(top_n)

    def export_graph_json(self, filepath: str):
        """Exports graph architecture to JSON."""
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)