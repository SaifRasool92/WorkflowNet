import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.skill_extractor import SkillExtractor
from src.graph_builder import SkillGraphBuilder
from src.workflow_agent import WorkflowMatchingAgent
from src.metrics import WorkflowPerformanceTracker

def generate_synthetic_data(n_samples: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    regions = ['North America', 'Europe', 'Asia-Pacific']
    seniorities = ['Junior', 'Mid-Level', 'Senior', 'Lead']
    
    # Core core pool per role + shared noisy hub skills
    roles_skills_pool = {
        'Machine Learning Engineer': ['Python', 'PyTorch', 'Docker/Kubernetes', 'Cloud (AWS/GCP/Azure)', 'Git/CI-CD', 'Deep Learning', 'MLOps'],
        'Data Scientist': ['Python', 'R', 'SQL', 'Scikit-Learn', 'Tableau/PowerBI', 'Machine Learning', 'Data Modeling'],
        'Data Engineer': ['SQL', 'PySpark/Spark', 'ETL Pipelines', 'NoSQL', 'Cloud (AWS/GCP/Azure)', 'Docker/Kubernetes', 'Python'],
        'Cloud Solution Architect': ['Cloud (AWS/GCP/Azure)', 'Docker/Kubernetes', 'Git/CI-CD', 'Go', 'Linux', 'Python'],
        'AI Research Scientist': ['Python', 'PyTorch', 'TensorFlow', 'LLMs/NLP', 'Deep Learning', 'Mathematics'],
        'BI Developer': ['SQL', 'Tableau/PowerBI', 'Data Modeling', 'ETL Pipelines', 'Excel', 'Python']
    }
    
    roles = list(roles_skills_pool.keys())
    data = []

    for i in range(1, n_samples + 1):
        role = np.random.choice(roles)
        region = np.random.choice(regions, p=[0.45, 0.35, 0.20])
        seniority = np.random.choice(seniorities)
        
        # Introduce intra-role variance: sample 4 to 6 skills out of the pool
        all_pool = roles_skills_pool[role]
        n_skills = np.random.randint(4, len(all_pool) + 1)
        sampled_skills = list(np.random.choice(all_pool, size=n_skills, replace=False))
        
        skills_text = ", ".join(sampled_skills)
        
        data.append({
            'posting_id': f"JOB-{i:04d}",
            'job_title': role,
            'region': region,
            'seniority_level': seniority,
            'required_skills_text': f"Seeking {seniority} {role} proficient in {skills_text}.",
            'ground_truth_skills': skills_text
        })

    df = pd.DataFrame(data)
    os.makedirs('data/regional_splits', exist_ok=True)
    df.to_csv('data/job_postings.csv', index=False)
    for reg in regions:
        df[df['region'] == reg].to_csv(f"data/regional_splits/{reg.lower().replace(' ', '_')}.csv", index=False)
    return df

def run_comprehensive_evaluation(agent, df):
    # Candidates with shared hub skills (Python, SQL, Cloud) that confuse naive overlap counters
    candidates = [
        {'skills': ['Python', 'SQL', 'Cloud (AWS/GCP/Azure)', 'MLOps'], 'exact_role': 'Machine Learning Engineer'},
        {'skills': ['SQL', 'Python', 'Excel', 'Tableau/PowerBI'], 'exact_role': 'BI Developer'},
        {'skills': ['SQL', 'Python', 'Cloud (AWS/GCP/Azure)', 'PySpark/Spark'], 'exact_role': 'Data Engineer'},
        {'skills': ['Python', 'SQL', 'LLMs/NLP', 'Mathematics'], 'exact_role': 'AI Research Scientist'},
        {'skills': ['Cloud (AWS/GCP/Azure)', 'Python', 'Docker/Kubernetes', 'Go'], 'exact_role': 'Cloud Solution Architect'},
        {'skills': ['Python', 'SQL', 'Scikit-Learn', 'R'], 'exact_role': 'Data Scientist'},
        {'skills': ['Python', 'Cloud (AWS/GCP/Azure)', 'Deep Learning', 'PyTorch'], 'exact_role': 'Machine Learning Engineer'},
        {'skills': ['SQL', 'Python', 'Data Modeling', 'Excel'], 'exact_role': 'BI Developer'},
        {'skills': ['SQL', 'Cloud (AWS/GCP/Azure)', 'ETL Pipelines', 'NoSQL'], 'exact_role': 'Data Engineer'},
        {'skills': ['Python', 'SQL', 'TensorFlow', 'LLMs/NLP'], 'exact_role': 'AI Research Scientist'},
        {'skills': ['Cloud (AWS/GCP/Azure)', 'Docker/Kubernetes', 'Linux', 'Git/CI-CD'], 'exact_role': 'Cloud Solution Architect'},
        {'skills': ['Python', 'SQL', 'Machine Learning', 'Scikit-Learn'], 'exact_role': 'Data Scientist'},
        {'skills': ['Python', 'Docker/Kubernetes', 'MLOps', 'Git/CI-CD'], 'exact_role': 'Machine Learning Engineer'},
        {'skills': ['SQL', 'Python', 'Tableau/PowerBI', 'Data Modeling'], 'exact_role': 'BI Developer'},
        {'skills': ['SQL', 'Cloud (AWS/GCP/Azure)', 'Docker/Kubernetes', 'PySpark/Spark'], 'exact_role': 'Data Engineer'},
        {'skills': ['Python', 'Deep Learning', 'Mathematics', 'PyTorch'], 'exact_role': 'AI Research Scientist'},
        {'skills': ['Cloud (AWS/GCP/Azure)', 'Python', 'Go', 'Linux'], 'exact_role': 'Cloud Solution Architect'},
        {'skills': ['Python', 'SQL', 'Data Modeling', 'R'], 'exact_role': 'Data Scientist'},
        {'skills': ['Python', 'Cloud (AWS/GCP/Azure)', 'MLOps', 'Deep Learning'], 'exact_role': 'Machine Learning Engineer'},
        {'skills': ['SQL', 'Python', 'ETL Pipelines', 'Excel'], 'exact_role': 'BI Developer'}
    ]

    graph_precisions, base_precisions = [], []

    for cand in candidates:
        exact_role = cand['exact_role']
        target_ids = set(df[df['job_title'] == exact_role]['posting_id'])
        
        g_matches = agent.match_candidate_graph(cand['skills'], top_k=5)
        b_matches = agent.match_candidate_baseline(cand['skills'], top_k=5)

        g_prec = len([m for m in g_matches if m['posting_id'] in target_ids]) / 5.0
        b_prec = len([m for m in b_matches if m['posting_id'] in target_ids]) / 5.0

        graph_precisions.append(g_prec)
        base_precisions.append(b_prec)

    return float(np.mean(graph_precisions)), float(np.mean(base_precisions))

def main():
    print("....WorkflowNet Benchmark Engine Initializing....\n")
    
    # 1. Dataset & Skill Extraction
    df = generate_synthetic_data(250)
    extractor = SkillExtractor()
    df['extracted_skills'] = df['required_skills_text'].apply(extractor.extract_skills)
    
    acc = extractor.evaluate_extraction_accuracy(df)
    print(f"[Part 1 - Extraction] Processed {len(df)} postings | Extraction Accuracy: {acc*100:.2f}%")

    # 2. Knowledge Graph Construction
    builder = SkillGraphBuilder()
    graph = builder.build_graph(df)
    os.makedirs('results/charts', exist_ok=True)
    builder.export_graph_json('results/skill_graph.json')
    
    bridge_df = builder.compute_bridge_skills(top_n=5)
    print("\n[Part 2 - Knowledge Graph] Top 5 Bridge Skills:")
    print(bridge_df[['skill', 'degree_centrality', 'betweenness_centrality', 'composite_bridge_score']].to_string(index=False))

    # Save Charts
    plt.figure(figsize=(8, 4))
    plt.barh(bridge_df['skill'][::-1], bridge_df['composite_bridge_score'][::-1], color='#1f77b4')
    plt.xlabel('Composite Centrality Score')
    plt.title('Top 5 Bridge Skills (Transferability Index)')
    plt.tight_layout()
    plt.savefig('results/charts/bridge_skills.png')
    plt.close()

    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=42, k=0.35)
    node_colors = ['#2ca02c' if attr.get('node_type') == 'Skill' else '#1f77b4' if attr.get('node_type') == 'Role' else '#ff7f0e' for _, attr in graph.nodes(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=250, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, alpha=0.15, edge_color='gray')
    labels = {node: node for node in graph.nodes() if node in set(bridge_df['skill']) or graph.nodes[node].get('node_type') == 'Region'}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.title('WorkflowNet Skill-Role-Region Knowledge Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/charts/skill_gap_network.png')
    plt.close()

    # 3. Benchmark Matching Engine
    agent = WorkflowMatchingAgent(graph, df)
    
    graph_prec, base_prec = run_comprehensive_evaluation(agent, df)
    
    sample_skills = ['Python', 'PyTorch', 'MLOps', 'Deep Learning']
    graph_perf = WorkflowPerformanceTracker.measure_latency_and_throughput(lambda s: agent.match_candidate_graph(s), sample_skills)
    base_perf = WorkflowPerformanceTracker.measure_latency_and_throughput(lambda s: agent.match_candidate_baseline(s), sample_skills)

    print("\n" + "="*65)
    print("BENCHMARK RESULTS: GRAPH AGENT vs NAIVE BASELINE (20-Profile Suite)")
    print("="*65)
    print(f"Graph Agent   -> Precision@5: {graph_prec*100:.1f}% | Latency: {graph_perf['avg_latency_ms']:.2f} ms | Throughput: {graph_perf['throughput_ops_sec']:.1f} ops/sec")
    print(f"Baseline Naive-> Precision@5: {base_prec*100:.1f}% | Latency: {base_perf['avg_latency_ms']:.2f} ms | Throughput: {base_perf['throughput_ops_sec']:.1f} ops/sec")
    print("="*65 + "\n")

    # Metrics Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    methods = ['Graph Agent', 'Naive Baseline']
    ax1.bar(methods, [graph_prec * 100, base_prec * 100], color=['#2ca02c', '#d62728'])
    ax1.set_ylabel('Precision@5 (%)')
    ax1.set_title('Match Precision (Quality)')

    ax2.bar(methods, [graph_perf['avg_latency_ms'], base_perf['avg_latency_ms']], color=['#1f77b4', '#ff7f0e'])
    ax2.set_ylabel('Avg Latency (ms)')
    ax2.set_title('Execution Latency (ms)')

    plt.tight_layout()
    plt.savefig('results/charts/workflow_efficiency.png')
    plt.close()

    # 4. Regional Breakdown Analysis
    print("[Part 4 - Regional Analysis] Computing Regional Bridge Skills...")
    for reg in ['Asia-Pacific', 'North America', 'Europe']:
        reg_df = df[df['region'] == reg]
        reg_builder = SkillGraphBuilder()
        reg_graph = reg_builder.build_graph(reg_df)
        top_skills = reg_builder.compute_bridge_skills(top_n=3)['skill'].tolist()
        print(f"  -> {reg} Top Bridge Skills: {', '.join(top_skills)}")

    # Export Notebook
    try:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        with open('main.py') as f:
            nb['cells'] = [
                nbf.v4.new_markdown_cell('# WorkflowNet: Distributed Agentic Business Process Intelligence'),
                nbf.v4.new_code_cell(f.read())
            ]
        with open('workflownet_notebook.ipynb', 'w') as f:
            nbf.write(nb, f)
    except Exception as e:
        pass

    print("\nAll artifacts generated and saved to results/")

if __name__ == "__main__":
    main()
