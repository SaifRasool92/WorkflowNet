# WorkflowNet: Distributed Agentic Business Process Intelligence with Skill-Gap Graphs

**WorkflowNet** is an agentic business workflow engine designed for Mitacs Globalink Research evaluation. It automates job-posting skill extraction, constructs a heterogeneous Knowledge Graph, and simulates regional/distributed skill-demand discovery.

---

## 🏗️ Architecture & Pipeline Overview

1. **Skill Extraction Engine**: Regex & NLP-backed keyword extraction mapping raw text into canonical skill nodes.
2. **Heterogeneous Knowledge Graph**: Built with NetworkX connecting `Skills <-> Roles <-> Regions`.
3. **Agentic Workflow Engine**: Knowledge Graph-guided job-to-candidate matching evaluating degree centrality and topological shortest paths.
4. **Distributed / Federated Simulation**: Regional data partitions analyze local skill hubs before aggregating bridge-skill metrics globally.

---

## 🚀 Quickstart

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline & benchmark suite
python3 main.py
```

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/aeb40495-3870-41fe-b2d4-38090b65c52c" />


<img width="900" height="700" alt="image" src="https://github.com/user-attachments/assets/2190362c-79f3-4af6-93a4-1c43e14b5579" />


<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/99c41ad0-ada7-4f2d-90db-1674006622ff" />
