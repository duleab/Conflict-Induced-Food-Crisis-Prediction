import json
import os
from pathlib import Path

BASE_DIR = Path(r"D:\Project\10academy\Conflict-Induced Food Crisis Prediction")
NB_DIR = BASE_DIR / "notebooks"

notebooks_to_process = [
    (NB_DIR / "Crisis_task1_data_collection.ipynb", "Task 1: Data Collection & Raw Pipeline"),
    (NB_DIR / "crisis_task2_feature_engineering.ipynb", "Task 2: Feature Engineering & Splits"),
    (NB_DIR / "crisis_task3_model_training.ipynb", "Task 3: Baseline Models & XGBoost Tuning"),
    (NB_DIR / "crisis_task4_evaluation_FINAL.ipynb", "Task 4: Final Evaluation & SHAP Analysis")
]

report_output_path = BASE_DIR / "FINAL_PIPELINE_RESULTS_REPORT.md"

def process_notebook(nb_path, task_title):
    md_content = f"# {task_title}\n\n"
    md_content += f"**File:** `{nb_path.name}`\n\n---\n"
    
    if not nb_path.exists():
        return f"### {task_title}\n*File not found: {nb_path}*\n\n"

    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        return f"### {task_title}\n*Error reading {nb_path}: {e}*\n\n"

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            markdown_text = "".join(cell.get('source', []))
            md_content += f"{markdown_text}\n\n"
        
        elif cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            if outputs:
                md_content += "#### 🏁 Output\n"
                md_content += "```text\n"
                for out in outputs:
                    if 'text' in out:
                        md_content += "".join(out['text'])
                    elif 'data' in out:
                        data = out['data']
                        if 'text/plain' in data:
                            md_content += "".join(data['text/plain'])
                    elif 'ename' in out:
                        md_content += f"Error: {out['ename']}: {out['evalue']}\n"
                md_content += "\n```\n\n"
    
    md_content += "\n\n"
    return md_content

full_report = "# 🌍 Africa Conflict-Induced Food Crisis Prediction — Full Pipeline Report\n\n"
for nb_p, title in notebooks_to_process:
    full_report += process_notebook(nb_p, title)

with open(report_output_path, 'w', encoding='utf-8') as f:
    f.write(full_report)

print(f"Report successfully saved to: {report_output_path}")
