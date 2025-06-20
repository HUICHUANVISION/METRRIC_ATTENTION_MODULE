# METRRIC_ATTENTION_MODULE
# Metric Attention Module (MAM) for Software Defect Prediction

This repository contains the implementation of the **Metric Attention Module (MAM)** for enhancing software defect prediction models via interpretable attention-based feature enhancement. The model captures intra-instance, inter-instance, intra-metric, and cross-metric relationships, improving traditional classifiers on benchmark datasets such as NASA CM1.

## 📂 Project Structure
├── models/

│   └── metric_attention_module.py   # MAM module 

├── WPDP_EXPERIMENTAL/           # WPDP_Experimental model & data

│   └── data  

│   └── Implementation

│   └── WPDP_results/

├── CPDP_EXPERIMENTAL/           # CPDP_Experimental model & data

│   └── data  

│   └── Implementation   
│   └── CPDP_results/
├── Interpretable_EXPERIMENTAL/           # Interpretable_Experimental model & data
│   └── data  
│   └── Interpretable results/
├── requirements.txt
└── README.md

📊 Outputs

All generated plots and CSVs will be saved in the results/ directory

🧠 Model Details

The Metric Attention Module (MAM) computes four types of relational features:
	•	Corr1: Inter-metric feature relationships within instance
	•	Corr2: Inter-instance feature relationships
	•	Corr3: Correlation within metric features
	•	Corr4: Correlation between single metric features

Each branch uses a scaled dot-product attention mechanism with learnable weights. Outputs are fused with coefficients (α, β, γ, δ) and applied to raw features for enhanced representations.

📄 Dataset
	•	Source: AEEEM NASA JIRA Promise Repository
	•	File: data
	•	Format: Tabular CSV with metric features and binary Defective labels (Y/N)

📜 License

This project is licensed under the MIT License. See LICENSE for details.

🔬 Citation

If you use this work in your research, please cite the corresponding paper (to be updated).

📫 Contact

For questions, please contact [cs_din.yc@whu.edu.com].
