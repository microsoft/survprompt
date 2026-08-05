"""Feature sets used by Survprompt ablation experiments."""

ABLATION_FEATURE_SETS = {
    "all_features": ["path", "genomics", "treatment", "demographics", "met", "stage", "lab"],
    "demographics_only": ["demographics"],
    "met_only": ["met"],
    "path_only": ["path"],
    "genomics_only": ["genomics"],
    "treatment_only": ["treatment"],
    "stage_only": ["stage"],
    "lab_only": ["lab"],
}

ABLATION_FEATURE_LABELS = {
    "all_features": "All features",
    "demographics_only": "Demographics only",
    "met_only": "Metastases only",
    "path_only": "Pathology only",
    "genomics_only": "Genomics only",
    "treatment_only": "Treatment only",
    "stage_only": "Disease state only",
    "lab_only": "Tumor markers only",
}
