"""
Text templates for Remote Sensing datasets
Used for zero-shot classification with CLIP
"""

# Generic remote sensing template
remote_sensing_template = [
    lambda c: f"a satellite image of {c}.",
    lambda c: f"an aerial view of {c}.",
    lambda c: f"a remote sensing image of {c}.",
    lambda c: f"a top-down view of {c}.",
    lambda c: f"an overhead photo of {c}.",
    lambda c: f"satellite imagery showing {c}.",
    lambda c: f"an aerial photograph of {c}.",
    lambda c: f"a bird's eye view of {c}.",
    lambda c: f"a satellite photo of {c}.",
    lambda c: f"an overhead view of {c}.",
]

# Dataset-specific templates
REMOTE_SENSING_TEMPLATES = {
    "AID": remote_sensing_template,
    "CLRS": remote_sensing_template,
    "EuroSAT_RGB": remote_sensing_template,
    "Million-AID": remote_sensing_template,
    "MLRSNet": remote_sensing_template,
    "MultiScene": remote_sensing_template,
    "NWPU-RESISC45": remote_sensing_template,
    "Optimal-31": remote_sensing_template,
    "PatternNet": remote_sensing_template,
    "RS_C11": remote_sensing_template,
    "RSD46-WHU": remote_sensing_template,
    "RSI-CB128": remote_sensing_template,
    "RSI-CB256": remote_sensing_template,
    "RSSCN7": remote_sensing_template,
    "SAT-4": remote_sensing_template,
    "SAT-6": remote_sensing_template,
    "SIRI-WHU": remote_sensing_template,
    "UC_Merced": remote_sensing_template,
    "WHU-RS19": remote_sensing_template,
}


def get_remote_sensing_template(dataset_name):
    """Get text template for a remote sensing dataset"""
    # Remove 'Val' suffix if present
    if dataset_name.endswith("Val"):
        dataset_name = dataset_name.replace("Val", "")
    
    if dataset_name not in REMOTE_SENSING_TEMPLATES:
        print(f"Warning: No template found for {dataset_name}, using generic remote sensing template")
        return remote_sensing_template
    
    return REMOTE_SENSING_TEMPLATES[dataset_name]

