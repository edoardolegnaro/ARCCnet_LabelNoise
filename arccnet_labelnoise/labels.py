"""Label mappings: the ARCCnet v0.4.0 mappings plus the two label policies of the δ ablation."""

from arccnet.models.labels import CLASS_NAMES_DICT as _ARCCNET_CLASS_NAMES_DICT
from arccnet.models.labels import LABEL_MAPPING_DICT as _ARCCNET_LABEL_MAPPING_DICT

# Three classes with δ-bearing regions dropped (Section 4.3, "Drop delta rows").
A_B_BG_NO_DELTA_MAPPING = {
    "QS": None,
    "IA": None,
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": None,
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": None,
    "Gamma": None,
    "Gamma-Delta": None,
}

# Flat five-class problem (Section 4.3).
A_B_BD_BG_BGD_MAPPING = {
    "QS": None,
    "IA": None,
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta-Delta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma-Delta",
    "Gamma": None,
    "Gamma-Delta": None,
}

LABEL_MAPPING_DICT = {
    **_ARCCNET_LABEL_MAPPING_DICT,
    "a-b-bg-no-delta": A_B_BG_NO_DELTA_MAPPING,
    "a-b-bd-bg-bgd": A_B_BD_BG_BGD_MAPPING,
}

CLASS_NAMES_DICT = {
    **_ARCCNET_CLASS_NAMES_DICT,
    "a-b-bg-no-delta": ["Alpha", "Beta", "Beta-Gamma"],
    "a-b-bd-bg-bgd": ["Alpha", "Beta", "Beta-Delta", "Beta-Gamma", "Beta-Gamma-Delta"],
}
