from data.rossmann.rossmann_preprocessing import generate_rossmann_classification
from data.unilever.preprocessing import preprocess_rule_based_data

dataset_loader = dict(
    rossmann = generate_rossmann_classification,
    rule_based = preprocess_rule_based_data,
)