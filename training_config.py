@dataclass
class TrainingConfig:
    model_name: str
    model_type: str
    model_params: dict
    model_path: str
    model_history: dict
    model_metrics: dict
    model_predictions: dict
    model_confusion_matrix: dict
    model_classification_report: dict

