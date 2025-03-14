from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from report_pipeline.content.content import Content
from dataclasses import dataclass

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, content: Any) -> Any:
        pass
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create a standardized error response."""
        return {
            "categories": [],
            "severity_score": 0.0,
            "rationale": f"Classification error: {error_message}",
            "content_type": None,
            "risk_patterns": None
        }

    def _parse_response(self, response: Dict) -> str:
        return {
            "labels": [],
            "severity": 0,
            "rationale": f"Fallback parse: {response}",
            "content_type": [],
            "risk_patterns": []
        }


@dataclass
class BaseClassifierConfig(ABC):
    model_name: str
    api_key: str

    def validate(self):
        if not self.model_name:
            raise ValueError("Model name cannot be empty.")
        if not self.api_key:
            raise ValueError("API key is missing from configuration.")

    



