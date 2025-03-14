import numpy as np
import pandas as pd 
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import uuid

## category imports
from ..categories.categories import Categories, ContentType, RiskPattern
 

@dataclass
class ContentClassification:
    """
    Represents a single content classification event.
    Includes metadata for auditing and analysis.
    """
    content_id: str
    categories: Dict[str, float]
    severity_score: float = np.nan
    classified_at: datetime = field(default_factory=datetime.now())
    classifier_version: str = None
    classifier_id: str = None # 
    metadata: Dict = None  # Additional context, source info, etc.

    @staticmethod
    def from_json(json_data, content_id):
      categories = {}
      for category in Categories().as_keys():
        if category in json_data:
          categories[category] = float(json_data[category])
        else:
          categories[category] = np.nan
    
      return ContentClassification(
          content_id=content_id,
          categories=categories,
          classifier_version=json_data.get("classifier_version","legacy"),
          classifier_id=json_data.get("classifier_id", None),
          metadata={}
      )

@dataclass
class Content:
    prompt: str
    content_id: str = field(default_factory = uuid.uuid4)
    classifications: List[ContentClassification] = field(default_factory=list)
    
    @property
    def latest_classification(self) -> Optional[ContentClassification]:
        return self.classifications[-1] if self.classifications else None

    def add_classification(self, classification: ContentClassification) -> None:
        self.classifications.append(classification)

