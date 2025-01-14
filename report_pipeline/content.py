import gzip
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, FrozenSet, Set, Tuple, Union, Callable
from datetime import datetime
import uuid
import pickle
import os

## category imports
from .categories import Labels
# Get current working directory
current_dir = os.getcwd()

# Construct the full path
file_path = os.path.join(current_dir, "data", "samples-1680.jsonl.gz")



@dataclass
class ContentClassification:
    """
    Represents a single content classification event.
    Includes metadata for auditing and analysis.
    """
    content_id: str
    categories: Dict[str, float]
    #confidence_scores: Dict[CategoryLabel, float]
    severity_score: float = np.nan
    classified_at: datetime = None
    classifier_version: str = None
    classifier_id: str = None # e.g., "model", "human-rater-123"
    #metadata: Dict  # Additional context, source info, etc.

    @staticmethod
    def from_json(json_data, content_id):
      categories = {}
      for category in Labels().as_keys():
        if category in json_data:
          categories[category] = float(json_data[category])
        else:
          categories[category] = np.nan
    
      return ContentClassification(
          content_id=content_id,
          categories=categories,
          classified_at=None,
          classifier_version="legacy",
          classifier_id=0,
      )

@dataclass
class Content:
  content_id: str
  prompt: str
  all_classifications: List[ContentClassification]

  @property
  def classification(self):
    """Return classification based on some logic, .e.g latest/time based"""
    return self.all_classifications[-1]

  @staticmethod
  def from_json(json_data):
    content_id=str(uuid.uuid4())
    classification = ContentClassification.from_json(json_data, content_id=content_id)
    return Content(
        content_id=content_id,
        prompt=json_data["prompt"],
        all_classifications=[classification]
    )
  
  def needs_classification(self, content_classifier):
    for classification in self.all_classifications:
      if content_classifier.classifier_version == classification.classifier_version:
        return False
    return True
  
  def classify(self, content_classifier):
    classification = content_classifier.classify_content(self)
    self.all_classifications.append(classification)

class ContentDataSet:
  data: List[Content] | None

  def __init__(self):
    self.data = []

  def __str__(self):
    return f"ContentDataSet, length {len(self.data)}"
  
  def __repr__(self):
    return f"ContentDataSet, length {len(self.data)}"
  
  def load_from_file(self, file_path: str = file_path, file_type: str ="jsonl.gz"):
    all_prompts = self.get_all_prompts()
    if file_type == "jsonl.gz":
      with gzip.open(file_path, 'rt') as f:
        # Read lines and parse JSON
        for line in f:
          json_data = json.loads(line)
          if json_data["prompt"] not in all_prompts:
            content = Content.from_json(json_data)
            self.data.append(content) 
    else:
      raise NotImplementedError(f"file_type {file_type} not supported.")
  
  def get_all_prompts(self) -> list[str]:
    all_prompts = [content.prompt for content in self.data]
    return all_prompts

  def to_pandas(self):
    """
    Converts self.data (List[Content]) into a DataFrame that
    has one row *per classification* rather than one row per content.
    """
    records = []

    for content in self.data:
      for classification_idx, classification in enumerate(content.all_classifications):
        row_dict = {
                "content_id": content.content_id,
                "prompt": content.prompt,
                "classification_idx": classification_idx,
                "classified_at": classification.classified_at,
                "classifier_version": classification.classifier_version,
                "classifier_id": classification.classifier_id,
            }

        for category_label, score in classification.categories.items():
                row_dict[f"{category_label}"] = score
        records.append(row_dict)
    
    return pd.DataFrame(records)

  def to_pickle(self):
    """ Save ContentDataSet to pickle file """
    with open('dataframe.pkl', 'wb') as file:
        pickle.dump(self.data, file)

  def to_json(self):
      """Future work"""
      raise NotImplementedError    
  
  def get_by_content_id(self, content_id: str) -> Content:
    for content in self.data:
      if content.content_id == content_id:
        return content
      else:
        raise f"Content ID {content_id} does not exist."
  
  def update_classifications(self, content_classifier, force_reclassify=False, index_range: tuple | None = None):
    data = self.data
    if index_range:
        data = self.data[index_range[0]:index_range[1]]
    
    for content in tqdm(data, desc="Classifying content", unit="item"):
        if force_reclassify or content.needs_classification(content_classifier):
            content.classify(content_classifier)
      
