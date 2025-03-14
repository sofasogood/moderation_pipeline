import gzip
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import pickle
import os
from datasets import load_dataset, Dataset


## category imports
from report_pipeline.content.content import Content, ContentClassification
from report_pipeline.categories.categories import ContentType, RiskPattern
 
class ContentDataSet:
  # data: List[Content] | None

  def __init__(self):
    self.data: List[Content] = []
    self._by_id: Dict[str, Content]= {}
  
  def __repr__(self):
     return f"ContentDataSet, length {len(self.data)}"
  
  def __len__(self):
     return len(self.data)

  def all_items(self) -> List[Content]:
     return self.data

  
  def get_by_content_id(self, content_id: str) -> Content:
    if content_id in self._by_id:
       return self._by_id[content_id]
    else:
        raise f"Content ID {content_id} does not exist."

  def load_from_dataset(self, name: str = "google/civil_comments", index_range: tuple | None = None, random_sample: int | None = None):
    dataset = load_dataset(name)["train"].to_pandas()
    if random_sample:
        dataset = dataset.sample(n=random_sample)
    if index_range:
        dataset = dataset[index_range[0]:index_range[1]]
    for idx, row in dataset.iterrows():
        content = Content(prompt=row["text"])
        self.data.append(content)
        self._by_id[content.content_id] = content
    print(f"Loaded {len(self.data)} items from dataset.")

  def load_from_pickle(self, pickle_path: str):
    print(f"Attemping to load from {pickle_path}")
    if os.path.exists(pickle_path):
        # To load data from a pickle file
      with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
        self.data = data.data
      print(f"Loaded {len(self.data)} items from pickle file.")
    else:
      print("Pickle file does not exist.")
  
  def process_json_data(self, file_path: str, file_type: str ="jsonl.gz"):
    print(f"Processing new data from {file_path}")
    all_prompts = self.get_all_prompts()
    new_count = 0
    if file_type == "jsonl.gz":
      with gzip.open(file_path, 'rt') as f:
        # Read lines and parse JSON
        for line in f:
          json_data = json.loads(line)
          if json_data["prompt"] not in all_prompts:
            content = Content.from_json(json_data)
            self.data.append(content) 
            new_count += 1
      print(f"Added {new_count} new items. Total items: {len(self.data)}")
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
      for classification_idx, classification in enumerate(content.classifications):

        row_dict = {
                "content_id": content.content_id,
                "prompt": content.prompt,
                "classification_idx": classification_idx,
                "classified_at": classification.classified_at,
                "classifier_version": classification.classifier_version,
                "classifier_id": classification.classifier_id,
                "severity_score": classification.severity_score,
                "rationale": (classification.metadata or {}).get("rationale", None),
                "needs_human_review": (classification.metadata or {}).get("human_review_needed", "N/A"),

            }
        for category_label, score in classification.categories.items():
                row_dict[f"{category_label}"] = score
        for content_type, score in classification.metadata.get("content_type", {}).items():
                row_dict[f"CONTENT_TYPE_{content_type}"] = score
        for risk_pattern, score in classification.metadata.get("risk_patterns", {}).items():    
                row_dict[f"RISK_PATTERN_{risk_pattern}"] = score
        
        records.append(row_dict)
    return pd.DataFrame(records)

  def save_to_pickle(self, pickle_path: str = "dataframe.pkl", create_latest_link: bool = False):
      """
      Save ContentDataSet to pickle file with timestamp and optional "latest" symlink   
      """
      # Split the path into directory and filename
      directory, filename = os.path.split(pickle_path)
      name, ext = os.path.splitext(filename)
      
      # Create timestamp and new filename
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      timestamped_filename = f"{name}_{timestamp}{ext}"
      
      # Combine directory and new filename
      if directory:
          final_path = os.path.join(directory, timestamped_filename)
      else:
          final_path = timestamped_filename

      # Save the pickle file
      with open(final_path, 'wb') as file:
          pickle.dump(self, file)

      # Create/update the "latest" symlink if requested
      if create_latest_link:
          latest_path = os.path.join(directory, f"{name}_latest{ext}")
          # Remove existing symlink or file if it exists
          try:
              if os.path.islink(latest_path) or os.path.exists(latest_path):
                  os.unlink(latest_path)
          except OSError as e:
              print(f"Warning: Could not remove existing link: {e}")
          
          # Create new symlink
          try:
              os.symlink(os.path.basename(timestamped_filename), latest_path)
          except OSError as e:
              print(f"Warning: Could not create new symlink: {e}")          
  

  def process_content_batch(self, content_items, classifier_class, classifier_kwargs):
    """Process a batch of content items in parallel and return updated items
        """
    # Create a new classifier instance in this process using the provided class
    classifier = classifier_class(**classifier_kwargs)
    updated_contents = []
    
    for content in content_items:
        try:
            if content.needs_classification(classifier):
                content.classify(classifier)
            updated_contents.append(content)
        except Exception as e:
            print(f"Error processing content {content.content_id}: {str(e)}")
    
    return updated_contents
