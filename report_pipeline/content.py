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
import concurrent.futures
from tqdm import tqdm
import math

## category imports
from .categories import Labels, ContentType, RiskPattern
 

@dataclass
class ContentClassification:
    """
    Represents a single content classification event.
    Includes metadata for auditing and analysis.
    """
    content_id: str
    categories: Dict[str, float]
    severity_score: float = np.nan
    classified_at: datetime = None
    classifier_version: str = None
    classifier_id: str = None # e.g., "model", "human-rater-123"
    metadata: Dict = None  # Additional context, source info, etc.

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
  
  def process_new_data(self, file_path: str, file_type: str ="jsonl.gz"):
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
      for classification_idx, classification in enumerate(content.all_classifications):

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
        
        # Handle content types
        content_types = (classification.metadata or {}).get("content_type", [ContentType.UNCLASSIFIED])
        if not isinstance(content_types, list):
            content_types = [content_types]
        
        for ct in ContentType.as_keys():
            if len(ct) > 0:  # Skip empty keys
                # Convert enum values to strings for comparison
                content_type_values = [ct_enum.value for ct_enum in content_types]
                row_dict[f"CONTENT_TYPE_{ct}"] = 1.0 if ct in content_type_values else 0.0
        
        # Handle risk patterns
        risk_patterns = (classification.metadata or {}).get("risk_patterns", [RiskPattern.UNCLASSIFIED])
        if not isinstance(risk_patterns, list):
            risk_patterns = [risk_patterns]
            
        for rp in RiskPattern.as_keys():
            if len(rp) > 0:  # Skip empty keys
                # Convert enum values to strings for comparison
                risk_pattern_values = [rp_enum.value for rp_enum in risk_patterns]
                row_dict[f"RISK_PATTERN_{rp}"] = 1.0 if rp in risk_pattern_values else 0.0


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
  def to_json(self):
      """Future work"""
      raise NotImplementedError    
  
  def get_by_content_id(self, content_id: str) -> Content:
    for content in self.data:
      if content.content_id == content_id:
        return content
      else:
        raise f"Content ID {content_id} does not exist."
  

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

  def update_classifications(self, content_classifier, force_reclassify=False, 
                          index_range: tuple | None = None, parallel=False, 
                          batch_size=10, max_workers=4):
      """
      Update classifications with optional parallel processing.
      
      Args:
          content_classifier: Classifier instance to use
          force_reclassify (bool): Whether to reclassify all content
          index_range (tuple): Optional range of indices to process
          parallel (bool): Whether to use parallel processing
          batch_size (int): Number of items to process in each parallel batch
          max_workers (int): Maximum number of parallel workers
      """
      data = self.data
      if index_range:
          data = self.data[index_range[0]:index_range[1]]
      
      if not parallel:
          # Original sequential processing
          for content in tqdm(data, desc="Classifying content", unit="item"):
              if force_reclassify or content.needs_classification(content_classifier):
                  content.classify(content_classifier)
          return
      
      # Get classifier class and initialization parameters
      classifier_class = content_classifier.__class__
      classifier_kwargs = {
          'model_name': content_classifier.model_name,
          'classifier_version': content_classifier.classifier_version
      }
      
      # Create batches with their original indices
      num_items = len(data)
      num_batches = math.ceil(num_items / batch_size)
      batches = []
      batch_indices = []
      
      for i in range(num_batches):
          start_idx = i * batch_size
          end_idx = min(start_idx + batch_size, num_items)
          batch_data = data[start_idx:end_idx]
          batches.append(batch_data)
          batch_indices.append((start_idx, end_idx))
      
      # Process batches in parallel
      with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
          future_to_indices = {
              executor.submit(self.process_content_batch, batch, classifier_class, classifier_kwargs): idx 
              for batch, idx in zip(batches, batch_indices)
          }
          
          # Monitor progress and update dataset
          with tqdm(total=num_items, desc="Classifying content", unit="item") as pbar:
              for future in concurrent.futures.as_completed(future_to_indices):
                  indices = future_to_indices[future]
                  try:
                      updated_contents = future.result()
                      # Update the main dataset with the processed content
                      start_idx, end_idx = indices
                      for i, content in enumerate(updated_contents):
                          original_idx = start_idx + i
                          if index_range:
                              # adjust the index
                              original_idx = index_range[0] + original_idx
                          self.data[original_idx] = content
                      
                      pbar.update(end_idx - start_idx)
                  except Exception as e:
                      print(f"Batch processing error for indices {indices}: {str(e)}")

