from openai import OpenAI
from datetime import datetime
from typing import Dict
import time
import os
from dotenv import load_dotenv
from report_pipeline.categories.categories import Categories, RiskPattern, ContentType
from report_pipeline.content.content import Content, ContentClassification
from report_pipeline.classification.utils.prompt_generator import PromptGenerator
from report_pipeline.classification.utils.base import BaseClassifier

load_dotenv(override=True)

class ContentClassifier:
    """
    Main classifier service that handles content classification.
    """
    def __init__(self, classifier: BaseClassifier, prompt_generator: PromptGenerator, version: str):
        self._classifier = classifier
        self.prompt_generator = prompt_generator
        self._classifier_version = version
    
    def classify_content(self, content: Content) -> ContentClassification:
        """
        Classify new content and return classification details.
        """
        prompt = self.prompt_generator.generate_classification_prompt(content.prompt)
        response = self._classifier.classify(prompt)
        formatted_response = self._format_results(response, content.content_id)
        

        return formatted_response

    def _format_results(self, result: Dict, content_id: str) -> ContentClassification:
        categories = {}
        content_types = {}
        risk_patterns = {}
        try:
            for category in Categories().as_keys():
                categories[category] = 1.0 if category in result["labels"] else 0.0
            for content_type in ContentType.as_keys():
                content_types[content_type] = 1.0 if content_type in result["content_type"] else 0.0
            for risk_pattern in RiskPattern.as_keys():
                risk_patterns[risk_pattern] = 1.0 if risk_pattern in result["risk_patterns"] else 0.0
            
            return ContentClassification(
                content_id=content_id,
                categories=categories,
                severity_score=result.get("severity", None),
                classified_at=datetime.now(),
                classifier_version=self._classifier_version,
                metadata={
                    "human_review_needed": self.needs_human_review(result),
                    "rationale": result.get("rationale", None),
                    "content_type": content_types,
                    "risk_patterns": risk_patterns,
                }
            )
        
        except Exception as e:
            print(f"Error formatting results for {content_id}: {e}")
            return self._create_error_classification(content_id, str(e))

    def _create_error_classification(self, content_id: str, error: str) -> ContentClassification:
        return ContentClassification(
            content_id=content_id,
            categories={},
            severity_score=None,
            classified_at=datetime.now(),
            classifier_version=self._classifier_version,
            metadata={
                "human_review_needed": True,
                "rationale": error,
                "content_type": {},
                "risk_patterns": {},
            }
        )
    
    def needs_human_review(self, result: Dict) -> bool:
        """Determine if content needs human review based on classification."""
        categories = Categories().as_list()
        for category in categories:
            if (((category.requires_human_review) and category in result["labels"])) or (result["severity"]>=4):
                return True
        return False
      
    def get_init_kwargs(self):
        """
        Return the keyword arguments needed to re-instantiate this classifier.
        """
        return {
            "classifier": self._classifier,
            "version": self._classifier_version,
            "prompt_generator": self.prompt_generator,
        }
