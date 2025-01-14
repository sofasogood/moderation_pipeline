from openai import OpenAI
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from .categories import Labels
from .content import Content, ContentClassification, ContentDataSet

MODEL_NAME= "gpt-3.5-turbo"
CLASSIFIER_VERSION="gpt3.5"

load_dotenv()

class ContentClassifier:
    """
    Main classifier service that handles content classification.
    """
    def __init__(self, model_name: str = MODEL_NAME, classifier_version: str = CLASSIFIER_VERSION):
        self.model_name = model_name
        self.model_api_key=os.environ.get('OPEN_AI_API_KEY')
        # API call
        self._client = OpenAI(api_key=self.model_api_key)
        self._classifier_version = classifier_version

    @property
    def classifier_version(self):
      return self._classifier_version

    def classify_content(self, content: Content) -> ContentClassification:
        """
        Classify new content and return classification details.
        """
        # In practice, this would call your ML model or classification system
        prompt = content.prompt
        raw_classification = self._predict(prompt)
        classification = self.parse_response(raw_classification)["classification"]
        categories = {}
        for category in Labels().as_keys():
          categories[category] = 1.0 if category in classification["label"] else 0.0
        
        
        return ContentClassification(
            content_id=content.content_id,
            categories=categories,
            #confidence_scores=confidence,
            severity_score=classification["severity"][0],
            classified_at=datetime.utcnow(),
            classifier_version=self._classifier_version,
            classifier_id=1,
            # metadata={
            #     "content_length": len(content),
            #     "language": "en",  # Would be detected
            #     # Other relevant metadata
            # }
        )


    def create_detection_prompt(self, content: str) -> str:
        """Creates a structured prompt for harmful content detection."""
        category_labels = Labels()
        label_keys, category_names, descriptions = category_labels.metadata_as_list()

        
        # Base prompt with clear instructions and expectations
        prompt_parts = [
            "Analyze the following content for harmful elements:",
            f"Content: {content}",
            "\nInstructions:",
            "1. Determine if the content contains harmful elements",
            "2. Choose the most appropriate classification label from:",
        ]
        # Add labels in a structured format
        for label, category, description in zip(
            label_keys, category_names, descriptions
        ):
            prompt_parts.append(
                f"   - {label}: {category}\n     Description: {description}"
            )

         # Add emergent behavior handling with specific format
        prompt_parts.extend([
            "\nRequired Output Format:",
            '{"classification": {',
            '    "label": ["<primary_label>", "<secondary_label>",....],',
            '    "severity": [<severity>],',
            '    "rationale": ["<detailed_explanation>"]',
            '}}',
            "\nNotes:",
            "- Provide as many labels as necessary to detect harm.",
            "- Assign only one severity score. It must be an integer from 0-5",
            "- Rationale must be a single, detailed explanation"
        ])
                # Add severity guidelines with specific scoring instructions
        severity_guidelines = {
            0: {
                "name": "Negligible",
                "description": "No clear potential for harm",
                "indicators": [
                    "Content is benign or neutral",
                    "No safety concerns present",
                    "May be off-topic but harmless",
                    "Standard user queries or interactions"
                ]
            },
            1: {
                "name": "Minor",
                "description": "Limited potential for indirect harm",
                "indicators": [
                    "Mild stereotypes or biases",
                    "Minor factual inaccuracies",
                    "Subtle attempts to bypass filters",
                    "Low-level spamming behavior"
                ]
            },
            2: {
                "name": "Moderate",
                "description": "Direct but limited potential for harm",
                "indicators": [
                    "Deliberate misinformation",
                    "Basic privacy violations",
                    "Clear attempts to evade content filters",
                    "Generation of misleading content"
                ]
            },
            3: {
                "name": "Substantial",
                "description": "Clear potential for significant harm",
                "indicators": [
                    "Coordinated harmful behavior",
                    "Sophisticated system manipulation",
                    "Serious misinformation campaigns",
                    "Generation of harmful content"
                ]
            },
            4: {
                "name": "Critical",
                "description": "Immediate and severe potential for harm",
                "indicators": [
                    "Dangerous misinformation about critical topics",
                    "Promotion of harmful extremist content",
                    "Generation of dangerous instructions",
                    "Advanced system exploitation attempts"
                ]
            },
            5: {
                "name": "Catastrophic",
                "description": "Potential for widespread, severe harm",
                "indicators": [
                    "Large-scale attack coordination",
                    "Critical vulnerability exploitation",
                    "Potential for cascading negative effects",
                    "Severe real-world harm potential"
                ]
            }
        }
        
        # Add severity scoring instructions
        prompt_parts.extend([
            "\nSeverity Scoring Instructions:",
            "1. Review the content against each severity level's indicators",
            "2. Choose the level that best matches the content's harmful elements",
            "3. Assign a single severity score",
            "\nSeverity Scale:"
        ])
        
        for level, info in severity_guidelines.items():
            prompt_parts.extend([
                f"\nLevel {level}: {info['name']} - {info['description']}",
                "Indicators:"
            ])
            for indicator in info['indicators']:
                prompt_parts.append(f"- {indicator}")
        
        return "\n".join(prompt_parts)

        
    
    def _predict(self, content: str) -> None:
        """
        Make predictions for content.

        """
        prompt = self.create_detection_prompt(content=content)

        start_time = time.time()
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt

                }
            ]
        )
        print(f"Taken for {content[:10]}...{content[-10:]}: {time.time() - start_time:.3f}")

        return (completion.choices[0].message.content)


    def parse_response(self, llm_response: str) -> dict:
        """
        Parses the LLM response into a structured format.
        Expects response in the format specified in the prompt.
        """
        import re
        try:
            # Simple patterns to match content within each field
            label_pattern = r'"label":\s*\[(.*?)\]'
            severity_pattern = r'"severity":\s*(\[.*?\]|\d+(?:\.\d+)?)'
            rationale_pattern = r'"rationale":\s*(\[.*?\]|".*?")'
            
            # Extract each field
            labels = re.findall(r'"(.*?)"', re.search(label_pattern, llm_response, re.DOTALL).group(1))
            
            # For severity, handle both array and single number
            severity_match = re.search(severity_pattern, llm_response, re.DOTALL).group(1)
            severities = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', severity_match)]
            
            # For rationale, extract quoted strings
            rationale_match = re.search(rationale_pattern, llm_response, re.DOTALL).group(1)
            rationales = re.findall(r'"(.*?)"', rationale_match)
            
            return {
                "classification": {
                    "label": labels,
                    "severity": severities,
                    "rationale": rationales
                }
            }
            
        except Exception as e:
            print(f"Debug - Raw response: {repr(llm_response)}")
            return {
                "classification": {
                    "label": ["ERROR"],
                    "severity": [0],
                    "rationale": [f"Failed to parse LLM response: {str(e)}"]
                }
            }

      
    def update_content_dataset(self, dataset: ContentDataSet, content_id: str, prompt:str, content_classification: ContentClassification) -> None:
      content_data = dataset.data
      for content in content_data:
        if content.content_id==content_id:
          content.all_classifications.append(content_classification)
      else:
        content_data.append(Content(content_id=content_id, prompt=prompt, all_classifications=[content_classification]))


    def needs_human_review(self, classification: ContentClassification) -> bool:
        """Determine if content needs human review based on classification."""
        for category in classification.categories:
            if (self.categories[category].requires_human_review): #TODO
                return True
        return False