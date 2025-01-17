from openai import OpenAI
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from .categories import Labels, RiskPattern, ContentType
from .content import Content, ContentClassification, ContentDataSet

MODEL_NAME= "gpt-4o-mini"
CLASSIFIER_VERSION="gpt4omini"

load_dotenv(override=True)

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
        start_time = time.time()
        raw_classification = self._predict(prompt)
        print(f"[{content.content_id}]: time for {prompt[:10]}...{prompt[-10:]}: {time.time() - start_time:.3f}")
        
        #classification parsing
        classification = self.parse_response(raw_classification)["classification"]
        categories = {}
        for category in Labels().as_keys():
          categories[category] = 1.0 if category in classification["label"] else 0.0
        
        
        return ContentClassification(
            content_id=content.content_id,
            categories=categories,
            #confidence_scores=confidence,
            severity_score=classification["severity"][0],
            classified_at=datetime.now(),
            classifier_version=self._classifier_version,
            classifier_id=1,
            metadata={
                 "human_review_needed": self.needs_human_review(classification),
                 "rationale": classification["rationale"],
                 "content_type": classification["content_type"],
                 "risk_patterns": classification["risk_patterns"],
                 
            }
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
            "1. Determine if the content contains harmful elements, along with severity and rationale.",
            "2. Choose the most appropriate classification label from:",
        ]
        # Add labels in a structured format
        for label, category, description in zip(
            label_keys, category_names, descriptions
        ):
            prompt_parts.append(
                f"   - {label}: {category}\n     Description: {description}"
            )
        
        prompt_parts.append(f"""Please analyze the content and classify it according to these taxonomies:

                                1. CONTENT_TYPE (select one):
                                {ContentType.get_prompt_format()}

                                2. RISK_PATTERNS (select all that apply):
                                {RiskPattern.get_prompt_format()}
                                
                            """)

        
        # Add emergent behavior handling with specific format
        prompt_parts.extend([
            "\nRequired Output Format:",
            '{"classification": {',
            '    "label": ["<primary_label>"],',
            '    "severity": [<severity>],',
            '    "rationale": ["<detailed_explanation>"]',
            '    "content_type": ["<content_type>"]',
            '    "risk_patterns": ["<risk_pattern>"]',
            '}}',
            "\nNotes:",
            "- Provide as many labels as necessary to detect harm. However, if one of the labels is 'N', pick only one label, the one that best describes the content.",
            "- Assign only one severity score. It must be an integer from 0-5",
            "- Rationale must be a single explanation with maximum 300 characters, be concise and precise."
            "- Content type and risk patterns must be selected from the provided lists. If the pattern is not included in the list, select 'Unclassified'. If no patterns apply, return 'None'."
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
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt

                }
            ],
        )
        return (completion.choices[0].message.content)


    def parse_response(self, llm_response: str) -> dict:
        """
        Parses the LLM response into a structured format.
        Expects response in the format specified in the prompt.
        """
        import re
        try:
            # Default values
            labels = ["ERROR"]
            severities = [0]
            rationales = ["Failed to parse"]
            content_type = [ContentType.UNCLASSIFIED]
            risk_patterns = [RiskPattern.NONE]
            

            # Label extraction with fallback
            label_match = re.search(r'"label":\s*\[(.*?)\]', llm_response, re.DOTALL)
            if not label_match:
                raise ValueError(f"Could not find label field in response: {llm_response[:100]}...")
                
            label_content = label_match.group(1)
            found_labels = re.findall(r'"([^"]+)"', label_content)
            if not found_labels:
                raise ValueError(f"Could not parse labels from content: {label_content}")
            labels = found_labels

            # Severity extraction with fallback
            severity_match = re.search(r'"severity":\s*(\[.*?\]|\d+(?:\.\d+)?)', llm_response, re.DOTALL)
            if not severity_match:
                raise ValueError(f"Could not find severity field in response: {llm_response[:100]}...")
                
            severity_content = severity_match.group(1)
            found_severities = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', severity_content)]
            if not found_severities:
                raise ValueError(f"Could not parse severities from content: {severity_content}")
            severities = found_severities


            # Rationale extraction
            rationale_match = re.search(r'"rationale":\s*\[\s*"([^"]*)', llm_response)
            if not rationale_match:
                raise ValueError(f"Could not find rationale field in response: {llm_response[:100]}...")
            rationales = [rationale_match.group(1)]

            # Content type extraction - always returns a list with at least one valid ContentType
            type_match = re.search(r'"content_type":\s*\[(.*?)\]', llm_response, re.DOTALL)
            if type_match:
                type_content = type_match.group(1)
                type_values = re.findall(r'"([^"]+)"', type_content)
                if type_values:
                    found_types = [
                        ct for val in type_values
                        for ct in ContentType
                        if ct.value.lower() == val.strip().lower()
                    ]
                    content_type = found_types if found_types else [ContentType.UNCLASSIFIED]

            # Risk patterns extraction - always returns a list with at least one valid RiskPattern
            risk_match = re.search(r'"risk_patterns":\s*\[(.*?)\]', llm_response, re.DOTALL)
            if risk_match:
                risk_content = risk_match.group(1)
                risk_values = re.findall(r'"([^"]+)"', risk_content)
                if risk_values:
                    found_patterns = [
                        rp for val in risk_values
                        for rp in RiskPattern
                        if rp.value.lower() == val.strip().lower()
                    ]
                    risk_patterns = found_patterns if found_patterns else [RiskPattern.NONE]
        
            return {
                "classification": {
                    "label": labels,
                    "severity": severities,
                    "rationale": rationales,
                    "content_type": content_type,
                    "risk_patterns": risk_patterns
                }
            }
                
        except Exception as e:
            print(f"Debug - Error details for content: {str(e)} \n{llm_response}")
            return {
                "classification": {
                    "label": ["ERROR"],
                    "severity": [0],
                    "rationale": [f"Parser error: {str(e)}"],
                    "content_type": [ContentType.UNCLASSIFIED],
                    "risk_patterns": [RiskPattern.NONE]
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
        categories = Labels().as_list()
        for category in categories:
            if (((category.requires_human_review) and category in classification["label"])) or (classification["severity"][0]>=4): #TODO
                return True
        return False