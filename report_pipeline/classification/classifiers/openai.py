from report_pipeline.classification.utils.base import BaseClassifier, BaseClassifierConfig
from openai import OpenAI
from typing import Dict
from report_pipeline.categories.categories import Categories, ContentType, RiskPattern

import json

class OpenAIClassifier(BaseClassifier):
    def __init__(self, config: BaseClassifierConfig):
        self.model = config.model_name
        self.api_key = config.api_key
        self.client = OpenAI(api_key=config.api_key)

    def classify(self, content: str) -> Dict:
        try:
            functions = [
                {
                    "name": "classify_content",
                    "description": "Classify content into predefined categories",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "labels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": Categories().as_keys()
                            },
                            "description": "Classification labels (can include multiple)"
                                },
                        "severity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 5,
                            "description": "Severity of the content"
                        },
                                                    "rationale": {
                                "type": "string",
                                "description": "Explanation for the classification"
                            },
                            "content_type": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ContentType.as_keys()
                                },
                                "description": "Type of content"
                            },
                            "risk_patterns": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": RiskPattern.as_keys()
                                },
                                "description": "Risk patterns identified in the content"
                            }
                        },
                        "required": ["labels", "severity", "rationale", "content_type", "risk_patterns"]
                            }
                        }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a content moderation assistant."},
                    {"role": "user", "content": content},
                ],
                functions=functions
            )
            function_response = response.choices[0].message.function_call
            if function_response and function_response.name == "classify_content":
                result = json.loads(function_response.arguments)
                return result
            else:
                return self._parse_response(function_response)


        except Exception as e:
            # Log the error and return a default error response
            print(f"Error in OpenAI classification: {str(e)}")
            return self._create_error_response(str(e))


class OpenAIClassifierConfig(BaseClassifierConfig):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.validate()