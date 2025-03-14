from report_pipeline.classification.utils.base import BaseClassifier, BaseClassifierConfig
from anthropic import Anthropic
from typing import Dict
from report_pipeline.categories.categories import Categories, ContentType, RiskPattern

class AnthropicClassifier(BaseClassifier):
    def __init__(self, config: BaseClassifierConfig):
        self.model = config.model_name
        self.api_key = config.api_key
        self.client = Anthropic(api_key=config.api_key)
    def classify(self, content: str) -> Dict:
        try:
            tools = [{
                "name": "classify_content",
                "description": "Classify content into predefined categories",
                "input_schema": {
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
            response = self.client.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                tools=tools,
                messages=[{"role": "user", "content": content}])
            for content in response.content:
                if content.type == "tool_use" and content.name == "classify_content":
                    result = content.input
                    if result:
                        return result
                    else:
                        return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            # Log the error and return a default error response
            print(f"Error in Anthropic classification: {str(e)}")
            return self._create_error_response(str(e))

