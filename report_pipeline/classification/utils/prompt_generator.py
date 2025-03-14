
from report_pipeline.content.content import Content
from report_pipeline.categories.categories import Categories, ContentType, RiskPattern

class PromptGenerator:
    def generate_classification_prompt(self, content: str) -> str:
        """Creates a structured prompt for harmful content detection."""
        category_labels = Categories()
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

