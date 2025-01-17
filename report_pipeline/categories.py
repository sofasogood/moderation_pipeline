from dataclasses import dataclass   
from enum import Enum 
from typing import Optional, List

@dataclass(frozen=True)
class Category:
    """
    Immutable category definition.
    Contains metadata about a content moderation category.
    """
    # label: CategoryLabel
    label: str
    category_name: str
    description: str
    requires_human_review: Optional[bool] = None  # Flag for categories that need human verification
    def __hash__(self):
          # Using label as the hash since it's unique
          return hash(self.label)
      
    def __eq__(self, other):
          if not isinstance(other, Category):
              return False
          return self.label == other.label


@dataclass(frozen=True)
class Labels:
    S = Category(
        label="S",
        category_name="sexual",
        description="Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
    )
    H = Category(
        label="H",
        category_name="hate",
        description="Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
    )
    V = Category(
        label="V",
        category_name="violence",
        description="Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
    )
    HR = Category(
        label="HR",
        category_name="harassment",
        description="Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.",
    )
    SH = Category(
        label="SH",
        category_name="self-harm",
        description="Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
    )
    S3 = Category(
        label="S3",
        category_name="sexual/minors",
        description="Sexual content that includes an individual who is under 18 years old.",
    )
    H2 = Category(
        label="H2",
        category_name="hate/threatening",
        description="Hateful content that also includes violence or serious harm towards the targeted group."
    )
    V2 = Category(
        label="V2",
        category_name="violence/graphic",
        description="Violent content that depicts death, violence, or serious physical injury in extreme graphic detail."
    )
    U = Category(
        label="U",
        category_name="unclassified",
        description="Harmful content that does not fit into known categories",
        requires_human_review=True
    )
    N = Category(
        label="N",
        category_name="none",
        description="Content contains some keywords that feature in harmful content, but is not classified as harmful.",
        requires_human_review=True
    )   


    def as_list(self):
      categories = []
      for attr_name in dir(self):
        if not attr_name.startswith('_'):
            attr = getattr(self, attr_name)
            if isinstance(attr, Category):
                categories.append(attr)
      return categories

    def as_dict(self):
        category_list = self.as_list()
        return {category.label:category.category_name for category in category_list}
    
    def as_keys(self):
      category_list = self.as_list()
      return [category.label for category in category_list]

    def metadata_as_list(self):
      category_list = self.as_list()
      label_list = []
      category_name_list = []
      description_list = []

      for c in category_list:
        label_list.append(c.label)
        category_name_list.append(c.category_name)
        description_list.append(c.description)

      return label_list, category_name_list, description_list
      
class ContentType(Enum):
    QUESTION_QUERY = "Question/Query"
    CALL_TO_ACTION = "Call_to_Action"
    NEWS_INFORMATION = "News/Information_Sharing"
    COMPLAINT_GRIEVANCE = "Complaint/Grievance"
    DEBATE_ARGUMENT = "Debate/Argument"
    EMOTIONAL_EXPRESSION = "Emotional_Expression"
    PERSONAL_EXPERIENCE = "Personal_Experience"
    EDUCATIONAL_CONTENT = "Educational_Content"
    PROMOTION_ADVERTISEMENT = "Promotion/Advertisement"
    UNCLASSIFIED = "Unclassified"

    @classmethod
    def get_prompt_format(cls) -> str:
        return "\n".join([f"- {member.value}" for member in cls])

    @classmethod
    def as_keys(cls) -> List[str]:
      return [content.value for content in ContentType]


class RiskPattern(Enum):
    DIVISIVE_CONTENT = "Divisive_Content"
    RADICALIZATION = "Radicalization_Patterns"
    HARASSMENT = "Harassment_Indicators"
    SPAM = "Spam_Patterns"
    SCAM = "Scam_Indicators"
    DATA_MINING = "Data_Mining_Attempts"
    COORDINATED_BEHAVIOR = "Coordinated_Behavior"
    BAN_EVASION = "Ban_Evasion_Attempts"
    TESTING_BOUNDARIES = "Testing_Boundaries"
    MALICIOUS_LINKS = "Malicious_Links/External_Redirection"
    POLICY_CIRCUMVENTION = "Content_Policy_Circumvention"
    UNCLASSIFIED = "Unclassified"
    NONE = "None"

    @classmethod
    def get_prompt_format(cls) -> str:
        return "\n".join([f"- {member.value}" for member in cls])
    
    @classmethod
    def as_keys(cls) -> List[str]:
      return [pattern.value for pattern in RiskPattern]

