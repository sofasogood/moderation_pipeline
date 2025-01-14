from dataclasses import dataclass    
from typing import Optional   

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
        description="Harmful content that does not fit into known categories"
    )
    N = Category(
        label="N",
        category_name="none",
        description="Content contains some keywords that feature in harmful content, but is not classified as harmful."
    )



    def as_list(self):
      categories = []
      for attr_name in dir(self):
        if not attr_name.startswith('_'):
            attr = getattr(self, attr_name)
            if isinstance(attr, Category):
                categories.append(attr)
      return categories

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
      
