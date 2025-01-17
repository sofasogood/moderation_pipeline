## Abuse Vector Reporting Pipeline

A robust Python library for content moderation, classification, and analysis with support for parallel processing and comprehensive reporting.
Features

- 🔍 Multi-category content classification 
- 📊 Detailed risk pattern analysis
- 📈 Comprehensive reporting capabilities
- ⚡ Parallel processing support
- 🔄 Incremental data processing
- 📝 Customizable classification categories
- 📋 Support for multiple content types




The dataset `data/samples-1680.jsonl.gz` is the test set used in [the following paper](https://arxiv.org/abs/2208.03274):

```
@article{openai2022moderation,
  title={A Holistic Approach to Undesired Content Detection},
  author={Todor Markov and Chong Zhang and Sandhini Agarwal and Tyna Eloundou and Teddy Lee and Steven Adler and Angela Jiang and Lilian Weng},
  journal={arXiv preprint arXiv:2208.03274},
  year={2022}
}
```
## Instructions
The library is developed using Python 3.13.1.

0) Copy and rename .env.example and fill in your API key file, don't add quotes. This is critical!
```
OPEN_AI_API_KEY=sk....
```

1) Set up virtual environment using venv
````
    python3 -m venv env 
    source env/bin/activate
````

2) Install dependencies
```
    pip3 install -r requirements.txt
```

3) Generate report. Note: This will take ~3-4 minutes to classify all the prompts and generate a text file with the analysis, which automatically includes new data when added to the json without reclassifying events with labels.

```
    python -m generate_report 
```


## Key Concepts
```
graph LR
    A[Input Data] --> B[ContentDataSet]
    B --> C[ContentClassifier]
    C --> D[Classification Results]
    D --> E[ReportGenerator]
    E --> F[Analysis Reports]
  ```

The library is organized into several key components:

- Categories: Defines the taxonomy and classification schema

| Category | Label | Definition |
| -------- | ----- | ---------- |
| sexual   | `S`   | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
| hate     | `H`   | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. |
| violence | `V`   | Content that promotes or glorifies violence or celebrates the suffering or humiliation of others. |
| harassment       | `HR`   | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. |
| self-harm        | `SH`   | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. |
| sexual/minors    | `S3`   | Sexual content that includes an individual who is under 18 years old. |
| hate/threatening | `H2`   | Hateful content that also includes violence or serious harm towards the targeted group. |
| violence/graphic | `V2`   | Violent content that depicts death, violence, or serious physical injury in extreme graphic detail. |
| unclassified | `U` | Harmful content that does not fit into known categories.
| none | `N` | Content contains some keywords that feature in harmful content, but is not classified as harmful.

- Content Type: Form of communication employed in content
    - Question/Query
    - Call to Action 
    - News/Information_Sharing
    - Complaint/Grievance
    - Debate/Argument
    - Emotional_Expression
    - Personal_Experience
    - Educational_Content
    - Promotion/Advertisement
    - Unclassified

- Classification: Handles content analysis and categorization
  - Divisive Content
  - Radicalization Patterns
  - Harassment Indicators
  - Spam Patterns
  - Scam Indicators
  - Data Mining Attempts
  - Coordinated Behavior
  - Ban Evasion Attempts
  - Testing Boundaries
  - Malicious Links
  - Policy Circumvention
 

- Report: Generates comprehensive analysis reports


## Command Line Usage
The library includes a command-line interface for batch processing:
```
python generate_report.py \
    --input-file data/samples.jsonl.gz \
    --pickle-path data_latest.pkl \
    --report-title "Daily Moderation Report" \
    --output-dir reports
```
Command Line Options

`--input-file`: Path to input data file (default: data/samples-1680.jsonl.gz)

`--pickle-path`: Path for data serialization (default: data_latest.pkl)

`--report-title`: Custom report title

`--keep-last-classification`: Only use most recent classifications

`--skip-classification`: Skip classification step

`--output-dir`: Output directory for reports