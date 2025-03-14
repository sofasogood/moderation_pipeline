import os
import argparse
from pathlib import Path

from report_pipeline.content.dataset import ContentDataSet
from report_pipeline.classification.classification import ContentClassifier
from report_pipeline.classification.classifiers.openai import OpenAIClassifier, OpenAIClassifierConfig
from report_pipeline.classification.utils.prompt_generator import PromptGenerator
from report_pipeline.classification.classification_runner import update_classifications_in_parallel
from report_pipeline.report.report import ReportGenerator
from dotenv import load_dotenv

load_dotenv(override=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process content data and generate reports")
    
    # Data paths
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/samples-1680.jsonl.gz",
        help="Path to input data file (default: data/samples-1680.jsonl.gz)"
    )
    parser.add_argument(
        "--pickle-path",
        type=str,
        default="data_latest.pkl",
        help="Path to pickle file for loading/saving data (default: data_latest.pkl)"
    )
    
    # File type
    parser.add_argument(
        "--file-type",
        type=str,
        default="jsonl.gz",
        choices=["jsonl.gz", "json", "csv"],
        help="Type of input file (default: jsonl.gz)"
    )
    
    # Report options
    parser.add_argument(
        "--report-title",
        type=str,
        default="Daily Report",
        help="Title for the generated report (default: 'Daily Report')"
    )
    parser.add_argument(
        "--keep-last-classification",
        action="store_true",
        help="Keep only the last classification in the report"
    )
    
    # Additional options
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip the classification step"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # # Construct the full paths
    input_file = os.path.join(os.getcwd(), args.input_file)
    pickle_path = os.path.join(output_dir, args.pickle_path)
    
    data = ContentDataSet()
    # Initialize and load data
    if os.path.exists(pickle_path):
        data.load_from_pickle(pickle_path=pickle_path)
    else:
        data.load_from_dataset(name = "google/civil_comments", prompt_col = "text", random_sample=3000)
    
    # Run classification if not skipped
    if not args.skip_classification:
        model = OpenAIClassifier(OpenAIClassifierConfig(model_name="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]))
        prompt_generator = PromptGenerator()
        classifier = ContentClassifier(model, prompt_generator, version="v1")
        update_classifications_in_parallel(data, classifier, max_workers=16, batch_size=5)

    # Save processed data
    data.save_to_pickle(
        pickle_path=os.path.join(output_dir, "data.pkl"),
        create_latest_link=True
    )

   # Generate report
    df = data.to_pandas()
    df.to_csv('df.csv')
    ReportGenerator(df).generate_report(
        title=args.report_title,
        keep_last_classification=args.keep_last_classification
    )


if __name__ == "__main__":
    main()