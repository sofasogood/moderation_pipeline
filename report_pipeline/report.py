from datetime import datetime
from .categories import Labels 
import pandas as pd
from typing import List, Tuple, Union, Callable
from .classification import CLASSIFIER_VERSION
REPORT_VERSION="v1.0" # Initial report version

class ReportGenerator:
  """
  Main report generator class that takes in a dataframe and generates a .txt report"""
  def __init__(self, content_data: pd.DataFrame, report_version: str = REPORT_VERSION, report_date: str = None):
    self._data = content_data ## 
    self._report_version = report_version
    self._report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  @property
  def report_version(self):
    return self._report_version
  
  @property
  def report_date(self):
    return self._report_date
  ##get latest classification, remove dupes
  def get_latest_classifications(self):
    return self._data.drop_duplicates(subset=["content_id"], keep="last")
  
  def generate_summary_statistics(self, keep_last_classification: bool = True):
    category_labels = Labels()
    label_keys, category_names, descriptions = category_labels.metadata_as_list()

    if keep_last_classification:
      data = self.get_latest_classifications()
      # Calculate the percentage of 1s in each column (excluding NaN)
      count = (data[label_keys] == 1.0).sum()  ##Nas need to be filled
    
    # Create a DataFrame with labels and their percentages
      stats_df = pd.DataFrame({
          'category_name': category_names,
          'label': label_keys,
          'count': count,
          'description': descriptions,
      })
      return stats_df.sort_values(by="count", ascending=False)
    else:
        data = self._data
        stats_dfs = []
        for version in data["classifier_version"].unique():
            data_subset = data[data["classifier_version"] == version]
            # Calculate the percentage of 1s in each column (excluding NaN)
            count = (data_subset[label_keys] == 1.0).sum() ##Nas need to be filled
              # Create a DataFrame for this version
            version_stats_df = pd.DataFrame({
                'category_name': category_names,
                'label': label_keys,
                f'count_{version}': count,
                'description': descriptions,
            })
            stats_dfs.append(version_stats_df)
    
        combined_stats_df = stats_dfs[0]
        # Combine all version stats
        for stats_df in stats_dfs[1:]:
          combined_stats_df = combined_stats_df.merge(stats_df, on=["label", "category_name", "description"])
        final_cols = self.rearrange_columns(combined_stats_df, stat = "count")
    combined_stats_df[final_cols].sort_values(by=[f"count_{CLASSIFIER_VERSION}"], ascending=False).to_csv("summary_stats.csv")
    return combined_stats_df[final_cols].sort_values(by=[f"count_{CLASSIFIER_VERSION}"], ascending=False)

    
  def rearrange_columns(self, df: pd.DataFrame, stat: str = "count") -> List[str]:
    """Rearrange columns in a DataFrame."""
        # Reorder columns to group percent columns together
    percent_cols = [col for col in df.columns if col.startswith(stat+ '_')]
    other_cols = ['category_name', 'label']
    final_cols = other_cols + percent_cols + ['description']

    return final_cols
  
  def generate_report(self,     
                        title: str, 
                        sections: List[Tuple[str, Union[pd.DataFrame, str, Callable[[], str]]]] | None = None, 
                        output_path: str = 'report.txt',
                        keep_last_classification: bool = True):
      # convert stats to string and add formatting
      # optionally print out
      # write to .txt

      ## generate data
      if sections is None:
        summary_stats = self.generate_summary_statistics(keep_last_classification=keep_last_classification)
        sections = [("Summary Statistics", summary_stats)]

      with open(output_path, "w") as f:
        # Write title
        f.write(f"{title}\n")
        f.write("="*len(title)+"\n")
        f.write(f"Report Date: {self.report_date}\n")
        f.write(f"Report Version: {self.report_version}\n\n")



        for section_title, section_content in sections:
          f.write(f"{section_title}\n")
          f.write("-" * len(section_title) + "\n\n")
            # Handle different content types
          if isinstance(section_content, pd.DataFrame):
              # Convert DataFrame to string with nice formatting
              df_str = section_content.to_string(
                  index=True,
                  float_format=lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else str(x)
              )
              f.write(df_str + "\n\n")

