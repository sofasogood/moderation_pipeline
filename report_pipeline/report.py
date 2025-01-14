from datetime import datetime
from .categories import Labels 
import pandas as pd
from typing import List, Tuple, Union, Callable
REPORT_VERSION="v1.0" # Initial report version

class ReportGenerator:
  """
  Main report generator class that takes in a dataframe and generates a .txt report"""
  def __init__(self, content_data: pd.DataFrame, report_version: str = REPORT_VERSION, report_date: str = None):
    self._data = content_data ## can I add a better try-catch
    self._report_version = report_version
    self._report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  @property
  def report_version(self):
    return self._report_version
  
  @property
  def report_date(self):
    return self._report_date
  ##get latest classification, remove dupes
  
  def generate_summary_statistics(self):
    category_labels = Labels()
    label_keys, category_names, descriptions = category_labels.metadata_as_list()
    
    # Calculate the percentage of 1s in each column (excluding NaN)
    percentages = (self._data[label_keys] == 1.0).mean() * 100 ##Nas need to be filled
    
    # Create a DataFrame with labels and their percentages
    stats_df = pd.DataFrame({
        'category_name': category_names,
        'label': label_keys,
        'percent': percentages.round(1),
        'description': descriptions


    })
    
    
    
    return stats_df.sort_values(by="percent", ascending=False)

  def generate_report(self,     
                        title: str, 
                        sections: List[Tuple[str, Union[pd.DataFrame, str, Callable[[], str]]]] | None = None, 
                        output_path: str = 'report.txt'):
      # convert stats to string and add formatting
      # optionally print out
      # write to .txt

      ## generate data
      if sections is None:
        summary_stats = self.generate_summary_statistics()
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

