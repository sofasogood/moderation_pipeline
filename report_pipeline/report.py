from datetime import datetime
from .categories import Labels 
import pandas as pd
from typing import List, Tuple, Union, Callable
from tabulate import tabulate
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

  def calculate_percent_change(self, old_value, new_value):
      percent_change = ((new_value - old_value) / old_value) * 100
      return percent_change.round(1)

  def extract_abuse_patterns(self, df: pd.DataFrame, severity_range: tuple = (4,5)):
      filter_severity = (df["severity_score"] >= severity_range[0]) & (df["severity_score"] <= severity_range[1])
      filtered_df = df[filter_severity].filter(like="RISK_PATTERN_", axis=1)
      result_series = filtered_df.mean() * 100
      
      # 3. Convert to DataFrame and clean up column names
      result_df = pd.DataFrame({
          'Pattern': result_series.index.str.replace('RISK_PATTERN_', ''),
          'Percentage': result_series.values
      })
    
      return result_df

        
  def generate_executive_summary(self, data: pd.DataFrame, summary_stats: pd.DataFrame, prev_day_data: pd.DataFrame = None) -> str:
    """
    Generate an executive summary highlighting critical findings and patterns.
    
    Args:
        data: Current day's classification data
        summary_stats: Summary statistics DataFrame
        prev_day_data: Previous day's classification data for comparison (optional)
    
    Returns:
        str: Formatted executive summary
    """
    # Calculate critical metrics
    critical_incidents = data[data["needs_human_review"] == True]["needs_human_review"].sum()
    unknown_abuse = data["U"].sum()
    unknown_content = data["CONTENT_TYPE_Unclassified"].sum()
    unknown_risk = data["RISK_PATTERN_Unclassified"].sum()
    
    # Identify top categories types, excluding "N"
    category_cols = [x for x in Labels().as_keys() if x!='N']
    top_category_types = data[category_cols].mean().sort_values(ascending=False).head(3)
  

    
    # Identify top risk patterns
    risk_pattern_cols = [col for col in data.columns if (col.startswith('RISK_PATTERN_') and (col!='RISK_PATTERN_None'))]
    top_risk_patterns = data[risk_pattern_cols].sum().sort_values(ascending=False).head(3)
    
    # Identify top content types
    content_type_cols = [col for col in data.columns if (col.startswith('CONTENT_TYPE_'))]
    top_content_types = data[content_type_cols].sum().sort_values(ascending=False).head(3)

    
    # Calculate significant changes
    significant_changes = []
    if prev_day_data is not None:
        for col in risk_pattern_cols:
            curr_count = data[col].sum()
            prev_count = prev_day_data[col].sum()
            if prev_count > 0:  # Avoid division by zero
                change_pct = ((curr_count - prev_count) / prev_count) * 100
                if abs(change_pct) >= 20:  # Threshold for significant change
                    significant_changes.append(f"{col.replace('RISK_PATTERN_', '')}: {change_pct:+.1f}%")
    
    # Format the executive summary
    summary = [
        "# Executive Summary\n",
        "CRITICAL ALERTS:",
        f"- {critical_incidents} critical severity incidents detected",
        f"- {unknown_abuse} incidents with Unclassified Harm Category",
        f"- {unknown_content} incidents with Unclassified Content Type",
        f"- {unknown_risk} incidents with Unclassified Risk Patterns",
    ]
    
    if significant_changes:
        summary.append("\nSIGNIFICANT CHANGES:")
        summary.extend([f"- {change}" for change in significant_changes])
    
    category_names = Labels().as_dict()
    summary.extend([
        "\nTOP CATEGORIES:",
        *[f"- {category_names[category]}: {percent*100:.0f} % of incidents" 
          for category, percent in top_category_types.items()]
    ])

    
    summary.extend([
        "\nTOP RISK PATTERNS:",
        *[f"- {pattern.replace('RISK_PATTERN_', '')}: {count:.0f} incidents" 
          for pattern, count in top_risk_patterns.items()]
    ])
    summary.extend([
        "\nTOP CONTENT TYPES:",
        *[f"- {content.replace('CONTENT_TYPE_', '')}: {count:.0f} incidents" 
          for content, count in top_content_types.items()]
    ])

    
    # Add recommended actions based on thresholds
    actions = []
    if unknown_abuse > 5:
        actions.append(f"- Escalate investigation of {unknown_abuse} Unclassified Abuse Category incidents.")

    if critical_incidents > 100:
        actions.append(f"- Review Critical incidents")
    if unknown_risk > 5:
        actions.append(f"- Analyze {unknown_risk} Unclassified Risk Patterns for emerging threats")
    if unknown_content > 5:
        actions.append(f"- Analyze {unknown_content} Unclassified Content Types for jailbreak risk")


    if actions:
        summary.extend(["\nRECOMMENDED ACTIONS:", *actions])
    
    return "\n".join(summary) + "\n"
  
  def analyze_critical_patterns(self, data: pd.DataFrame, 
                                min_critical_correlation: float = 0.7,
                                min_significant_correlation: float = 0.3) -> str:
      """
      Analyze patterns and format output.
      """
      # Create header
      header = ["Category", "Count", "Sev", "Content Types", "Critical Patterns", "Emerging"]

      # Create DataFrame for output
      rows = []

      # Helper to create bullet lists
      def make_text_bullets(items):
          if not items:
              return ""
          return "\n".join(f"- {item}" for item in items)

      
      # Map short codes to full names
      category_names = Labels().as_dict()

      # Process each category
      for category in sorted(Labels().as_keys()):
          if category == 'N':
              continue
              
          category_data = data[data[category] == 1]
          count = len(category_data)
          if count == 0:
              continue

          # Basic category info
          severity_score = category_data['severity_score'].mean()
          print(severity_score)
          severity_level = "HIGH" if severity_score > 4.0 else "MED" if severity_score > 3.5 else "LOW"
          
          # Get content types
          content_types = []
          for ct in [col for col in data.columns if col.startswith('CONTENT_TYPE_')]:
              corr = (category_data[ct].sum() / count) if count > 0 else 0
              if corr > min_significant_correlation:
                  content_types.append(f"- {ct.replace('CONTENT_TYPE_', '')}")
          content_type_text = make_text_bullets(content_types)


          # Get patterns
          critical_patterns = []
          for pattern in [col for col in data.columns if col.startswith('RISK_PATTERN_')]:
              pattern_name = pattern.replace('RISK_PATTERN_', '')
              pattern_count = category_data[pattern].sum()
              if pattern_count > 0:
                  correlation = pattern_count / count
                  if correlation >= min_critical_correlation:
                      critical_patterns.append(f"- **{pattern_name}**({correlation*100:.0f}%)")
                  elif correlation >= min_significant_correlation:
                      critical_patterns.append(f"- {pattern_name}({correlation*100:.0f}%)")
          critical_patterns_text = make_text_bullets(critical_patterns)

          # Get emerging patterns
          emerging_patterns = []
          for pattern in [col for col in data.columns if col.startswith('RISK_PATTERN_')]:
              pattern_name = pattern.replace('RISK_PATTERN_', '')
              pattern_count = category_data[pattern].sum()
              if pattern_count <= 3 and category_data[category_data[pattern] == 1]['severity_score'].mean() > 0.7:
                  emerging_patterns.append(f"* [!]{pattern_name}")
          emerging_text = make_text_bullets(emerging_patterns)

        # Add row
          rows.append([
            category_names.get(category, category),
            count,
            severity_level,
            content_type_text,
            critical_patterns_text,
            emerging_text
        ])
    
    # Convert to DataFrame
      df = pd.DataFrame(rows, columns=header)
      
      result_str = tabulate(df, headers=header, tablefmt="fancy_grid", showindex=False)
      return result_str

      
  def generate_report(self,     
                          title: str, 
                          sections: List[Tuple[str, Union[pd.DataFrame, str]]] | None = None, 
                          output_path: str = 'report.txt',
                          keep_last_classification: bool = True):

        data = self.get_latest_classifications()
        ## generate data
        if sections is None:
          summary_stats = self.generate_summary_statistics(keep_last_classification=keep_last_classification)
          #sections = [("Summary Statistics", summary_stats), ("Critical Severity Events", self.extract_abuse_patterns(data))] ## add category vs pattern heatmap
          risk_pattern_cols = [col for col in data.columns if col.startswith('RISK_PATTERN_')]
          category_cols = [col for col in data.columns if col in Labels().as_keys()]  # Add other category names as needed
          
          sections = [
              ("Executive Summary", self.generate_executive_summary(data, summary_stats)),
              ("Detailed Analysis: Critical and Emerging Patterns", self.analyze_critical_patterns(data)),
              ("Unclassified Harm Category Incidents", 
              f"Content IDs: \n{'\n'.join(map(str, data[data['U']==1.0]['content_id'].to_list()))}"),
              ("Unclassified Risk Pattern Incidents", 
              f"Content IDs: \n{'\n'.join(map(str, data[data['RISK_PATTERN_Unclassified']==1.0]['content_id'].to_list()))}")
          ]


        with open(output_path, "w") as f:
          # Write title
          f.write(f"{title}\n")
          f.write("="*len(title)+"\n")
          f.write(f"Report Date: {self.report_date}\n")
          f.write(f"Report Version: {self.report_version}\n\n")
          if not keep_last_classification:
            legacy_classifications = summary_stats["count_legacy"].sum()
            new_classifications = summary_stats[f"count_{CLASSIFIER_VERSION}"].sum()
            f.write(f"{new_classifications - legacy_classifications} New Content Classifications: {new_classifications - legacy_classifications}, {self.calculate_percent_change(legacy_classifications, new_classifications)} % change")

          for section_title, section_content in sections:
            f.write(f"\n{section_title}\n")
            f.write("-" * len(section_title) + "\n\n")
            if isinstance(section_content, pd.DataFrame):

                # Convert DataFrame to string with nice formatting
                df_str = section_content.to_string(
                    index=True,
                    float_format=lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else str(x)
                )
                f.write(df_str + "\n\n")
              

            elif isinstance(section_content, str):
                # Write string content directly
                f.write(section_content + "\n\n")
                
