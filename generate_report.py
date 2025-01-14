from report_pipeline.content import ContentDataSet
from report_pipeline.classification import ContentClassifier
from report_pipeline.report import ReportGenerator
if __name__=="__main__":
   data = ContentDataSet()
   data.load_from_file()

   classifier = ContentClassifier()
   data.update_classifications(classifier, index_range=(0,3))
   df = data.to_pandas()
   ReportGenerator(df).generate_report(title="Daily Report")


