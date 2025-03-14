from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from tqdm import tqdm
from report_pipeline.content.dataset import ContentDataSet
from report_pipeline.classification.utils.base import BaseClassifier
from datetime import datetime, timedelta
from threading import Lock
import time

def _process_batch(contents_batch, classifier_class, classifier_kwargs, force_reclassify, rate_limiter):
    local_clf = classifier_class(**classifier_kwargs)
    for content in contents_batch:
        with rate_limiter:
            if force_reclassify or _needs_classification(content, local_clf):
                new_clf = local_clf.classify_content(content)
                content.add_classification(new_clf)
    return contents_batch


def update_classifications_in_parallel(
    content_dataset: ContentDataSet,
    classifier: BaseClassifier,
    force_reclassify: bool = False,
    batch_size: int = 10,
    max_workers: int = 4,
    requests_per_minute: int = 2000,
) -> None:
    """
    Example function that processes items in parallel.
    """
    all_contents = content_dataset.all_items()
    num_items = len(all_contents)
    num_batches = math.ceil(num_items / batch_size)

    # Prepare data for workers
    classifier_class = classifier.__class__
    classifier_kwargs = classifier.get_init_kwargs()

    # Create rate limiter to manage requests per minute
    rate_limiter = RateLimiter(requests_per_minute)



    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_items)
            batch = all_contents[start:end]
            futures[executor.submit(_process_batch, batch, classifier_class, classifier_kwargs, force_reclassify, rate_limiter)] = (start, end)

        with tqdm(total=num_items, desc="Classifying", unit="item") as pbar:
            for future in as_completed(futures):
                (start, end) = futures[future]
                contents_batch = future.result()
                # They’re already updated in-place, so just update progress
                pbar.update(len(contents_batch))

def _needs_classification(content, classifier):
    for clf in content.classifications:
        if clf.classifier_version == classifier._classifier_version:
            return False
    return True

class RateLimiter:
    """
    Rate limiter that restricts operations to a maximum number per minute.
    
    Implements the context manager protocol so it can be used with 'with' statements.
    """
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests in seconds
        self.last_request_times = []
        self.lock = Lock()
        
    def __enter__(self):
        """
        Block until a request can be made according to the rate limit.
        """
        with self.lock:
            # Clean up old timestamps that are no longer relevant (older than 1 minute)
            current_time = datetime.now()
            self.last_request_times = [t for t in self.last_request_times 
                                       if current_time - t < timedelta(minutes=1)]
            
            # If we've reached the limit, wait until we can make another request
            if len(self.last_request_times) >= self.requests_per_minute:
                # Calculate wait time based on the oldest request in our window
                wait_time = 60 - (current_time - self.last_request_times[0]).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Record this request
            self.last_request_times.append(datetime.now())
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        """
        pass

