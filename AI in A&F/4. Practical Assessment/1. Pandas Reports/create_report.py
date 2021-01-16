import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

listing_df = pd.read_csv('listings.csv')
reviews_df = pd.read_csv('reviews.csv')

profile = ProfileReport(listing_df)
profile.to_file("listing_report.html")
profile = ProfileReport(reviews_df)
profile.to_file("reviews_report.html")