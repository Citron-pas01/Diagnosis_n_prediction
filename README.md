Data from 121 monitoring points is included in the database, with one point recorded every 15 seconds from January 2017 to January 2019.

Data processing and analysis are conducted through MySQL by fetching data. (Currently fetching 1000 rows temporarily)

Based on the physical relevance of the high-pressure cylinder structure and parameters, select data from 20 parameters as shown in the right figure.

Calculate the correlation of data belonging to multiple monitoring points, and remove duplicate monitoring point data.

Extract statistical parameters describing Max, Min, Mean, and Std features for each monitoring point data.

Standardize the data in each column (dimensionless) and perform outlier detection.

Select t-SNE (currently the best high-dimensional data dimensionality reduction and visualization algorithm) to transform the 20-dimensional feature parameters before and after K-means clustering into two dimensions.





