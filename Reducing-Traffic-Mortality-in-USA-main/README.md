Reducing Traffic Mortality in the USA

How can we find a good strategy for reducing traffic-related deaths?

Project Description
While the rate of fatal road accidents has been decreasing steadily since the 80s, the past ten years have seen a stagnation in this reduction. Coupled with the increase in number of miles driven in the nation, the total number of traffic related-fatalities has now reached a ten year high and is rapidly increasing.

By looking at the demographics of traﬃc accident victims for each US state, we find that there is a lot of variation between states. Now we want to understand if there are patterns in this variation in order to derive suggestions for a policy action plan. In particular, instead of implementing a costly nation-wide plan we want to focus on groups of states with similar profiles. How can we find such groups in a statistically sound way and communicate the result effectively?

Project Tasks
1. The raw data files and their format
2. Read in and get an overview of the data
3. Create a textual and a graphical summary of the data
4. Quantify the association of features and accidents
5. Fit a multivariate linear regression
6. Perform PCA on standardized data
7. Visualize the first two principal components
8. Find clusters of similar states in the data
9. KMeans to visualize clusters in the PCA scatter plot
10. Visualize the feature differences between the clusters
11. Compute the number of accidents within each cluster
12. Make a decision when there is no clear right choice



1. The raw data files and their format
 
While the rate of fatal road accidents has been decreasing steadily since the 80s, the past ten years have seen a stagnation in this reduction. Coupled with the increase in number of miles driven in the nation, the total number of traffic related-fatalities has now reached a ten year high and is rapidly increasing.
Per request of the US Department of Transportation, we are currently investigating how to derive a strategy to reduce the incidence of road accidents across the nation. By looking at the demographics of traﬃc accident victims for each US state, we find that there is a lot of variation between states. Now we want to understand if there are patterns in this variation in order to derive suggestions for a policy action plan. In particular, instead of implementing a costly nation-wide plan we want to focus on groups of states with similar profiles. How can we find such groups in a statistically sound way and communicate the result effectively?
To accomplish these tasks, we will make use of data wrangling, plotting, dimensionality reduction, and unsupervised clustering.
The data given to us was originally collected by the National Highway Traffic Safety Administration and the National Association of Insurance Commissioners. This particular dataset was compiled and released as a CSV-file by FiveThirtyEight under the CC-BY4.0 license.



Task 1: Instructions
Explore your current folder and view the main dataset file.

Check the name of the current folder using !pwd.
List all files in this folder using !ls.
List all files in the datasets\ folder using !ls and the name of the folder.
View the first 20 lines of road-accidents.csv in the datasets\ folder using !head.



2. Read in and get an overview of the data
Next, we will orient ourselves to get to know the data with which we are dealing.

Task 2: Instructions
Read in the main dataset file and start exploring the data.

Import the pandas module aliased as pd.
Read in road-accidents.csv (which is in the datasets/ folder) using read_csv() from pandas. Set the comment and sep parameters based on the output from task 1.
Save the number of rows columns as a tuple, using the shape attribute.
Generate an overview of the DataFrame using the info() method.

3. Create a textual and a graphical summary of the data
We now have an idea of what the dataset looks like. To further familiarize ourselves with this data, we will calculate summary statistics and produce a graphical overview of the data. The graphical overview is good to get a sense for the distribution of variables within the data and could consist of one histogram per column. It is often a good idea to also explore the pairwise relationship between all columns in the data set by using a using pairwise scatter plots (sometimes referred to as a "scatterplot matrix").


Task 3: Instructions

Create a textual and graphical overview of the data.

Compute the summary statistics of all columns in the car_acc DataFrame, using the describe() method.
Create a pairwise scatter plot to explore the data, using sns.pairplot().
4. Quantify the association of features and accidents
We can already see some potentially interesting relationships between the target variable (the number of fatal accidents) and the feature variables (the remaining three columns).
To quantify the pairwise relationships that we observed in the scatter plots, we can compute the Pearson correlation coefficient matrix. The Pearson correlation coefficient is one of the most common methods to quantify correlation between variables, and by convention, the following thresholds are usually used:
•	0.2 = weak
•	0.5 = medium
•	0.8 = strong
•	0.9 = very strong

Task 4: Instructions
Explore the correlation between all column pairs in the DataFrame.

Compute the correlation coefficient for all column pairs in car_acc, using the corr() method.

5. Fit a multivariate linear regression
From the correlation table, we see that the amount of fatal accidents is most strongly correlated with alcohol consumption (first row). But in addition, we also see that some of the features are correlated with each other, for instance, speeding and alcohol consumption are positively correlated. We, therefore, want to compute the association of the target with each feature while adjusting for the effect of the remaining features. This can be done using multivariate linear regression.
Both the multivariate regression and the correlation measure how strongly the features are associated with the outcome (fatal accidents). When comparing the regression coefficients with the correlation coefficients, we will see that they are slightly different. The reason for this is that the multiple regression computes the association of a feature with an outcome, given the association with all other features, which is not accounted for when calculating the correlation coefficients.
A particularly interesting case is when the correlation coefficient and the regression coefficient of the same feature have opposite signs. How can this be? For example, when a feature A is positively correlated with the outcome Y but also positively correlated with a different feature B that has a negative effect on Y, then the indirect correlation (A->B->Y) can overwhelm the direct correlation (A->Y). In such a case, the regression coefficient of feature A could be positive, while the correlation coefficient is negative. This is sometimes called a masking relationship. Let’s see if the multivariate regression can reveal such a phenomenon.

Task 5: Instructions
Fit a multivariate linear regression model using the fatal accident rate as the outcome.

Import the linear_model function from sklearn.
Create the features and target DataFrames, by subsetting the DataFrame car_acc.
Create a linear regression object, using linear_model.LinearRegression().
Fit a multivariate linear regression model, using fit().
Retrieve the regression coefficients from the coef_ attribute of the fitted regression object.

6. Perform PCA on standardized data
We have learned that alcohol consumption is weakly associated with the number of fatal accidents across states. This could lead us to conclude that alcohol consumption should be a focus for further investigations and maybe strategies should divide states into high versus low alcohol consumption in accidents. But there are also associations between alcohol consumptions and the other two features, so it might be worth trying to split the states in a way that accounts for all three features.
One way of clustering the data is to use PCA to visualize data in reduced dimensional space where we can try to pick up patterns by eye. PCA uses the absolute variance to calculate the overall variance explained for each principal component, so it is important that the features are on a similar scale (unless we would have a particular reason that one feature should be weighted more).
We'll use the appropriate scaling function to standardize the features to be centered with mean 0 and scaled with standard deviation 1.


Task 6: Instructions
Perform a principal component analysis on the standardized data.

Standardize and center the feature columns, using the StandardScaler from sklearn and its fit_transform() method.
Import the PCA class from sklearn.
Fit the standardized data to the PCA class using its fit() method.
Compute the cumulative proportion of variance explained by the first two principal components, either by adding them together or by using the cumulative summation method (cumsum) of the explained variance array.

7. Visualize the first two principal components
The first two principal components enable visualization of the data in two dimensions while capturing a high proportion of the variation (79%) from all three features: speeding, alcohol influence, and first-time accidents. This enables us to use our eyes to try to discern patterns in the data with the goal to find groups of similar states. Although clustering algorithms are becoming increasingly efficient, human pattern recognition is an easily accessible and very efficient method of assessing patterns in data.
We will create a scatter plot of the first principle components and explore how the states cluster together in this visualization.
Task 7: Instructions
Transform the data and visualize the first two principal components in a scatter plot.

Create a PCA object with two components. Assign the result to the variable, pca.
Transform the scaled features using two principal components and the fit_transform() method of the PCA object.
Extract the first and second component to use for the scatter plot. Assign the results to p_comp1 and p_comp2, respectively.
Plot the first two principal components in a scatter plot, using plt.scatter.
The n_components parameter controls the number of components with which to initialize the PCA class.
8. Find clusters of similar states in the data
It was not entirely clear from the PCA scatter plot how many groups in which the states cluster. To assist with identifying a reasonable number of clusters, we can use KMeans clustering by creating a scree plot and finding the "elbow", which is an indication of when the addition of more clusters does not add much explanatory power.

Task 8: Instructions
Cluster the states using the KMeans algorithm and visualize the explanatory power for different numbers of clusters.

Import KMeans from sklearn.cluster.
Initialize the KMeans object using the current number of clusters (k).
Fit the scaled features to the KMeans object.
Append the inertia for km to the list of inertias.
Plot the results in a line plot using matplotlib.pyplot.plot. This type of plot is also called a scree plot.

9. KMeans to visualize clusters in the PCA scatter plot
Since there wasn't a clear elbow in the scree plot, assigning the states to either two or three clusters is a reasonable choice, and we will resume our analysis using three clusters. Let's see how the PCA scatter plot looks if we color the states according to the cluster to which they are assigned.

Task 9: Instructions
Highlight the clusters of the K-means fit with three clusters in the PCA scatter plot.

Create a KMeans object with 3 clusters, setting random_state to 8 as in the previous task.
Fit the data to the km object.
Create a scatter plot of the first two principal components and color it according to the KMeans cluster assignment.
10. Visualize the feature differences between the clusters
Thus far, we have used both our visual interpretation of the data and the KMeans clustering algorithm to reveal patterns in the data, but what do these patterns mean?
Remember that the information we have used to cluster the states into three distinct groups are the percentage of drivers speeding, under alcohol influence and that has not previously been involved in an accident. We used these clusters to visualize how the states group together when considering the first two principal components. This is good for us to understand structure in the data, but not always easy to understand, especially not if the findings are to be communicated to a non-specialist audience.
A reasonable next step in our analysis is to explore how the three clusters are different in terms of the three features that we used for clustering. Instead of using the scaled features, we return to using the unscaled features to help us interpret the differences.

Task 10: Instructions
Visualize the distribution of speeding, alcohol influence and percentage of first-time accidents in a direct comparison of the clusters.

Create a new column with the labels from the KMeans clustering, using km.labels_.
Reshape the DataFrame to the long format, using pd.melt(). Use the features as the value variables and give them the name 'measurement' in the new DataFrame. Name the value column 'percent'.
Create a violin plot splitting and coloring the results according to the km-clusters using the hue parameter. Plot the measurements along the y-axis and the percent values along the x-axis.

11. Compute the number of accidents within each cluster
Now it is clear that different groups of states may require different interventions. Since resources and time are limited, it is useful to start off with an intervention in one of the three groups first. Which group would this be? To determine this, we will include data on how many miles are driven in each state, because this will help us to compute the total number of fatal accidents in each state. Data on miles driven is available in another tab-delimited text file. We will assign this new information to a column in the DataFrame and create a violin plot for how many total fatal traffic accidents there are within each state cluster.
In [11]:

Task 11: Instructions
Add data on the number of miles driven per state to compute total number of fatal accidents and total accidents for each cluster.

Merge the car_acc DataFrame with the miles_driven DataFrame. Merge on the common column state.
Create a new column for the number of drivers involved in fatal accidents. Use the columns 'drvr_fatl_col_bmiles' and 'million_miles_annually', note that these are in billions and million respectively.
Calculate the number of states in each cluster and their average and total number of drivers involved in fatal accidents using the DataFrame agg() method.
Using sns.barplot, create a bar plot of the total number of fatal accidents per cluster, setting the estimator parameter to sum.

12. Make a decision when there is no clear right choice
As we can see, there is no obvious correct choice regarding which cluster is the most important to focus on. Yet, we can still argue for a certain cluster and motivate this using our findings above. Which cluster do you think should be a focus for policy intervention and further investigation?
Task 12: Instructions
Decide which cluster to focus your resources on.

Which cluster would you choose: 1, 2, or 3? Assign one of these integers to cluster_num.




