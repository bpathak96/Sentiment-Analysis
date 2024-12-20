## 1. Install libraries

%pip install wordcloud matplotlib
%pip install gensim

# Restart the kernel
dbutils.library.restartPython()

# Import libraries
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, lower, avg, stddev
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#-------------------------------------------
## 2. Load SA table

# Load the SA DataFrame from the registered table
sa_table = (
    spark.read.format("delta")
    .option("header", "true")
    .load("dbfs:/FileStore/tables/john_marshman/Bhavya_pathak/sa_table"))

# Verify by showing the schema or some rows
sa_table.printSchema()
sa_table.limit(10).display()

# Display the number of rows and columns
row_num = sa_table.count()
col_num = len(sa_table.columns)

print(f"Number of rows are: {row_num} and number of columns are: {col_num}")


#-------------------------------------------
## 3. Prepare the table

# Standardize the Call_Type column to lower case in Spark DataFrame
sa_table = sa_table.withColumn('Call_Type', lower(col('Call_Type')))

# Create a score difference column
sa_table = sa_table.withColumn('Score_Difference', col('Survey_Score') - col('Predicted_Sentiment_Score'))

# Show the updated DataFrame and the new differences column
sa_table.limit(10).display()

# Convert Spark DataFrame to Pandas DataFrame for WordCloud
sa_table_pd = sa_table.toPandas()

# Group feedback based on sentiment scores
negative_feedback = sa_table_pd[sa_table_pd['Predicted_Sentiment_Score'] <= 2]['Expanded_Comment']
positive_feedback = sa_table_pd[sa_table_pd['Predicted_Sentiment_Score'] >= 4]['Expanded_Comment']


#-------------------------------------------
## 4. Analysis and Interpretation

# Distribution Analysis of Score Differences
score_difference_distribution = sa_table.groupBy('Score_Difference') \
    .count() \
    .withColumn('Percentage', round((col('count') / row_num) * 100, 2)) \
    .orderBy(col('Score_Difference'))

display(score_difference_distribution)

### 4.1. Create 4 word clouds for score difference -4, -3, 4 and 3
# Define a function to create a word cloud for specific score differences
def create_word_cloud(df, score_diff, title):
    comments = " ".join(df[df["Score_Difference"] == score_diff]["Expanded_Comment"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()

# Create word clouds for the specified score differences
create_word_cloud(sa_table_pd, -4, "Word Cloud for Score Difference -4")
create_word_cloud(sa_table_pd, 4, "Word Cloud for Score Difference 4")
create_word_cloud(sa_table_pd, -3, "Word Cloud for Score Difference -3")
create_word_cloud(sa_table_pd, 3, "Word Cloud for Score Difference 3")


# 4.2. Create 2 word clouds for score difference (-4, -3) and (4, 3)
# Function to create n-gram word cloud
def create_ngram_word_cloud(df, score_diffs, title, n):
    comments = " ".join(df[df["Score_Difference"].isin(score_diffs)]["Expanded_Comment"])
    vectorizer = CountVectorizer(ngram_range=(n, n)).fit([comments])
    ngrams = vectorizer.transform([comments])
    ngram_freq = dict(zip(vectorizer.get_feature_names_out(), ngrams.toarray().sum(axis=0)))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()

# Display unigram word clouds for the specified score differences
# Create word clouds for the combined score differences
create_ngram_word_cloud(sa_table_pd, [-4, -3], "Unigram Word Cloud for Score Differences -4 and -3", 1)
create_ngram_word_cloud(sa_table_pd, [4, 3], "Unigram Word Cloud for Score Differences 4 and 3", 1)

# Display bigram word clouds for the specified score differences
create_ngram_word_cloud(sa_table_pd, [-4, -3], "Bigram Word Cloud for Score Differences -4 and -3", 2)
create_ngram_word_cloud(sa_table_pd, [4, 3], "Bigram Word Cloud for Score Differences 4 and 3", 2)

# Display trigram word clouds for the specified score differences
create_ngram_word_cloud(sa_table_pd, [-4, -3], "Trigram Word Cloud for Score Differences -4 and -3", 3)
create_ngram_word_cloud(sa_table_pd, [4, 3], "Trigram Word Cloud for Score Differences 4 and 3", 3)

# Printing the comments with score difference of -4 and -3
sa_table.filter((col('score_difference') == -4) | (col('score_difference') == -3)).limit(1000).select('Comment', 'Expanded_Comment','score_difference', 'Survey_Score', 'Predicted_Sentiment_Score').limit(10).display()

# Printing comment with score difference 4 and 3
sa_table.filter((col('score_difference') == 4) | (col('score_difference') == 3)).limit(1000).select('Comment', 'Department', 'Agent_Username', 'Expanded_Comment', 'score_difference', 'Survey_Score', 'Predicted_Sentiment_Score').limit(10).display()


# 4.3. Extract common themes or topics
def extract_common_words(feedback, n=20):
    vectorizer = CountVectorizer(stop_words='english')
    word_count = vectorizer.fit_transform(feedback)
    word_count_sum = word_count.sum(axis=0)
    words_freq = [(word, word_count_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

negative_common_words = extract_common_words(negative_feedback)
positive_common_words = extract_common_words(positive_feedback)

print("Common issues (negative feedback):")
for word, freq in negative_common_words:
    print(f"{word}: {freq}")

print("\nAreas of excellence (positive feedback):")
for word, freq in positive_common_words:
    print(f"{word}: {freq}")


# Visualize the results with WordClouds
def plot_wordcloud(words_freq, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

plot_wordcloud(negative_common_words, "Common Issues (Negative Feedback)")
plot_wordcloud(positive_common_words, "Areas of Excellence (Positive Feedback)")


# 4.4. Topic modeling to identify themes by agent, team manager, department, call type

from pyspark.sql.functions import avg, stddev
import seaborn as sns

# Function to calculate statistics and generate insights
def get_insights(df, group_by_column, score_column="Predicted_Sentiment_Score"):
    # Calculate mean and standard deviation of the predicted sentiment score
    stats_df = df.groupby(group_by_column).agg(
        avg_predicted_score=pd.NamedAgg(column=score_column, aggfunc="mean"),
        stddev_predicted_score=pd.NamedAgg(column=score_column, aggfunc="std"),
        count=pd.NamedAgg(column=score_column, aggfunc="count")
    ).reset_index()
    
    # Identify entities with consistently high and low predicted sentiment scores
    high_performers = stats_df[stats_df["avg_predicted_score"] > stats_df["avg_predicted_score"].quantile(0.5)].round(4)
    low_performers = stats_df[stats_df["avg_predicted_score"] < stats_df["avg_predicted_score"].quantile(0.5)].round(4)
    
    return stats_df, high_performers, low_performers

# Get insights for agents
agent_stats, high_perform_agents, low_perform_agents = get_insights(sa_table_pd, "Agent_Username")
# Get insights for managers
manager_stats, high_perform_managers, low_perform_managers = get_insights(sa_table_pd, "Team_Manager")
# Get insights for departments
dept_stats, high_perform_depts, low_perform_depts = get_insights(sa_table_pd, "Department")
# Get insights for call types
call_stats, high_perform_calls, low_perform_calls = get_insights(sa_table_pd, "Call_Type")


# Display insights on agents
# Flag agents who are also team managers
high_perform_agents['is_team_manager'] = high_perform_agents['Agent_Username'].isin(manager_stats['Team_Manager'])
low_perform_agents['is_team_manager'] = low_perform_agents['Agent_Username'].isin(manager_stats['Team_Manager'])
# Display the insights
print("High Performing Agents:")
high_perform_agents.sort_values(by="avg_predicted_score", ascending=False).head(10).display()
print("\nLow Performing Agents:")
low_perform_agents.sort_values(by="avg_predicted_score", ascending=True).head(10).display()


# Display insights on managers
print("\nHigh Performing Managers:")
high_perform_managers.sort_values(by="avg_predicted_score", ascending=False).head(10).display()
print("\nLow Performing Managers:")
low_perform_managers.sort_values(by="avg_predicted_score", ascending=True).head(10).display()


# Display insights on Departments
print("\nHigh Performing Departments:")
high_perform_depts.sort_values(by="avg_predicted_score", ascending=False).head(10).display()
print("\nLow Performing Departments:")
low_perform_depts.sort_values(by="avg_predicted_score", ascending=True).head(10).display()


# Display insights on call type
print("\nHigh Performing Call Types:")
high_perform_calls.sort_values(by="avg_predicted_score", ascending=False).head(10).display()

print("\nLow Performing Call Types:")
low_perform_calls.sort_values(by="avg_predicted_score", ascending=True).head(10).display()


#-------------------------------------------
## 5. Plot results

## 5.1. For entire dataset

# Function to plot the results in descending order
def plot_insights(stats_df, title, x_label, y_label):
    sorted_df = stats_df.sort_values(by="avg_predicted_score", ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sorted_df, x=sorted_df.columns[0], y="avg_predicted_score")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.show()

plot_insights(agent_stats, "Average Predicted Sentiment Score by Agent", "Agent Username", "Average Predicted Sentiment Score")
plot_insights(manager_stats, "Average Predicted Sentiment Score by Manager", "Team Manager", "Average Predicted Sentiment Score")
plot_insights(dept_stats, "Average Predicted Sentiment Score by Department", "Department", "Average Predicted Sentiment Score")
plot_insights(call_stats, "Average Predicted Sentiment Score by Call Type", "Call Type", "Average Predicted Sentiment Score")


## 5.2. Plot only the top and bottom 5% of high and low performers

# Function to plot high performers in descending order and low performers in ascending order
def plot_extreme_insights(stats_df, title, x_label, y_label, high_performers=True):
    if high_performers:
        sorted_df = stats_df.sort_values(by="avg_predicted_score", ascending=False)
    else:
        sorted_df = stats_df.sort_values(by="avg_predicted_score", ascending=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sorted_df, x=sorted_df.columns[0], y="avg_predicted_score")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Plot the high performers in descending order and low performers in ascending order
plot_extreme_insights(high_perform_agents, "High Performing Agents", "Agent Username", "Average Predicted Sentiment Score", high_performers=True)
plot_extreme_insights(low_perform_agents, "Low Performing Agents", "Agent Username", "Average Predicted Sentiment Score", high_performers=False)

plot_extreme_insights(high_perform_managers, "High Performing Managers", "Team Manager", "Average Predicted Sentiment Score", high_performers=True)
plot_extreme_insights(low_perform_managers, "Low Performing Managers", "Team Manager", "Average Predicted Sentiment Score", high_performers=False)

plot_extreme_insights(high_perform_depts, "High Performing Departments", "Department", "Average Predicted Sentiment Score", high_performers=True)
plot_extreme_insights(low_perform_depts, "Low Performing Departments", "Department", "Average Predicted Sentiment Score", high_performers=False)

plot_extreme_insights(high_perform_calls, "High Performing Call Types", "Call Type", "Average Predicted Sentiment Score", high_performers=True)
plot_extreme_insights(low_perform_calls, "Low Performing Call Types", "Call Type", "Average Predicted Sentiment Score", high_performers=False)



# 5.3. check avg predicted score and no. of agents by no. of contacts grouped at an internval of 5
# Calculate the number of rows for each agent
agent_row_counts = sa_table_pd.groupby('Agent_Username').size().reset_index(name='row_count')

# Merge the row counts with the original DataFrame
merged_df = pd.merge(sa_table_pd, agent_row_counts, on='Agent_Username')

# Define the bin edges and labels for the intervals
max_rows = agent_row_counts['row_count'].max()
bin_edges = list(range(0, max_rows + 5, 5))
bin_labels = [f"{i+1}-{i+5}" for i in range(0, max_rows, 5)]


# Create a new column for the intervals in both DataFrames
merged_df['row_count_interval'] = pd.cut(merged_df['row_count'], bins=bin_edges, labels=bin_labels, right=False)
agent_row_counts['row_count_interval'] = pd.cut(agent_row_counts['row_count'], bins=bin_edges, labels=bin_labels, right=False)

# Calculate the average predicted score for each interval
avg_predicted_scores = merged_df.groupby('row_count_interval')['Predicted_Sentiment_Score'].mean().reset_index()

# Calculate the number of agents in each interval
agents_per_interval = agent_row_counts.groupby('row_count_interval').size().reset_index(name='agent_count')

# Merge the average predicted scores with the agent counts
result_df = pd.merge(avg_predicted_scores, agents_per_interval, on='row_count_interval')

# Filter out intervals with zero agent count
result_df = result_df[result_df['agent_count'] > 0]

# Display the result
result_df.sort_values(by="Predicted_Sentiment_Score", ascending=True).head(100).display()

# Plot agent count by row count interval in ascending order
plt.figure(figsize=(12, 6))
sorted_result_df = result_df.sort_values(by='agent_count', ascending=True)
barplot = sns.barplot(data=sorted_result_df, x='row_count_interval', y='agent_count')
for index, row in sorted_result_df.iterrows():
    barplot.text(index, row['agent_count'], str(row['agent_count']), color='black', ha="center")
plt.title('Agent Count by Row Count Interval')
plt.xlabel('Row Count Interval')
plt.ylabel('Agent Count')
plt.xticks(rotation=45)
plt.show()



# 5.4. check avg predicted score and no. of agents by no. of contacts grouped at an internval of 10

# Calculate the number of rows for each agent
agent_row_counts = sa_table_pd.groupby('Agent_Username').size().reset_index(name='row_count')

# Merge the row counts with the original DataFrame
merged_df = pd.merge(sa_table_pd, agent_row_counts, on='Agent_Username')

# Define the bin edges and labels for the intervals
max_rows = agent_row_counts['row_count'].max()
bin_edges = list(range(0, max_rows + 11, 10))
bin_labels = [f"{i+1}-{i+10}" for i in range(0, max_rows, 10)]


# Create a new column for the intervals in both DataFrames
merged_df['row_count_interval'] = pd.cut(merged_df['row_count'], bins=bin_edges, labels=bin_labels, right=False)
agent_row_counts['row_count_interval'] = pd.cut(agent_row_counts['row_count'], bins=bin_edges, labels=bin_labels, right=False)

# Calculate the average predicted score for each interval
avg_predicted_scores = merged_df.groupby('row_count_interval')['Predicted_Sentiment_Score'].mean().reset_index()

# Calculate the number of agents in each interval
agents_per_interval = agent_row_counts.groupby('row_count_interval').size().reset_index(name='agent_count')

# Merge the average predicted scores with the agent counts
result_df = pd.merge(avg_predicted_scores, agents_per_interval, on='row_count_interval')

# Filter out intervals with zero agent count
result_df = result_df[result_df['agent_count'] > 0]

# Display the result
result_df.sort_values(by="Predicted_Sentiment_Score", ascending=True).head(100).display()

# Plot agent count by row count interval in ascending order
plt.figure(figsize=(12, 6))
sorted_result_df = result_df.sort_values(by='agent_count', ascending=True)
barplot = sns.barplot(data=sorted_result_df, x='row_count_interval', y='agent_count')
for index, row in sorted_result_df.iterrows():
    barplot.text(index, row['agent_count'], str(row['agent_count']), color='black', ha="center")
plt.title('Agent Count by Row Count Interval')
plt.xlabel('Row Count Interval')
plt.ylabel('Agent Count')
plt.xticks(rotation=45)
plt.show()

# Quality Check
# Calculate the total number of unique Agent_Username values
total_unique_agents = sa_table_pd['Agent_Username'].nunique()
print(f"Total number of unique agents: {total_unique_agents}")

# Calculate the sum of agent_count
total_agent_count = result_df['agent_count'].sum()
print(f"Sum of agent_count: {total_agent_count}")

# 5.5. Find cut points for tom and bottom 5%
# - High Performing Agents: Selected based on a straightforward threshold or quantile. This method is simpler but might include agents who are not consistently high-performing.
# - True High Performing Agents: Selected based on their position within the overall distribution of scores, ensuring that only agents who are consistently within the top 5% are included. This method accounts for the overall distribution and reduces the influence of outliers.

# Calculate the cumulative density of avg_predicted_score
sorted_scores = agent_stats['avg_predicted_score'].sort_values().reset_index(drop=True)
cumulative_density = np.cumsum(sorted_scores) / np.sum(sorted_scores)

# Determine the cut points (e.g., top 5% and bottom 5%)
high_cut_point = sorted_scores[cumulative_density >= 0.95].iloc[0]
low_cut_point = sorted_scores[cumulative_density <= 0.05].iloc[-1]

print(f"High cut point: {high_cut_point}")
print(f"Low cut point: {low_cut_point}")


# FOR AGENTS
# Filter high performing agents
true_high_perform_agents = agent_stats[agent_stats['avg_predicted_score'] >= high_cut_point]

# Filter low performing agents
true_low_perform_agents = agent_stats[agent_stats['avg_predicted_score'] <= low_cut_point]


# Display the top 100 high performing agents with only Agent_Username and avg_predicted_score
print("True High Performing Agents:")
true_high_perform_agents[['Agent_Username', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=False).head(10).display()

# Display the top 100 low performing agents with only Agent_Username and avg_predicted_score
print("\nTrue Low Performing Agents:")
true_low_perform_agents[['Agent_Username', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=True).head(10).display()


# Plot high performing agents
plt.figure(figsize=(12, 6))
high_perform_plot = sns.barplot(data=true_high_perform_agents[['Agent_Username', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=False).head(100), 
                                x='Agent_Username', y='avg_predicted_score')
high_perform_plot.set_xticklabels(high_perform_plot.get_xticklabels(), rotation=90)
plt.ylim(0, 5)
plt.title('Top 100 High Performing Agents')
plt.xlabel('Agent Username')
plt.ylabel('Average Predicted Sentiment Score')
plt.show()

# Plot low performing agents
plt.figure(figsize=(12, 6))
low_perform_plot = sns.barplot(data=true_low_perform_agents[['Agent_Username', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=True).head(100), 
                               x='Agent_Username', y='avg_predicted_score')
low_perform_plot.set_xticklabels(low_perform_plot.get_xticklabels(), rotation=90)
plt.ylim(0, 5)
plt.title('Top 100 Low Performing Agents')
plt.xlabel('Agent Username')
plt.ylabel('Average Predicted Sentiment Score')
plt.show()


# FOR MANAGERS
# Calculate the cumulative density of avg_predicted_score
sorted_scores = manager_stats['avg_predicted_score'].sort_values().reset_index(drop=True)
cumulative_density = np.cumsum(sorted_scores) / np.sum(sorted_scores)

# Determine the cut points (e.g., top 5% and bottom 5%)
high_cut_point = sorted_scores[cumulative_density >= 0.95].iloc[0]
low_cut_point = sorted_scores[cumulative_density <= 0.05].iloc[-1]

print(f"High cut point: {high_cut_point}")
print(f"Low cut point: {low_cut_point}")


# Filter high performing manager
true_high_perform_managers = manager_stats[manager_stats['avg_predicted_score'] >= high_cut_point]

# Filter low performing managers
true_low_perform_managers = manager_stats[manager_stats['avg_predicted_score'] <= low_cut_point]


# Display the top 100 high performing managers with only Team_Manager and avg_predicted_score
print("True High Performing Managers:")
true_high_perform_managers[['Team_Manager', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=False).head(100).display()

# Display the top 100 low performing managers with only Team_Manager and avg_predicted_score
print("\nTrue Low Performing Managers:")
true_low_perform_managers[['Team_Manager', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=True).head(100).display()


# Plot high performing managers
plt.figure(figsize=(12, 6))
high_perform_plot = sns.barplot(data=true_high_perform_managers[['Team_Manager', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=False).head(100), 
                                x='Team_Manager', y='avg_predicted_score')
high_perform_plot.set_xticklabels(high_perform_plot.get_xticklabels(), rotation=90)
plt.ylim(0, 5)
plt.title('Top 100 High Performing Managers')
plt.xlabel('Team Manager')
plt.ylabel('Average Predicted Sentiment Score')
plt.show()

# Plot low performing managers
plt.figure(figsize=(12, 6))
low_perform_plot = sns.barplot(data=true_low_perform_managers[['Team_Manager', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=True).head(100), 
                               x='Team_Manager', y='avg_predicted_score')
low_perform_plot.set_xticklabels(low_perform_plot.get_xticklabels(), rotation=90)
plt.ylim(0, 5)
plt.title('Top 100 Low Performing Managers')
plt.xlabel('Team Manager')
plt.ylabel('Average Predicted Sentiment Score')
plt.show()


# FOR DEPARTMENT
# Calculate the cumulative density of avg_predicted_score
sorted_scores = dept_stats['avg_predicted_score'].sort_values().reset_index(drop=True)
cumulative_density = np.cumsum(sorted_scores) / np.sum(sorted_scores)

# Determine the cut points (e.g., top 5% and bottom 5%)
high_cut_point = sorted_scores[cumulative_density >= 0.95].iloc[0]
low_cut_point = sorted_scores[cumulative_density <= 0.05].iloc[-1]

print(f"High cut point: {high_cut_point}")
print(f"Low cut point: {low_cut_point}")


# Filter high performing dept
true_high_perform_depts = dept_stats[dept_stats['avg_predicted_score'] >= high_cut_point]

# Filter low performing dept
true_low_perform_depts = dept_stats[dept_stats['avg_predicted_score'] <= low_cut_point]


# Display the top 100 high performing dept with only Department and avg_predicted_score
print("True High Performing Departments:")
true_high_perform_depts[['Department', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=False).head(10).display()

# Display the top 100 low performing dept with only Department and avg_predicted_score
print("\nTrue Low Performing Departments:")
true_low_perform_depts[['Department', 'avg_predicted_score']].sort_values(by="avg_predicted_score", ascending=True).head(10).display()