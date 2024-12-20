# Pre-processing the dataset

## 1. Install libraries


%pip install pyspellchecker
%pip install textblob


# Restart the kernel
dbutils.library.restartPython()


# Import libraries
from pyspark.sql.functions import col, sum, when, udf, array, avg, regexp_extract, regexp_replace, length, split, expr, explode, rand, collect_list, lower, array_except, size, abs
from pyspark.sql.types import IntegerType, BooleanType, TimestampType, StringType
from pyspark.sql import SparkSession
from spellchecker import SpellChecker
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from datasets import Dataset, DatasetDict
import torch
from sklearn.model_selection import train_test_split
import mlflow
import time
from transformers import Trainer, TrainingArguments
import datetime
import pandas as pd


# Initialize Spark session
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#---------------------------------------
## 2. Load the data

feedback_table = spark.table("ds_sandbox.rantandravedata_1206")
feedback_table.limit(100).display()

# Print number of rows and columns
def print_shape(df):
    row_num = df.count()
    col_num = len(df.columns)
    print(f"Number of rows: {row_num}, Number of columns: {col_num}")

print_shape(feedback_table)


#---------------------------------------
## 3. Convert column data types

# Convert column data types
feedback_table = feedback_table \
    .withColumn("Survey_Score", col("Survey_Score").cast(IntegerType())) \
    .withColumn("Query_Resolved", col("Query_Resolved").cast(BooleanType())) \
    .withColumn("Is_FCA_Auditable", col("Is_FCA_Auditable").cast(BooleanType())) \
    .withColumn("created", col("created").cast(TimestampType())) \
    .withColumn("CSAT_Score_2", col("CSAT_Score_2").cast(IntegerType())) \
    .withColumn("Query_Resolved_2", col("Query_Resolved_2").cast(IntegerType())) \
    .withColumn("id", col("id").cast(IntegerType())) \
    .withColumn("updated", col("updated").cast(TimestampType())) \
    .withColumn("Agent_Start_Date", col("Agent_Start_Date").cast(TimestampType())) \
    .withColumn("Agent_Leave_Date", col("Agent_Leave_Date").cast(TimestampType())) \
    .withColumn("Inquarter", col("Inquarter").cast(IntegerType())) \
    .withColumn("Sentiment_Score", col("Sentiment_Score").cast(IntegerType())) \
    .withColumn("Appealed", when(col("Appealed") == "Dispute", True).otherwise(False))

# Show updated schema
feedback_table.printSchema()


#---------------------------------------
## 4. Drop extra columns from the table

# Drop unnecessary columns
columns_to_drop = [
    "Workgroup_Name", "Location", "Full_Time_Part_Time_Indicator", 
    "Call_Resolution", "Department_Name", "updated", "Agent_Start_Date", 
    "Agent_Leave_Date", "Agent_Cost_Centre", "Agent_Site"
]
reduced_feedback_table = feedback_table.drop(*columns_to_drop)

# Show the updated schema to confirm the columns have been removed
reduced_feedback_table.printSchema()


#---------------------------------------
## 5. Remove rows with null values or 0s in sentiment score column

### 5.a. Drop rows with null values

# Drop rows with null values in critical columns
essential_columns = ['Comment', 'Survey_Score', 'Query_Resolved', 'Team_Manager', 'Agent_Username', 'Department', 'Interaction_Type', 'Call_Response', 'Call_Type', 'Sentiment_Score']
cleaned_feedback_table = reduced_feedback_table.dropna(how='any', subset=essential_columns)

# Print number of rows and columns
print_shape(cleaned_feedback_table)

# Check if all the null values have been removed
null_counts = cleaned_feedback_table.select(
  [sum(
    col(c).isNull().cast("int")
    ).alias(c) for c in cleaned_feedback_table.columns]
  )
display(null_counts)


### 5.b. Drop rows where Sentiment_Score is 0

# Calculate and display counts for zero Sentiment_Score
num_zeros = cleaned_feedback_table.filter(col('Sentiment_Score') == 0).count()
print(f"Number of 0s in Sentiment_Score column: {num_zeros}")

# Display unique comments with Sentiment_Score 0
# Filter the DataFrame where Sentiment_Score is 0 and select the Comment column
comments_with_zero_score = cleaned_feedback_table.filter(col('Sentiment_Score') == 0).select('Comment')

# Get unique comments
unique_comments = comments_with_zero_score.distinct().collect()

# Print the unique comments
print("Unique comments where Sentiment_Score is 0:")
for row in unique_comments:
    print(row['Comment'])

# Calculate the number of occurrences of I am very pleased with the service in the Comment column - CAN DELETE LATER
n = cleaned_feedback_table.filter(cleaned_feedback_table['Comment'] == "I am very pleased with the service").count()
print(f"Number of 0s in Comment column: {n}")

# Drop rows with Sentiment_Score 0
cleaned_feedback_table = cleaned_feedback_table.filter(col('Sentiment_Score') != 0)

# Print the number of remaining rows
print(f"Number of rows after dropping rows with Sentiment_Score = 0: {cleaned_feedback_table.count()}")


#---------------------------------------
## 6. Identifying enetries with Non-English characters in the 'Comment' column

### 6.a. Replace non standard characters with standard characters and remove emojies


# Clean up the 'Comment' column
def clean_comment(text):
    replacements = {
        "’": "'", "‘": "'", "–": "-", "⭐": "", "❤️": "", "♥": "", "♀": "",
        "☺": "", "☹": "", "�": "", "✔": "", "…": "...", "“": "\"", "”": "\"",
        "à": "a", "č": "c", "ć": "c", "è": "e", "é": "e", "ı": "i", "İ": "I", 
        "ò": "o", "ö": "o", "ÿ": "y", "&": "and", "\|": "", r'\uFE0F': "", 
        r'\u200D': "", r'\u2B50': "", "️": "", "‍": "", "✅": "", "❤": "", "✨": ""
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

clean_comment_udf = udf(clean_comment, StringType())
cleaned_feedback_table = cleaned_feedback_table.withColumn("Comment", clean_comment_udf(col("Comment")))


### 6.b. Identify english characters, numbers, common punctuations and pound sign

# Define a regular expression to identify non-English characters
# This regex matches any character that is not a basic Latin character, number, or common punctuation, the pound sign
regex_pattern = '[^\x00-\x7F£]+'

# Add a new column 'non_english' that flags rows with non-English characters in 'Comment'
feedback_table_with_flag = cleaned_feedback_table.withColumn(
    'non_english', 
    regexp_extract(col('Comment'), regex_pattern, 0)
)
non_english_count = feedback_table_with_flag.filter(col('non_english') != '').count()
print(f"Number of rows with non-English characters in the 'Comment' column: {non_english_count}")


### 6.c. Remove rows with non-English characters
feedback_table_english_only = feedback_table_with_flag.filter(col('non_english') == '').drop('non_english')

print_shape(feedback_table_english_only)


### 6.d. Compare average Sentiment_Score for uncleaned data (325 records) vs cleaned+uncleaned data (536104 records)

# Create a dataframe with rows containing non-English characters
non_english_rows = feedback_table_with_flag.filter(col('non_english') != '')

# Calculate average sentiment score - should this be survey score?
# Calculate the average Sentiment_Score for the entire cleaned_feedback_table
feedback_table_with_flag.agg(avg(col("Sentiment_Score")).alias("Average_Sentiment_Score")).show()

# Filter rows where 'non_english' is not empty to find non-English rows
non_english_rows = feedback_table_with_flag.filter(col('non_english') != '')

# Calculate the average Sentiment_Score for these non-English rows
non_english_rows.agg(avg(col("Sentiment_Score")).alias("Average_Sentiment_Score_Non_English")).show()


### 6.e. e. Check for the pending non-standard charcters in comment column in the non_english rows

# Display the pending non standard character for each comment
# Define a regex that will catch characters outside of expected ranges
regex_pattern = '[^\x00-\x7F]'

# Extract these characters for inspection
feedback_table_with_temp_flag = non_english_rows.withColumn(
    'non_standard_chars',
    regexp_extract(col('Comment'), regex_pattern, 0)
)

# Display rows where 'non_standard_chars' is not empty
feedback_table_with_temp_flag.filter(col('non_standard_chars') != '').select('Comment', 'non_standard_chars').limit(100).display(truncate=False)

# Display the pending non standard characters and the no of records they are present in
# # Assuming 'non_standard_chars' contains strings of non-standard characters
# First, let's create a new DataFrame that separates all non-standard characters into separate rows
char_exploded_df = feedback_table_with_temp_flag.withColumn(
    "non_standard_char",
    explode(split(regexp_extract(col("Comment"), "([^\x00-\x7F]+)", 1), ""))
)

# Now, group by the non-standard characters and count occurrences
character_counts = char_exploded_df.groupBy("non_standard_char").count().orderBy("count", ascending=False)

# Show the results, displaying the count of each non-standard character
character_counts.display(truncate=False, ascending=False)


#---------------------------------------
## 7. Expand abbreviations in cleaned_column

### 7.a. Load the abbreviations data

abbreviations = spark.table("ds_sandbox.abbreviations")
abbreviations.limit(10).display()


### 7.b. Preprocessing and broadcasting abbreviations dictionary

# Convert the 'Word' and 'Meaning' columns to lowercase
abbreviations = abbreviations.withColumn("Word", lower(col("Word"))).withColumn("Meaning", lower(col("Meaning")))

# Collect the abbreviations as a list and create a dictionary
abbreviations_dict = {row['Word']: row['Meaning'] for row in abbreviations.select("Word", "Meaning").collect()}

# Broadcast the abbreviations dictionary
abbreviations_broadcast = spark.sparkContext.broadcast(abbreviations_dict)


### 7.c. Expand abbreviations using the abbreviation map

def expand_abbreviations(text):
    abbreviations_map = abbreviations_broadcast.value
    words = text.split()
    expanded_words = [abbreviations_map.get(word.lower(), word) for word in words]
    return ' '.join(expanded_words)
  
expand_abbreviations_udf = udf(expand_abbreviations, StringType())


### 7.d. Convert the 'Comment' column to lowercase and update the DataFrame by adding a column with expanded comments
feedback_table_english_only = feedback_table_english_only.withColumn("Expanded_Comment", expand_abbreviations_udf(lower(col("Comment"))))


### 7.e. Display Expanded Abbreviations
# Show the original, expanded, and corrected comments to verify the process
feedback_table_english_only.select("Comment", "Expanded_Comment").limit(100).display(truncate=False)


### 7.f. Create a new column "Expanded_Words" to display the exact words that have been expanded for each comment
# Split the comments into arrays of words
feedback_table_english_only = feedback_table_english_only.withColumn("Original_Words", split(col("Comment"), " "))
feedback_table_english_only = feedback_table_english_only.withColumn("Expanded_Words", split(col("Expanded_Comment"), " "))

# Identify the differences between the original and expanded comments
feedback_table_english_only = feedback_table_english_only.withColumn("Expanded_Abbreviations", array_except(col("Expanded_Words"), col("Original_Words")))

# Filter out rows where no expansion occurred
expanded_rows = feedback_table_english_only.filter(expr("size(Expanded_Abbreviations) > 0"))

# Select and display the relevant columns to see the expansions
expanded_rows.select("Comment", "Expanded_Comment", "Expanded_Abbreviations").limit(100).display(truncate=False)


### 7.g. Display no. of rows where abbreviations have been expanded
# Count the number of rows where abbreviations have been expanded
expanded_rows_count = feedback_table_english_only.filter(size(col("Expanded_Abbreviations")) > 0).count()

# Print the number of rows
print(f"Number of rows where abbreviations got expanded: {expanded_rows_count}")


#---------------------------------------
## 8. Tokenization, Removal of Stop Words, and Lemmatization (using SpaCy)

!python -m spacy download en_core_web_sm

### 8.a. Load spaCy's English model
nlp = spacy.load('en_core_web_sm')


### 8.b. Perform tokenization, stop word removal, and lemmatization

# Create and register preprocess_and_lemmatize UDF
def preprocess_and_lemmatize(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmatized_tokens)

preprocess_and_lemmatize_udf = udf(preprocess_and_lemmatize, StringType())

# Apply the preprocess_and_lemmatize UDF to the DataFrame
feedback_table_english_only = feedback_table_english_only.withColumn("Expanded_Comment", preprocess_and_lemmatize_udf(col("Expanded_Comment")))


#---------------------------------------
## 9. Perform sentiment analysis with TextBlob (This is done only as a preliminary sentiment analysis to set a baseline and built an initial understanding of data once cleaned)

### 9.a. Create and register get_sentiment UDF to apply TextBlob SA

def get_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get the polarity score [-1, 1]
    polarity = blob.sentiment.polarity
    # Convert polarity to a scale of 1 to 5
    sentiment_score = int((polarity + 1) * 2) + 1  # Transforms -1 to 1 into 1 to 5
    return sentiment_score
  
sentiment_udf = udf(get_sentiment, IntegerType())


### 9.b. Apply the get_sentiment UDF to feedback_table_english_only dataframe

feedback_table_english_only = feedback_table_english_only.withColumn("New_Sentiment_Score", sentiment_udf(col("Expanded_Comment")))


#---------------------------------------
## 10. Compare New_Sentiment_Score, Survey_Score and Sentiment_Score

### 10.a. Calculate average for New_Sentiment_Score (3.1), Survey_Score (4.6), and Sentiment_Score (4.1)

# Function to calculate and show average scores
def calculate_and_display_averages(df, score_columns):
    for score_column in score_columns:
        average_score = df.agg(avg(col(score_column)).alias(f"Average_{score_column}"))
        average_score.show()

# Columns to calculate averages for
score_columns = ["Survey_Score", "Sentiment_Score", "New_Sentiment_Score"]

# Calculate and display average scores
calculate_and_display_averages(feedback_table_english_only, score_columns)


### 10.b. Compare the two columns New_Sentiment_Score with Sentiment_Score and Survey_Score

#### Step 1: Calculate the difference between the scores
# Function to calculate score differences
def calculate_score_differences(df):
    df = df.withColumn("Score_Difference_IS_CS", abs(col("Sentiment_Score") - col("Survey_Score"))) \
           .withColumn("Score_Difference_NS_CS", abs(col("New_Sentiment_Score") - col("Survey_Score"))) \
           .withColumn("Score_Difference_NS_IS", abs(col("New_Sentiment_Score") - col("Sentiment_Score")))
    return df
  
# Calculate score differences
feedback_table_english_only = calculate_score_differences(feedback_table_english_only)

#### Step 2: Statistical Summary of the Differences
# Function to display statistical summaries
def display_stat_summaries(df, columns):
    for column in columns:
        print(f"Summary for {column}")
        df.describe(column).show()

# Display statistical summaries of score differences
score_difference_columns = ["Score_Difference_IS_CS", "Score_Difference_NS_CS", "Score_Difference_NS_IS"]
display_stat_summaries(feedback_table_english_only, score_difference_columns)
# NS is for New Sentiment Score, IS for Initial Sentiment Score, and CS is customer rated Survey Score

#### Step 3: Distribution Analysis of Score_Difference_Sentiment_Score and Score_Difference_Survey_Score
# Function to display distribution analysis
def display_distribution_analysis(df, columns):
    for column in columns:
        print(f"Count for {column}")
        score_difference_distribution = df.groupBy(column).count().orderBy(column)
        score_difference_distribution.show()

# Display distribution analysis of score differences
display_distribution_analysis(feedback_table_english_only, score_difference_columns)


### 10.c. Calculate correlation of New_Sentiment_Score with Sentiment_Score and Survey_Score

# Convert Spark DataFrame to Pandas DataFrame for calculating correlation matrix
score_data = feedback_table_english_only.select("Survey_Score", "Sentiment_Score", "New_Sentiment_Score", "Score_Difference_IS_CS", "Score_Difference_NS_CS", "Score_Difference_NS_IS",).toPandas()

# Calculate the correlation matrix
# Calculate Correlation for columns of interest
columns_of_interest = ["Survey_Score", "Sentiment_Score", "New_Sentiment_Score"]
correlation_matrix = score_data[columns_of_interest].corr()

# Display Correlation
print('Correlation between Survey Score, Sentiment Score, and New Sentiment Score:')
print(correlation_matrix)


#---------------------------------------
## 11. Visual Comparison

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

### 11.a. Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


### 11.b. Plot histogram to display score differences in New Sentiment Score vs Survey Score and Initial Sentiment Score
# Function to plot histograms
def plot_histograms(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], bins=5, kde=False)
        plt.title(f'Score difference distribution: {column}')
        plt.xlabel('Score Difference')
        plt.ylabel('Frequency')
        plt.show()

# Plot histograms for score differences
plot_histograms(score_data, score_difference_columns)


### 11.c. Plot scatter plot so show linear relationship between New Sentiment Score vs Survey Score and Initial Sentiment Score

# Calculate count of pairs of:
# a) Initial Sentiment Score - Survey Score
# b) New Sentiment Score - Survey Score
# c) New Sentiment Score - Initial Sentiment Score

# Calculate the count of occurrences for each (Initial Sentiment_Score, Customer Rated Survey_Score) pair
score_data['count_IS_CS_score'] = score_data.groupby(['Survey_Score', 'Sentiment_Score'])['Survey_Score'].transform('count')

# Calculate the count of occurrences for each (New_Sentiment_Score, Initial Sentiment_Score) pair
score_data['count_NS_IS_score'] = score_data.groupby(['Sentiment_Score', 'New_Sentiment_Score'])['Sentiment_Score'].transform('count')

# Calculate the count of occurrences for each (New_Sentiment_Score, Customer Rated Survey_Score) pair
score_data['count_NS_CS_score'] = score_data.groupby(['Survey_Score', 'New_Sentiment_Score'])['Survey_Score'].transform('count')

# Scatter plot pairings: [(x, y, size)]
x_y_pairs = [
    ('Survey_Score', 'Sentiment_Score', 'count_IS_CS_score'),
    ('Survey_Score', 'New_Sentiment_Score', 'count_NS_CS_score'),
    ('Sentiment_Score', 'New_Sentiment_Score', 'count_NS_IS_score')
]

# Plot Scatter plots
# Function to plot scatter plots
def plot_scatter_plots(df, x_y_pairs):
    for x, y, size in x_y_pairs:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, size=size, sizes=(20, 200), data=df, legend=False)
        plt.title(f'Scatter Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.plot([1, 5], [1, 5], 'r--')  # Adding a reference line
        plt.show()

# Plot scatter plots for score differences
plot_scatter_plots(score_data, x_y_pairs)


### 11.d. Plot boxplot to display distribution of values in Survey Score, Initial Sentiment Score and New Sentiment_Score

# Melt the DataFrame to have a single 'Score' column and a 'Type' column to differentiate the scores
melted_score_data = pd.melt(score_data, value_vars=['Survey_Score','Sentiment_Score', 'New_Sentiment_Score'], var_name='Type', value_name='Score')

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Score', data=melted_score_data)
plt.title('Boxplot of Sentiment Scores')
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.show()


#-------------------------------------------
## 12. Save the preprocessed data

%run /Workspace/root/MarketingData/ETL/lib/etl_helper_path_config

### 12. a. Save the pre-processed DataFrame
feedback_table_english_only.write.mode('overwrite').format("delta").option("overwriteSchema", "true").save(f"{sandbox}/bhavya_pathak/preprocessed_feedback_table")


### 12.b. Verify by loading and displaying preprocessed data

# Load the pre-processed DataFrame from the registered table
preprocessed_feedback_table = spark.table("ds_sandbox.preprocessed_feedback_table")

# Verify by showing the schema or some rows
preprocessed_feedback_table.printSchema()
preprocessed_feedback_table.limit(100).display()

