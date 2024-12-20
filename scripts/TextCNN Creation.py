# Aspect-Based Sentiment Analysis using TextCNN

"""
TextCNN (Convolutional Neural Networks for Text) can capture local patterns in text, which is useful for aspect-based sentiment classification. This method is especially effective for shorter text segments.

Implementation Steps:

- Data Preprocessing: Tokenize and pad the text data.
- Model Construction: Build a CNN model tailored for text data.
- Training and Evaluation: Train the model on aspect-based sentiment data.
"""

%run /Workspace/root/MarketingData/ETL/lib/dip_etl_functions

## 1.Install libraries

# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev, corr, min


# Set random seeds to ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Initialize Spark session
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#--------------------------------------------
## 2. Load preprocessed feedback table

# Load the preprocessed data
preprocessed_feedback_table = spark.table("ds_sandbox.preprocessed_feedback_table")

# Display the number of rows and columns
# Print number of rows and columns
def print_shape(df):
    row_num = df.count()
    col_num = len(df.columns)
    print(f"Number of rows: {row_num}, Number of columns: {col_num}")

print_shape(preprocessed_feedback_table)

# Repartition the data
# Repartition the DataFrame to 8 partitions
preprocessed_feedback_table = preprocessed_feedback_table.repartition(8)


#--------------------------------------------
## 3. Prepare the data

# Convert Spark DataFrame to Pandas DataFrame
feedback_df = preprocessed_feedback_table.select("Comment", "Expanded_Comment", "Survey_Score", "Sentiment_Score", "Query_Resolved", "Team_Manager", "Agent_Username", "Department", "Call_Type").toPandas()

# Split the data into training and testing sets
train_df, test_df = train_test_split(feedback_df, test_size=0.2, random_state=42)

# Display the number of rows and columns in test_df
print_shape(test_df)

# Separate features and labels
train_texts, train_labels = train_df['Expanded_Comment'].tolist(), train_df['Survey_Score'].tolist()
test_texts, test_labels = test_df['Expanded_Comment'].tolist(), test_df['Survey_Score'].tolist()
test_sentiment_scores = test_df['Sentiment_Score'].tolist()  # Extract Sentiment Scores for the test set
test_query_resolved = test_df['Query_Resolved'].tolist()  # Extract Query_Resolved for the test set
test_team_manager = test_df['Team_Manager'].tolist()  # Extract Team_Manager for the test set
test_agent_username = test_df['Agent_Username'].tolist()  # Extract Agent_Username for the test set
test_department = test_df['Department'].tolist()  # Extract Department for the test set
test_call_type = test_df['Call_Type'].tolist()  # Extract Call_Type for the test set
test_original_comment = test_df['Comment'].tolist() # Extract Comment for the test set


#--------------------------------------------
## 4. Tokenize and Pad Sequences
# - The text data is tokenized and converted into sequences of integers.
# - The sequences are then padded to ensure uniform length across the dataset.

# Parameters
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

# Tokenize the text
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


#--------------------------------------------
## 5. Build the TextCNN Model
# - A Convolutional Neural Network (CNN) is built for text classification.
# - The model includes an embedding layer, convolutional layers, pooling layers, and dense layers.

def create_text_cnn(vocab_size, embedding_dim, max_sequence_length):
    inputs = Input(shape=(max_sequence_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
    convs = []
    filter_sizes = [3, 4, 5]
    
    for filter_size in filter_sizes:
        conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedding)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    
    concatenated = tf.keras.layers.Concatenate()(convs)
    dropout = Dropout(0.5)(concatenated)
    outputs = Dense(1, activation='linear')(dropout)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

# Create the model
model = create_text_cnn(vocab_size=MAX_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_sequence_length=MAX_SEQUENCE_LENGTH)
model.summary()


#--------------------------------------------
## 6. Train the Model
# - The model is trained on the training data with validation on a separate validation dataset.

# Convert labels to numpy arrays
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=2, callbacks=[early_stopping])

# Extract validation loss for each epoch
val_loss = history.history['val_loss']

# Find the epoch with the minimum validation loss using the built-in min function explicitly
min_val_loss = __builtins__.min(val_loss)
optimal_epoch = val_loss.index(min_val_loss) + 1  # Adding 1 to convert 0-based index to 1-based

print(f'Lowest validation loss: {min_val_loss:.4f} at epoch {optimal_epoch}')


#--------------------------------------------
## 7. Evaluate the Model
# - The model's performance is evaluated using MAE and loss metrics on the validation dataset.

loss, mae = model.evaluate(X_test, y_test, verbose=2)
print(f'Test MAE: {mae:.2f}')


#--------------------------------------------
## 8. Predict Sentiment Scores

# Predict sentiment scores on the test set
predicted_scores = model.predict(X_test)

# Clip the predicted scores to be within the range 1 to 5
predicted_scores = np.clip(predicted_scores, 1, 5).round().astype(int)

# Convert the numpy array to a list for easier handling
predicted_scores = predicted_scores.flatten().tolist()


#--------------------------------------------
## 9. Display predicted sentiment score

# Create a DataFrame to display initial vs. predicted sentiment scores and survey scores
results_df = pd.DataFrame({
    'Comment': test_original_comment,
    'Expanded_Comment': test_texts,
    'Sentiment_Score': test_sentiment_scores,
    'Predicted_Sentiment_Score': predicted_scores,
    'Survey_Score': test_labels,
    'Query_Resolved': test_query_resolved,
    'Team_Manager': test_team_manager,
    'Agent_Username': test_agent_username,
    'Department': test_department,
    'Call_Type': test_call_type,
})

# Display the first few rows of the DataFrame
results_df.head(10).display()

# Check if there are any NaN values
# Ensure there are no NaN values
print(results_df['Predicted_Sentiment_Score'].isna().sum())  # Should be 0
print(results_df['Sentiment_Score'].isna().sum())  # Should be 0
print(results_df['Survey_Score'].isna().sum())  # Should be 0


#--------------------------------------------
## 10. Show the correlation between Predicted_Sentiment_Score Survey Score

# Calculate Correlation for columns of interest
columns_of_interest = ["Survey_Score", "Sentiment_Score", "Predicted_Sentiment_Score"]
correlation_matrix = results_df[columns_of_interest].corr()

# Apply a background gradient that maps colors to the correlation values
styled_matrix = correlation_matrix.style.background_gradient(cmap='Blues').format("{:.2f}")

# Display the styled correlation matrix
print('Correlation between Survey Score, Sentiment Score, and Predicted Sentiment Score:')
display(styled_matrix)


#--------------------------------------------
## 11. Plot the Correlation

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


#--------------------------------------------
## 12. Compare Survey_Score and Sentiment_Score with Predicted_Sentiment_Score

# Convert pandas df back to spark df
results_spark_df = spark.createDataFrame(results_df)

# Specify the columns of interest
column_names = ['Survey_Score', 'Sentiment_Score', 'Predicted_Sentiment_Score']

# Function to calculate and display mean and standard deviation for given columns
def calculate_stats(df, column_names):
    expressions = []
    for col in column_names:
        expressions.append(mean(col).alias(f'mean_{col}'))
        expressions.append(stddev(col).alias(f'std_{col}'))
    metrics = df.agg(*expressions).collect()[0]
    
    # Create a dictionary to store the results
    results = {
        "Metric": ["Mean", "Standard Deviation"],
    }
    for col in column_names:
        results[col] = [f'{metrics[f"mean_{col}"]:.4f}', f'{metrics[f"std_{col}"]:.4f}']
    
    # Convert dictionary to DataFrame for better visualization
    results_df = pd.DataFrame(results)
    return results_df

# Calculate statistical metrics for specific columns and display them
stats_df = calculate_stats(results_spark_df, column_names)
stats_df.display()

# Step 1: Create Score_Difference column
results_df['Score_Difference_CS_PS'] = results_df['Survey_Score'] - results_df['Predicted_Sentiment_Score'] 
results_df['Score_Difference_IS_PS'] = results_df['Sentiment_Score'] - results_df['Predicted_Sentiment_Score']
# Here CS is for customer rated survey_score, IS is for Initial Sentiment_Score, and PS is Predicted_Sentiment_Score

# Step 2: Statistical Summary of the Differences
print(results_df['Score_Difference_CS_PS'].describe())
print(results_df['Score_Difference_IS_PS'].describe())

# Step 3: Distribution Analysis of Score Differences
CS_PS_score_difference_distribution = results_df['Score_Difference_CS_PS'].value_counts().sort_index()
IS_PS_score_difference_distribution = results_df['Score_Difference_IS_PS'].value_counts().sort_index()

print(CS_PS_score_difference_distribution)
print(IS_PS_score_difference_distribution)


#--------------------------------------------
## 13. Plot the Score Difference

# Separate function for plotting histograms
def plot_score_differences(df, column_name, column_for_title):
    title = f'Distribution of Score Differences between {column_for_title}'
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], bins=10, kde=True)
    plt.title(title)
    plt.xlabel('Score Difference')
    plt.ylabel('Frequency')
    plt.show()

plot_score_differences(results_df, 'Score_Difference_CS_PS', 'Customer Rated Survey_Score and Predicted_Sentiment_Score')
plot_score_differences(results_df, 'Score_Difference_IS_PS', 'Initial Sentiment_Score and Predicted_Sentiment_Score')


#--------------------------------------------
## 14. Validate the TextCNN model using appropriate metrics

# Extract loss and mae values for each epoch
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation MAE
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_mae, 'bo-', label='Training MAE')
plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()



#--------------------------------------------
## 15. Create a confusion matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Convert the predicted scores to integer labels
predicted_labels = np.clip(np.round(predicted_scores).astype(int), 1, 5)
true_labels = np.clip(np.round(y_test).astype(int), 1, 5)

# Calculating accuracy, precision, recall, and F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Printing the metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


#--------------------------------------------
## 16. Sentiment Analysis using Gradient Boosting

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

#train_df, test_df
train_df = spark.createDataFrame(train_df)
test_df = spark.createDataFrame(test_df)

# Tokenize the text
tokenizer = Tokenizer(inputCol="Expanded_Comment", outputCol="words")

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

# Apply TF-IDF
hashing_tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Define the Gradient Boosting model
gbt = GBTRegressor(featuresCol="features", labelCol="Survey_Score", maxIter=10, maxDepth=3)

# Create a Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, gbt])

# Train the model
gbt_model = pipeline.fit(train_df)

# Predict on the test set
predictions = gbt_model.transform(test_df)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Survey_Score", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print(f"Test MAE: {mae:.2f}")

evaluator = RegressionEvaluator(labelCol="Survey_Score", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Test RMSE: {rmse:.2f}")

# Convert predictions Spark DataFrame to a Pandas DataFrame
predictions_pandas = predictions.select("prediction").toPandas()

# Add the prediction results to the original Pandas DataFrame
results_df['GBT_Predicted_Sentiment_Score'] = predictions_pandas['prediction']

# Calculate Correlation for columns of interest
columns_of_interest = ["Survey_Score", "Sentiment_Score", "Predicted_Sentiment_Score", "GBT_Predicted_Sentiment_Score"]
correlation_matrix = results_df[columns_of_interest].corr(numeric_only=True)

# Apply a background gradient that maps colors to the correlation values
styled_matrix = correlation_matrix.style.background_gradient(cmap='Blues').format("{:.2f}")

# Display the styled correlation matrix
print('Correlation between Survey Score, Sentiment Score, and GBT Predicted Sentiment Score:')
display(styled_matrix)

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#--------------------------------------------
## 17. Save the Sentiment Analysis Results

### 17.a. Save the SA Results DataFrame
# Convert pandas df back to spark df
results_spark_df = spark.createDataFrame(results_df)

results_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("dbfs:/FileStore/tables/john_marshman/Bhavya_pathak/sa_table")


### 17.b. Verify by loading and displaying SA results data
# Load the SA DataFrame from the registered table
sa_table = (
    spark.read.format("delta")
    .option("header", "true")
    .load("dbfs:/FileStore/tables/john_marshman/Bhavya_pathak/sa_table"))

# Verify by showing the schema or some rows
sa_table.printSchema()
sa_table.limit(100).display()