# TextCNN Hyperparameter Tuning

## 1. Install libraries

%pip install scikit-learn

# Restart the kernel
dbutils.library.restartPython()

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#-------------------------------------------
## 2. Load preprocessed feedback table

# Load the preprocessed data
preprocessed_feedback_table = spark.table("ds_sandbox.preprocessed_feedback_table")

# Display the number of rows and columns
row_num = preprocessed_feedback_table.count()
col_num = len(preprocessed_feedback_table.columns)

print(f"Number of rows are: {row_num} and number of columns are: {col_num}")

# Repartition the data
# Repartition the DataFrame to 8 partitions
preprocessed_feedback_table = preprocessed_feedback_table.repartition(8)


#-------------------------------------------
## 3. Prepare the data

# Convert Spark DataFrame to Pandas DataFrame
feedback_df = preprocessed_feedback_table.select("Expanded_Comment", "Survey_Score", "Sentiment_Score").toPandas()

# Sample 100000 random rows
#feedback_sample = feedback_df.sample(n=100000, random_state=42)

# Split the data into training and testing sets
#train_df, test_df = train_test_split(feedback_sample, test_size=0.2, random_state=42)
train_df, test_df = train_test_split(feedback_df, test_size=0.2, random_state=42)

# Separate features and labels
train_texts, train_labels = train_df['Expanded_Comment'].tolist(), train_df['Survey_Score'].tolist()
test_texts, test_labels = test_df['Expanded_Comment'].tolist(), test_df['Survey_Score'].tolist()
test_sentiment_scores = test_df['Sentiment_Score'].tolist()  # Extract Sentiment Scores for the test set


#-------------------------------------------
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


#-------------------------------------------
## 5. Build the TextCNN Model
# - A Convolutional Neural Network (CNN) is built for text classification.
# - The model includes an embedding layer, convolutional layers, pooling layers, and dense layers.

# Function to create the TextCNN model
def create_text_cnn(vocab_size, embedding_dim, max_sequence_length, filter_sizes, num_filters, dropout_rate):
    inputs = Input(shape=(max_sequence_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
    convs = []
    
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embedding)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    
    concatenated = tf.keras.layers.Concatenate()(convs)
    dropout = Dropout(dropout_rate)(concatenated)
    outputs = Dense(1, activation='linear')(dropout)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model


# Function to evaluate the model with different hyperparameters
def evaluate_hyperparameters(filter_sizes_list, num_filters_list, dropout_rate_list, batch_size_list):
    epochs_list = [10]
    results = []

    for filter_sizes in filter_sizes_list:
        for num_filters in num_filters_list:
            for dropout_rate in dropout_rate_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                        print(f'Evaluating: filter_sizes={filter_sizes}, num_filters={num_filters}, dropout_rate={dropout_rate}, epochs={epochs}, batch_size={batch_size}')

                        # Create the model
                        model = create_text_cnn(vocab_size=MAX_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_sequence_length=MAX_SEQUENCE_LENGTH, filter_sizes=filter_sizes, num_filters=num_filters, dropout_rate=dropout_rate)
                        
                        # Train the model and capture the validation error
                        history = model.fit(X_train, np.array(train_labels), epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

                        # Predict sentiment scores
                        predicted_scores = model.predict(X_test)
                        predicted_scores = np.clip(predicted_scores.round().astype(int), 1, 5).flatten().tolist()

                        # Calculate correlation
                        correlation_with_survey_score = np.corrcoef(predicted_scores, test_labels)[0, 1]
                        correlation_with_sentiment_score = np.corrcoef(predicted_scores, test_sentiment_scores)[0, 1]
                        
                        results.append({
                            'filter_sizes': filter_sizes,
                            'num_filters': num_filters,
                            'dropout_rate': dropout_rate,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'correlation_with_survey_score': correlation_with_survey_score,
                            'correlation_with_sentiment_score': correlation_with_sentiment_score,
                        })

    return pd.DataFrame(results)


# Remove epoch_list keep it only for 10
# Define hyperparameters to evaluate
filter_sizes_list = [[3, 4, 5], [2, 3, 4]]
num_filters_list = [64, 128]
dropout_rate_list = [0.5, 0.6]
batch_size_list = [4, 8]

# Evaluate the hyperparameters
results_df = evaluate_hyperparameters(filter_sizes_list, num_filters_list, dropout_rate_list, batch_size_list)

# Display the results using Databricks display function
display(results_df)


#-------------------------------------------
## 6. Optimal hyperparameters

# Find the best hyperparameters based on the highest correlation with survey score
best_result_1 = results_df.loc[results_df['correlation_with_survey_score'].idxmax()]
print('Best Hyperparameters:')
print(best_result_1)


#-------------------------------------------
## 7. Save the Sentiment Analysis Tuning Results

# Convert Pandas DataFrame to Spark DataFrame
results_spark_df = spark.createDataFrame(results_df)

# Save the Spark DataFrame to Delta format
results_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("dbfs:/FileStore/tables/john_marshman/Bhavya_pathak/hp_tuning_table")


#-------------------------------------------
## 8. Check the saved data and schema

# Load the hp_tuning_table DataFrame from the registered table
hp_tuning_table = (
    spark.read.format("delta")
    .option("header", "true")
    .load("dbfs:/FileStore/tables/john_marshman/Bhavya_pathak/hp_tuning_table"))

# Verify by showing the schema or some rows
hp_tuning_table.printSchema()
hp_tuning_table.limit(100).display()