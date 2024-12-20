# Sentiment Analysis using Deep Learning

## Project Overview
This project leverages Natural Language Processing (NLP) and deep learning to analyze customer feedback for NEXT plc. Using TextCNN, the framework identifies sentiment at the aspect level, offering actionable insights to enhance business strategies and customer satisfaction. With over 700,000 textual comments analyzed, this work highlights the potential of sentiment analysis to improve customer experience and operational efficiency.

---

## Motivation
Understanding customer sentiment at a granular level is crucial for modern businesses to stay competitive. Traditional sentiment analysis approaches often fail to capture aspect-specific sentiments (e.g., product quality vs. delivery service). This project fills that gap by implementing an Aspect-Based Sentiment Analysis (ABSA) framework tailored for NEXT plc’s vast customer feedback dataset.

---

## Key Features
- **Aspect-Level Sentiment Analysis:** Breaks down customer feedback to analyze sentiments specific to different aspects like product quality and customer service.
- **Deep Learning with TextCNN:** Utilizes TextCNN for efficient and accurate sentiment scoring.
- **Actionable Insights:** Identifies high-performing and underperforming agents, departments, and processes.
- **Scalable Framework:** Designed to handle over 700,000 textual records efficiently.

---

## Tools and Technologies
- **Programming:** Python
- **Deep Learning:** TensorFlow, Keras
- **NLP:** SpaCy, GloVe, NLTK
- **Data Analysis and Visualization:** pandas, NumPy, Matplotlib, Seaborn
- **Big Data Processing:** Databricks, PySpark
- **Machine Learning:** scikit-learn

---

## Results
- Achieved a Mean Absolute Error (MAE) of 0.35 and a correlation score of 0.73 with customer-provided survey ratings.
- TextCNN outperformed baseline models like Gradient Boosting in sentiment prediction accuracy.
- Identified actionable discrepancies in customer feedback and ratings, leading to recommendations for improving NEXT plc’s rating system.

---

## Project Workflow
1. **Data Preprocessing:**
   - Cleaning and tokenizing text.
   - Expanding abbreviations and removing noise.
   - Preparing data for sentiment analysis with TextCNN.

2. **Model Development:**
   - Implementing TextCNN for aspect-level sentiment analysis.
   - Comparing performance with baseline models (e.g., Gradient Boosting).

3. **Evaluation:**
   - Using metrics like MAE, precision, recall, and F1 score.
   - Visualizing results with confusion matrices, word clouds, and correlation heatmaps.

4. **Insights & Recommendations:**
   - Highlighting top-performing agents and departments.
   - Proposing changes to improve customer rating systems.

---

## Future Work
- **Multilingual Sentiment Analysis:** Extend the framework to analyze feedback in multiple languages.
- **Hybrid Models:** Explore advanced architectures like hybrid models combining CNNs with transformers.
- **Customer Retention Analysis:** Investigate the relationship between sentiment and customer loyalty.

---

## License
This project is protected under an All Rights Reserved License. Unauthorized copying, distribution, or usage of the code, in whole or in part, is strictly prohibited without prior written consent from the author.

---

## Acknowledgments
Special thanks to NEXT plc and Lancaster University for providing the dataset and the opportunity to work on this impactful project.

