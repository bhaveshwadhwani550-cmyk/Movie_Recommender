# 🎬 Movie Recommender System

A Python-based movie recommendation system built with Streamlit that implements both **Content-Based Filtering** and **Collaborative Filtering** techniques to suggest movies to users.

## 📋 Overview

This interactive web application demonstrates two popular recommendation algorithms:
- **Content-Based Filtering**: Recommends movies similar to a selected movie based on genres and descriptions
- **Collaborative Filtering**: Recommends movies based on ratings from similar users using K-Nearest Neighbors (KNN)

## ✨ Features

- 🎯 **Content-Based Recommendations**: Find movies similar to your favorites
- 👥 **Collaborative Filtering**: Get personalized recommendations based on user behavior
- 📊 **Popular Movies**: View top-rated movies with minimum rating thresholds
- 📈 **Interactive UI**: Easy-to-use Streamlit interface
- 🔄 **Real-time Processing**: Instant recommendations with adjustable parameters
- 📁 **Sample Dataset**: Includes pre-loaded movie and rating data for testing

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. Clone the repository (or download the files)
```bash
git clone <your-repo-url>

Install required packages
Bash

pip install -r requirements.txt
Run the application
Bash

streamlit run app.py
The app will open in your default browser at http://localhost:8501

📦 Requirements
Create a requirements.txt file with the following dependencies:

txt

streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.2

movie-recommender/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
└── (optional future additions)
    ├── data/
    │   ├── movies.csv    # Extended movie dataset
    │   └── ratings.csv   # Extended ratings dataset
    └── models/           # Saved models

Future Enhancements
 Upload custom datasets via UI
 Add movie posters and images
 Implement hybrid recommendation (combine both methods)
 Add user authentication and rating submission
 Include more advanced algorithms (Matrix Factorization, Deep Learning)
 Export recommendations to CSV/PDF
 Add movie search and filtering
 Include movie metadata (year, director, cast)
🐛 Troubleshooting
Issue: "Load data first" error

Solution: Click "Load sample data" in the sidebar
Issue: No recommendations shown

Solution: Ensure setup buttons are clicked or reload the page
Issue: ModuleNotFoundError

Solution: Install all requirements: pip install -r requirements.txt
📚 Technologies Used
Streamlit: Web application framework
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Scikit-learn: Machine learning algorithms (TF-IDF, KNN, Cosine Similarity)
SciPy: Sparse matrix operations

📄 License
This project is open source.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
👨‍💻 Author
Your Name - Bhavesh Wadhwani
