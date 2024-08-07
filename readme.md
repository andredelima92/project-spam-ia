# Spam Message Analyzer

## Introduction

The Spam Message Analyzer project is designed to classify messages as spam or ham (not spam) using a machine learning model. This project uses a Random Forest Classifier to train the model and TfidfVectorizer to preprocess the text data. It includes functionality to save the trained model for future use, avoiding the need to retrain the model every time it's needed.

## Table of Contents

- [Spam Message Analyzer](#spam-message-analyzer)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model (optional):](#training-the-model-optional)
    - [Run the Model:](#run-the-model)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/spam-message-analyzer.git
    cd spam-message-analyzer
    ```

2. **Create a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Model (optional):

To train the model and save it, run:

```sh
python train_model.py
```

### Run the Model:

```sh
python run_model.py
```