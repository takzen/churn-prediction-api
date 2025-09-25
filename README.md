# Customer Churn Prediction API with FastAPI

### A machine learning project to build and deploy a real-time API for predicting customer churn.

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-blue.svg)
![Render](https://img.shields.io/badge/Deployed%20on-Render-blueviolet.svg)

## Project Overview

This project demonstrates the final, critical step in the machine learning lifecycle: **deployment**. It takes a trained Scikit-learn model for predicting customer churn and exposes it as a robust, real-time REST API using the **FastAPI** framework.

The project is designed to be a standalone, production-ready microservice that can be queried by other applications to get instant churn predictions.

## Live API Endpoint

The API is deployed on Render and is publicly accessible.

*   **Base URL:** `TUTAJ_WKLEIMY_LINK_PO_WDROZENIU`
*   **Interactive Docs (Swagger UI):** `TUTAJ_WKLEIMY_LINK/docs`

You can test the API by sending a POST request to the `/predict` endpoint.

## Key Features & Skills Demonstrated

*   **API Development:** Building a clean, fast, and well-documented API using **FastAPI**.
*   **Model Serving:** Loading a pre-trained machine learning model and scaler (`.joblib` artifacts) into an application to serve live predictions.
*   **Data Validation:** Using **Pydantic** models to define data schemas and ensure that incoming requests have the correct structure and data types.
*   **Cloud Deployment:** Preparing a Python application for deployment on a cloud platform (PaaS).
*   **End-to-End MLOps:** Showcases a simple but complete MLOps (Machine Learning Operations) cycle: from training and saving a model to deploying it as a live service.

## How to Run This Project Locally

1.  **Clone the repository and set up the environment.**
2.  **Generate model artifacts** by running the training script:
    ```bash
    python train_model.py
    ```
3.  **Run the FastAPI server locally:**
    ```bash
    uvicorn main:app --reload
    ```
4.  Access the interactive documentation at `http://127.0.0.1:8000/docs`.

## How to Deploy

This application is configured for easy deployment on platforms like Render:
1.  Create a new "Web Service" on Render and connect it to your GitHub repository.
2.  Set the "Start Command" to the command defined in the `Procfile`: `uvicorn main:app --host=0.0.0.0 --port=${PORT}`
3.  Render will automatically install dependencies from `requirements.txt` and deploy the application.