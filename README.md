# Big Data Analytics Project 🎓 👨🏻‍💻

## MaLuCS Team - Academic Year 2020-2021

This repository contains all the milestones implemented during the course "Big Data Analytics" @ the University of Pisa.
The team that worked on this project is the "MaLuCS" team which was composed by:

- Luca Corbucci
- Cinzia Lestini
- Marco Giuseppe Marino
- Simone Rossi

The goal of course was to develop a big data analytics project.
The projects were based on real-world datasets covering several thematic areas.

The project is divided into 3 main milestones:

* Data Understanding and Project Formulation
* Model(s) construction and evaluation
* Model interpretation/explanation

At the end of each of these milestones, we presented our results and we wrote a report.
At the end of the course, we developed a final notebook to show the results reached during all the midterm.

### Folder structures

- There is a folder for each Midterm, in each of these folders you can find a Jupyter Notebook, a dataset and the slides of the presentation.
- There is a folder called "Final Term" that contains the final notebook and the code of the Streamlit web app.
- There is a folder called "Report" which contains the report we wrote for the exam.

## Final Term

### Notebook

Inside the Jupyter notebook you can find all the most important task of our project:

- Data Cleaning: this part was developed during the first Midterm.
- Prediction: this part was developed during the second Midterm.
- Explanation: this part was developed during the third Midterm.

In the second box into our notebook we import the sample dataset, you can substitute the name "./sample_data_txt.txt" with the name of your file.

## Streamlit

We developed a simple web app using Streamlit to visualize our work.

In the web app, you can upload the sample dataset and then you will see the same pieces of information that you can compute in the notebook.

In the bottom of the page, you can select an instance of the dataset to see the explanation.

You can visualize the web app using this link: http://62.171.188.29:8501/.

Alternatively, you can host on your own computer, we used Docker to simplify the execution of this service:

#### Run Streamlit using Docker

Run `docker-compose up` in your terminal to run `src/main.py` in Streamlit, then open [localhost:8501/](http://localhost:8501/?name=main) in your browser to visualize our project.
