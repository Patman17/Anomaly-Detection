# Proof of Concept: Abnormal Activity Detection
## **Shell GameChanger Challenge  @ TCO19** 

**TL;DR** - Competition to detect abnormal behavior of a major production platform and subsea system. Created a XGBoost-based model to detect and predict this anomalous activity and placed 3rd in this contest.

Below is the core materials to get main overview of project:  
1) [**Summary of Competition**](https://www.topcoder.com/challenges/30104182)  
2) [**Pitch Presentation to Shell**](https://docs.google.com/presentation/d/1Q1h2LjehEDgOPxxEo6CC_wfPmSaSW0G8er0yon9njhc/edit?usp=sharing)  
3) Refer to main Proof of Concept workbook - **1_POC_Dual_Model.ipynb**.


## **In-Depth Details**
### **Context / Scope of Problem**

Producing hydrocarbons from deep-water reservoirs is nothing short of a technological marvel. Safe and efficient operations of production platforms and subsea systems are of prime importance for Shell. These systems are highly instrumented, collecting thousands of parameters every second at each platform for ensuring monitoring, reliability and optimal operations. To use this data is a unique opportunity to apply the latest methods in Artificial Intelligence to generate insights.

### **Challenge Summary**

In **TCO19 Shell GameChanger Challenge**, we are presented with the opportunity to detect abnormal activity of one major production subsea system. This dataset will be comprised of roughly 200+ sensors (tempreature, pressure, flowrate etc.) across the whole production platfrom and subsea system. These sensors will be scaled and anonymized to protect the proprietary operations of Shell and to drive a data-focused approach rather then a business knowledge-focused one. In analyzing the data, we should be able to find key variables / factors that could detect and predict the onset of abnormal activity.

### **Dual-Model System**

Our team decided choose the simplest approach to the problem; a binary classification problem - **Normal vs. Abnormal**  
In this approach, we hope to design a non-proprietary, elegant solution to the problem.    

Additionally to fulfill the business needs, we decided on a **Dual-Model System**.  
1) The first model is the **"Warning" Model** that is trained to be sensitive to any abnormal activity and will predict the onset of any abnormal event.  
2) The second model is the **"Critical" Model** that is trained strictly to detect and identify when we are in the abnormal phase.

By having this dual model scheme we hope to have a robust **anomaly detector** that can work in tandem with a great **anomaly predictor**. This dual system can enable us to operationally act in the appropriate manner. Such as during the **"Warning" Phase** we can take the necessary precautions and react with moderate interventation. However if we are in the **"Critical" Phase**, immediate actions must be taken and severe measures might have to be implemented.

Please read through this document to get your environment set up and to get more details on all the files in this submission.

## Table of Contents
1) Getting Started   
2) Installing Environment  
3) File Descriptions  
4) Overall Process Flow for Reviewing Submission  
5) Proof of Concept details  
6) Tableau Operational Demo Dashboard details  
7) Holdout Test Workbook details  
8) Tuning Model details  
9) Summary of Findings   
10) Last Words  
11) Authors  

## 1) Getting Started

All data analysis and model creation have been done through usage of Python 3.7 and non-proprietary, open-source libraries. For the all workbooks, you will need to have a working python environment that utilize either Jupyter Notebook or Jupyter Lab.

The following list is all libraries utilized in creation of this Proof of Concept.
- Pickle
- Pandas
- Numpy
- Matplotlib
- Datetime
- scikit-Sklearn
- xgboost

## 2) Installing Environment
If you do not have any python environment setup please visit following link:

[**Anaconda**- popular Python distribution w/ most common python libraries](https://www.anaconda.com/distribution/#download-section)

Anaconda should be able to set up the environment with most of the packages installed and apps such as Jupyter Notebook that can be launched from their dashboard.

The only library that is not packaged in Anaconda is **xgboost**. Please pip install with the following line in the notebook.

```python
!pip install xgboost
```
Any additional missing libraries you can conda install or pip install the library from the library documentation.

## 3) File Descriptions
This submission consists of several type of files that includes workbooks and supporting files. 

### Notebooks
1) **1_POC_Dual_Model.ipynb** - Jupyter notebook that is the walkthrough of our Proof of Concept. 
2) **2_Holdout_Validation.ipynb** - Jupyter notebook that validate the holdout dataset.
3) **3_Tuned_Model.ipynb** - Jupyter notebook where we do some tuning on the model with holdout set.
### Model Files (**Note - these files will not be released**)
4) **Critical_model.pickle.dat** - the XGBoost "Critical" Model utilized in Proof of Concept and Holdout validation
5) **warning_model.pickle.dat** - the XGBoost "Warning" Model utilized in Proof of Concept and Holdout validation
6) **Critical_model2.pickle.dat** - tuned XGBoost "Critical" Model utilized in Tuned Model workbook.
7) **warning_model2.pickle.dat** - tuned XGBoost "Warning" Model utilized in Tuned Model workbook.
### Misc Files
8) **utils2.py** - contains all functions for data manipulation, plotting and scoring for both workbooks.
9) **Tableau - Tco 2019-11-14 01-03-27-1.m4v** - video file displaying example implementation of model in practice.
### Provided Dataset (**Note - these datasets will not be released**)
10) **training.csv** - provided, unmodified training data
11) **validation.csv** - provided, unmodified test data
12) **holdout.csv** - corrected holdout dataset
## 4) Process Flow for Viewing Submission
Here is the guideline that we suggest you follow to view and evaluate our submission.

1) **Proof of Concept** - please review the **1_POC_Dual_Model.ipynb** first as this will walk through our dual model and give you the results of the model and a frame of reference for the rest of the submission.
2) **Tableau Ops Demo Dashboard** - after seeing how the model works, you can now see an example implementation through a tableau dashboard. It is called **Tableau - Tco 2019-11-14 01-03-27-1.m4v**.
3) **Holdout Workbook** - a good frame of reference on the result of the dual model and how it could be implemented, review the validation in **2_Holdout_Validation.ipynb**
4) **Tuned Model** - finally review this last workbook to see our initial attempts at tuning the model with the addition of the holdout dataset.

## 5) Proof of Concept details
Again this is a Jupyter Notebook that walks through the Proof of Concept for our Dual Model. The purpose of this workbook is to show you the brief outline of the creation of the model and then display how well the model fulfills the criteria. 
### Structure of Workbook
1) Load Libraries - loading all necessary tools and libraries to implement workbook.
2) Import Data - retrieve the dataset
3) Defining Abnormal Events - encoding the event times as target for the model validation. 
4) **Part 1: "Warning Model"**
  a) Feature Selection
  b) XGBoost modeling
  c) Scoring and Validation
5) **Part 2: "Critical Model"**
  a) Feature Seletion
  b) XGBoost modeling
  c) Scoring and Validation
6) **Summary** - the results of the model. 

### Scoring and Validation
An important aspect of this workbook is how we evaluate the model. Per dataset, we will evaluate the model on both business needs and the correct metrics for a classifer. The metrics are as follows:  
- **ROC-AUC:** ability to rank between the two classes of Normal or Abnormal  
- **Balanced Accuracy:** accuracy score accounting for imbalanced classes that is defined as the average of recall obtained on each class  
- **Confusion Matrix:**  
    - Displays the # of True Positives, True Negatives, False Positives and False Negatives  
    - Format is as follows:  
    [[True Negative  False Positive]  
     [False Negative True Positive]]  
- **Classification Report:**  
    - Overall classification metric scorecard.  
    - Display precision, recall and f1-score.      
- **Prediction vs Actuals Graphs**
  - Plot of the model predictions represented as *Normal* = 0 and *Abnormal* = 1 over the dataset period compared to the Actual Values defined by the Event Times.
  - Orange = Model Predictions  
  - Blue = True Values (**Note: We just scaled the Actual values to 2 for better graphical visiblity**)    
- **Feature Importances**    
    - Bar chart explaining what variables are important to the model and the magnitude of their importance to the model as well.    
- **Times Detected**  
    - Shows beginning and end timestamps of each event.   
    - In addition, shows the time that the model predicts that the abnormal event is starting.
## 6) Tableau Operational Demo Dashboard details
**The Operational Dashboard** is a prototype of the visualization tool that we recommend for the operational team. The design principles considered when creating the dashboard are: 

- **Simplicity**: Display the words Normal, Warning, and Critical at the top to quickly inform the operator of current status without any need to interpret data being collected.
- **Transparency**: Display the raw data coming in from the top sensors that affect the possibility of an abnormal event. This will allow the operator to evaluate / troubleshoot the situation when a Warning sign comes up.
- **Accuracy**: Display minute by minute data, directly from the sensors, to give the operator the most recent information on the system.

The models that we developed are fed into the dashboard, allowing it to determine the status of the subsea system. Without any context of for the variables, the prototype is very basic. We hope that with more context, the dashboard can be improved to quickly provide all the necessary information that an operator needs to assess and act on situations that may arise. The effectiveness of any predictive model depends on how the model can be implemented into the real work  environment.  
## 7) Holdout Test Workbook details
This is the workbook to validate the holdout dataset that uses similar format as the Proof of Concept. It will use models that was only trained on the train.csv to predict the holdout values. 
## 8) Tuned Model Workbook
With the hasty release of the holdout data, we try to tune the model with the holdout set as well. 

## 9) Summary of Findings
- Overall, both models are very stable on the dataset as a classifier with a **ROC AUC score = 0.99**.
- Both models were able to predict ahead of the abnormal event. However, stability is still an issue with our model and we need to do some more tuning on this. We did find that certain variables influenced the abnormal events at different instances and severity.
- The **"Critical" Model** was able to more strictly adhere to the abnormal phases and reduce the amount of false positives shown in "Warning" Model.
- Together, they cooperatively work to predict and detect anomalous behavior in the system.

Given more time we can furthur improve various aspects of the models such as:

- **Severity Analysis**: analyze the duration of various event to furthur improve accuracy
- **Fine tuning**: adjust to the parameters and better analysis of the transitional period. We believe it might be a bit overfit since we have a small dataset. More data can help improve this.
- **Model Simplification** - XGBoost model could be replaced with a cheaper model such as Logisitic Regression if we build in the feature interactions.

## 10) Last Words 
Thanks again for **Shell** and **topcoder** for hosting this competition. It was an interesting problem to tackle. Especially since everything was unknown we had to utilize different agnostic methods.   

## 11) Authors
- **Eddie Yuen**:   
a) [LinkedIn Profile](https://www.linkedin.com/in/edward-yuen-995145a8/)  
- **Patrick Ly**:   
a) [LinkedIn Profile](https://www.linkedin.com/in/patrick-m-ly/)  
b) [Github Profile](https://github.com/patman17)

Feel free to contact us if you have any questions!





