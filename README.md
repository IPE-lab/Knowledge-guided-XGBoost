# **Knowledge-guided XGBoost**
## Running screenshots show
- **Bottomhole pressure prediction results for well 2**
  <img src="img/Bottomhole pressure prediction results for well 2.jpg" width="400" />
***
## Paper Support
- Original information: Knowledge-Guided Machine Learning Method for Downhole Gauge Record Prediction in Deep Water Gas Field
- Recruitment Conference: Offshore Technology Conference Asia, 2024 (EI)
- Original DOI: https://doi.org/10.4043/34844-MS
***
## Description of the project
Deepwater gas field uses permanent downhole gauges (PDG) to monitor pressure, but PDGs are prone to failure under high temperatures and pressures, with high maintenance costs. To address this, a method for predicting bottomhole pressure combining physical models and machine learning is proposed. First, historical temperature and pressure data from the wellhead and bottomhole were collected, and a mechanistic model based on wellbore multiphase flow theory was established to describe the flow behavior of the wellbore pipe. Next, data-driven bottomhole pressure prediction models were constructed using XGBoost. Finally, based on these two models, a knowledge-guided machine learning (KGML) model was established. Domain knowledge based on the physical model was incorporated as adaptive weights into the loss function to ensure adherence to physical constraints during the training process.
Inspired by the study of (Ma et al., 2022), the loss functions of KGML methods (Eq.11) added an adaptive weighting factors. The adaptive weights allow the model to discern the significance of various samples. For instances where there are larger prediction errors in physical knowledge, higher weights are assigned. Consequently, the model focuses its efforts on fitting these samples.
***
## Functions of the project
The file uses XGBoost with added theoretical constraints for bottomhole pressure prediction. The theoretical knowledge comes from the calculation results of the mechanistic model, which have already been calculated and are included in the dataset. The main function description of the .py file is as follows:
1. Data Loading and Preprocessing Function: The function reads a CSV file into a DataFrame, extracts the input features (X) and target variables (y_i and y_emp). Here, y_i represents the ground truth, and y_emp is the calculated value based on the mechanistic equation.
2. Alpha and Gamma Calculation Function: This function calculates the alpha_DK and gamma_DK values used in the custom loss function. Alpha_DK is the absolute difference between the actual and predicted values divided by the actual value, and gamma_DK is the number of data points divided by the sum of alpha_DK.
3.	XGBoost Model Training Function: This function trains an XGBoost regression model using a custom loss function. It creates a DMatrix from the training data, defines the model parameters, and then defines the custom loss function. The xgb.train function is used to train the model with the specified parameters and the custom loss function.
4.	Model Evaluation Function: This function evaluates the trained model on the test set. It creates a DMatrix from the test data, makes predictions with the trained model, and calculates the Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-squared (R²).
***
## The operating environment of the project
-	Python == 3.9.2
-   XGBoost == 2.0.2
-	NumPy ==1.21.5
-	Pandas == 1.4.4
-   scikit-learn == 1.0.2
***
