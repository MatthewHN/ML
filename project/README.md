Salary estimator for tech positions in the USA

A Kaggle database was used with a sample of over 6,500 employees, for whom the following information was provided:
-	Age
-	Gender
-	Level of education
-	Position
-	Years of experience in the position
-	Salary

The model analyzes the data to look for patterns and correlations, so that if it were given some or all of the first five parameters, it could ultimately predict the last one -- the worker’s salary. 

It is possible to try out the model and predict a salary in the Predictions section of the Jupyter Notebook.
The model employs a DataFrame to process the input variables, which include the candidate's position and years of experience. 
A binary encoding scheme marks the active job title with a '1', while all others are set to '0', effectively tailoring the prediction to the specific role in question.
