# digit-classifier
Package for solving handwritten digits classification with different type of models 

Package has three models:
- RandomClassifier
- CnnClassifier
- RfClassifier

You can choose model and get result for test data by runnig main.py.

Structure of image_classifier package:
│   __init__.py                                                                                                         
│                                                                                                                       
├───dataset                                                                                                             
│       mnist.py                                                                                                        
│                                                                                                                       
└───models                                                                                                                    
    │   model.py                                                                                                            
    │   __init__.py   
    │                                                                                                      
    ├───cnn                                                                                                                 
    │     cnn_model.py                                                                                                    
    │     __init__.py                                                                                                     
    │                                                                                                                       
    ├───random                                                                                                                  
    │     random_model.py 
    │     __init__.py                                                                                                     
    │                                                                                                                       
    └───rf                                                                                                                          
          rf_model.py                                                                                                             
          __init__.py   

Dataset folder has function for downloading mnist dataset.
Models folder has DigitClassificationInterface class in model.py and each model (cnn, rf, rand) in individual folder.
If you want to add new model to package you have to:
	1. create new folder in /image_classifier/models
	2. add import of new model to __init__.py from /image_classifier/models
	3. add new condition in __init__.py in /image_classifier. This gives you opportunity to create instance of new class in main.py  