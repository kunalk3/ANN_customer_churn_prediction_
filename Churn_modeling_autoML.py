#!/usr/bin/env python
# coding: utf-8

# # Churn Modeling

# <img src = "https://slitayem.github.io/img/blog/2020-08-04/churn.png" width=50%>

# ### **Install Pycaret**

# In[ ]:


get_ipython().system('pip install pycaret')


# ### **Importing the libraries**

# In[ ]:


#ImportLib
import warnings
warnings.filterwarnings('ignore')
from pycaret.classification import *
import pandas as pd


# ### **Load and Prepare Data**

# In[ ]:


data = pd.read_csv("Churn.csv")
data.head(6)


# In[ ]:


data.describe()


# In[ ]:


#Drop the unwanted columns
data.drop(['RowNumber','CustomerId','Surname'], axis=1 ,inplace = True)


# In[ ]:


data.columns


# ### **Setup Auto-ML Model**

# In[ ]:


classification = setup(data= data, target='Exited',
                             remove_outliers=True,
                             normalize=True,
                             normalize_method='robust',
                             silent = True)


# ### **Compare All Auto-ML Models**

# In[ ]:


compare_models() 


# ### **Top-3** **AutoML** **Models**

# In[ ]:


top_3 =compare_models(n_select=3)


# ### **Create Best Model**

# In[ ]:


best_model =compare_models(n_select=1)


# In[ ]:


top_3


# In[ ]:


best_model


# ### **Accuracy Score Top-3 Models**
# 

# | Model | Accuracy Score | 
# | --- | --- | 
# | Gradient Boosting Classifier | 0.8586  |
# |LLGBM Classifier | 0.8633 |
# | Random Forest Classifier | **0**.**8636** |

# ### **blending Top-3 Models**

# In[ ]:


rf = create_model('rf')


# In[ ]:


plot_model(estimator = rf, plot = 'auc')


# In[ ]:


plot_model(estimator = rf, plot = 'feature')


# In[ ]:


gbc = create_model('gbc')     
lgm  = create_model('lightgbm')          
blend = blend_models(estimator_list=[rf,lgm,gbc])


# ### **Accuracy Score** **Blending Model**
# 
# 

# | Model | Accuracy Score | 
# | --- | --- | 
# | blending Top 3 models | **0**.**8652**  |

# ### **Confusion Matrix**

# In[ ]:


plot_model(estimator = blend, plot = 'confusion_matrix')


# ### **Evaluate Blend Model**

# In[ ]:


evaluate_model(blend)


# ### **Make** **Prediction**

# In[ ]:


pred = predict_model(blend, data = data)


# In[ ]:


pred


# ### **Tuned Blend Model**

# In[ ]:


tuned_blend = tune_model(blend)


# ### **Best Accuracy Score!**

# | Model | Accuracy Score | 
# | --- | --- | 
# | Tuned blending Top 3 models | **0**.**8671**  |

# ### **Evaluate Tuned Model And Make Prediction**

# In[ ]:


evaluate_model(tuned_blend) 


# In[ ]:


pred_1 = predict_model(tuned_blend, data = data)


# In[ ]:


pred_1


# ### **Save And Load Best Model**

# In[ ]:


save_model(tuned_blend, model_name='Navid-Model')


# In[ ]:


NavidModel = load_model('Navid-Model')

