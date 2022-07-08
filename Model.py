
####################### Import Libraries ################################

import mysql.connector
import pandas as pd 
import dtale 
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from autots import AutoTS
from sklearn.pipeline import Pipeline
import pickle

############### Extract data from MYSQL with python ##################### 

db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Pooja@19",
  database="project"
)
cur=db.cursor()
s= "select * from sales"
cur.execute (s)
table = cur.fetchall()
# Convert to DataFrame
df=pd.read_sql(s,db)

## Remove unwanted columns by using drop()
df.drop(['SalesYear', 'SalesMonth', 'SalesHour', 'WeekdayName'], axis = 1, inplace = True, )

############################## Auto EDA ###############################

# Dtale Auto EDA

def EDA ():
  report = dtale.show(df)
  report.open_browser()
  
########################## Data Pre-Processing ############################
  
def preprocessing ():
      
  # Convert DataFrame to datetime
      df["SalesDate"]=pd.to_datetime(df["SalesDate"])

  # To remove Duplicate records
      df.drop_duplicates()

  # Let's checking the missing values
      df.isna().sum()          # There is no missing values in this data
      
  # Let's checking outliers
      sns.boxplot(data=df)
      
# Capping outliers by using winsorizer techniques   
      
      def data_wrangling():
     
         cols=['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']

         for col in cols:
              print('winsorizing the',col)
              winsor= Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                                        tail='both', # cap left, right or both tails 
                                        fold=1.5,
                                        variables=[col])
              df[col]=winsor.fit_transform(df[[col]])
              
   # after winsorization
              for i, predictor in enumerate(cols):
                  plt.figure(i)
                  plt.boxplot(data=df, x=predictor)
                  
########################### Model Building ##################################                  

def model_building(Drug_name):
    
# AutoTS Model

# Split the train dataset
      train=df[-365:]

############################## M01AB #################################

      model_M01AB = AutoTS(
          forecast_length=7,
          frequency='infer',
          prediction_interval=0.95,
          ensemble='simple',
          model_list="fast",  # "superfast", "default", "fast_parallel"
          transformer_list="fast",  # "superfast",
          drop_most_recent=1,
          max_generations=4,
          num_validations=2,
          validation_method="backwards"
      )

      model_M01AB = model_M01AB.fit(train, date_col="SalesDate",value_col="M01AB", id_col=None)

# Print the details of the best model
      print(model_M01AB)
      
## Saving model to disk
      pickle.dump(model_M01AB, open('model_M01AB.pkl','wb'))
      model_M01AB = pickle.load(open('model_M01AB.pkl','rb'))
      
############################### M01AE ################################# 
    
      model_M01AE = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_M01AE = model_M01AE.fit(train, date_col="SalesDate",value_col="M01AE", id_col=None)

# Print the details of the best model
      print(model_M01AE)
      
## Saving model to disk
      pickle.dump(model_M01AE, open('model_M01AE.pkl','wb'))
      model_M01AE = pickle.load(open('model_M01AE.pkl','rb'))  
      
################################ N02BA #################################      
      
      model_N02BA = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
          
      model_N02BA = model_N02BA.fit(train, date_col="SalesDate",value_col="N02BA", id_col=None)

# Print the details of the best model
      print(model_N02BA)
      
## Saving model to disk
      pickle.dump(model_N02BA, open('model_N02BA.pkl','wb'))
      model_N02BA = pickle.load(open('model_N02BA.pkl','rb'))
      
############################## N02BE ##################################

      model_N02BE = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_N02BE = model_N02BE.fit(train, date_col="SalesDate",value_col="N02BE", id_col=None)

# Print the details of the best model
      print(model_N02BE)
      
## Saving model to disk
      pickle.dump(model_N02BE, open('model_N02BE.pkl','wb'))
      model_N02BE = pickle.load(open('model_N02BE.pkl','rb'))
      
################################ N05B #################################

      model_N05B = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_N05B = model_N05B.fit(train, date_col="SalesDate",value_col="N05B", id_col=None)

# Print the details of the best model
      print(model_N05B)
      
## Saving model to disk
      pickle.dump(model_N05B, open('model_N05B.pkl','wb'))
      model_N05B = pickle.load(open('model_N05B.pkl','rb'))
      
############################### N05C ###################################
      
      model_N05C = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_N05C = model_N05C.fit(train, date_col="SalesDate",value_col="N05C", id_col=None)

# Print the details of the best model
      print(model_N05C)
    
## Saving model to disk
      pickle.dump(model_N05C, open('model_N05C.pkl','wb'))
      model_N05C = pickle.load(open('model_N05C.pkl','rb'))
      
############################### R03 ####################################
      
      model_R03 = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_R03 = model_R03.fit(train, date_col="SalesDate",value_col="R03", id_col=None)

# Print the details of the best model
      print(model_R03)
      
## Saving model to disk
      pickle.dump(model_R03, open('model_R03.pkl','wb'))
      model_R03 = pickle.load(open('model_R03.pkl','rb'))
      
################################ R06 #################################
      
      model_R06 = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.95,
            ensemble='simple',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
      
      model_R06 = model_R06.fit(train, date_col="SalesDate",value_col="R06", id_col=None)

# Print the details of the best model
      print(model_R06)
      
## Saving model to disk
      pickle.dump(model_R06, open('model_R06.pkl','wb'))
      model_R06 = pickle.load(open('model_R06.pkl','rb'))
      
      return model_M01AB, model_M01AE, model_N02BA, model_N02BE, model_N05B, model_N05C, model_R03, model_R06 
  
pipe= Pipeline([("EDA",EDA()),("Preprocessing",preprocessing()),
                                 ("Model",model_building("M01AB")),
                                  ("Model",model_building("M01AE")),
                                   ("Model",model_building("N02BA")),
                                    ("Model",model_building("N02BE")),
                                     ("Model",model_building("N05B")),
                                      ("Model",model_building("N05C")),
                                       ("Model",model_building("R03")),
                                        ("Model",model_building("R06"))])














