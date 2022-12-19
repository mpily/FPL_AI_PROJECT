"""
# My first app
Here's our first attempt at using data to create a table:
"""
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
"""
# PRESENTATION STRUCTURE
1.  Description of Problem:
    - Problem statement (History of FPL how points are assigned)
    - Applications outside football

2. Methodology:
    - Data collection and cleaning
    - Selecting best parameters to use
    - Training models used

3. Results:
    - Accuracy on test data for different models.
    
4. Future Work:
    - Next step for project
"""


st.write('# 1. Problem Statement')
st.write('Design a model that will predict the number of points a player will get in a particular week in Fantasy Football')
image_top_coaches = Image.open('images/TopCoaches.png')
st.image(image_top_coaches)
image_recruitment = Image.open('images/recruitment_process.png')
image_scientific_process = Image.open('images/ScientificProcess.png')
"""
Models and techniques developed in solving this particular problem may be applied in the following areas:
1.  In recruitment by HR for different companies.
"""
st.image(image_recruitment)
'''
2. Scientific & Engineering Exploration
'''
st.image(image_scientific_process)
st.write('# 2. Methodology')
st.write('# 2.1 Data Collection')
'''
Every week  FPL producesa list of parameters about a particular player. This includes facts such as:

    1. How likely they are to play.

    2. Their average points per game so far.

    3. Any news about the player

    4. Which team the player is playing against.

    5. Several other statistics about the player.

A total of 83 datapoints are given for each player. 
The data is usually raw and of various types including boolean values, categorical data, numerical data, statements,
date and time and very many blanks.
Most models we were intending to use could not accomodate all these types and needed numerical data.
What we needed to do therefore was convert all the data as well as fill in the blanks. The strategies we used include:

    1. Using days since a particular set date for date/time values.
    
    2. Using integers to represent categorical data

    3. Filling in blanks with mean value.
'''

st.write('# 2.2 Selecting best parameters')

'''
After eliminating some of the features that did not have much of a baring such as the names, 43 features were left. The graph below shows their level of importances.

'''
feature  = {'red_cards_ex': 0.0010416827168238775, 'mode clean_sheets 3': 0.0014335970778390796, 'count 3 bonus 1': 0.0016994459196854113, 'mean assists 3': 0.001955857029443946, 'was_home': 0.0020183825357893417, 'mean goals_scored 3': 0.002038202548580626, 'mean bonus 3': 0.002114788763412199, 'count 3 yellow_cards 1': 0.002401066091519268, 'count 3 result draw': 0.0027051699297487826, 'count 3 result win': 0.0028316828486129953, 'count 3 result loss': 0.0029344672121808238, 'count 3 was_home True': 0.0029645974178867003, 'mode result 3': 0.003273074676439076, 'goals_scored_ex': 0.004087442083744496, 'assists_ex': 0.0042187139145892045, 'bonus_ex': 0.005249707816908287, 'yellow_cards_ex': 0.006143584805144236, 'kickoff_hour': 0.006587056892144186, 'mean result_difference 3': 0.0069626159843904425, 'team_last_position': 0.008007908616087302, 'threat_ex': 0.008098530730785227, 'opponent_last_position': 0.008559100656986927, 'clean_sheets_ex': 0.009093476951544947, 'creativity_ex': 0.009413086417900195, 'goals_conceded_ex': 0.009456660377823589, 'ict_index_ex': 0.010589845739421948, 'GW': 0.01184843769548998, 'total_points_ex': 0.01209547008024878, 'bps_ex': 0.012666679549518514, 'mean value 3': 0.014712066970953164, 'minutes_ex': 0.01751082655768938, 'influence_ex': 0.017941528255724738, 'value': 0.018819813775895932, 'mean goals_conceded 3': 0.028699146868287253, 'std minutes 3': 0.029219148332571485, 'mean threat 3': 0.04184889954135761, 'mean bps 3': 0.06414188678634167, 'mode minutes 3': 0.07492150521992096, 'count 3 minutes 1': 0.08265679035984075, 'mean total_points 3': 0.10067467923570006, 'mean creativity 3': 0.11350398307519098, 'mean ict_index 3': 0.11618802995142967, 'mean influence 3': 0.1166713619884059}
feature_names = list(feature.keys())[33:]
feature_values = list(feature.values())[33:]
feature_df = pd.DataFrame({
    'index': feature_names,
    'importance': feature_values,
}).set_index('index')
#feature_df = pd.DataFrame(feature_values,x=feature_names)

'''
The feature importances can be obtained using 2 methods, namely:
    
1. Using sklearn.random_forest.feature_importances_ . 
    -This returns the feature importances computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
'''
st.bar_chart(feature_df)
'''
    2. Using sklearn.SelectKBest. This performs a X^2 test to determine the importance of each feature.
'''
best_features = ['value' ,'total_points_ex', 'minutes_ex' ,'goals_conceded_ex', 'influence_ex',
 'bps_ex', 'ict_index_ex', 'clean_sheets_ex' ,'mode minutes 3' ,'mean bps 3',
 'mean creativity 3', 'mean ict_index 3', 'mean influence 3' ,'mean value 3',
 'mean threat 3' ,'mean goals_conceded 3', 'mean total_points 3',
 'std minutes 3', 'count 3 bonus 1', 'count 3 minutes 1']
st.write(best_features)

'''
Selecting the best features helps to improve the performance of a model in various ways.  First it reduces the computation cost especially in cases 
when dataset is large. Secondly it helps remove parameters which do not contribute much to the result.
'''
st.write('# 2.3 Models Used')
'''
The problem was split into two parts, first we required to determine whether a player would play or not. This is a categorisation problem and logistic regression was used
Then among the player who will play determine the number of points each player will get .This problem lends itself well to the use of regression models. For this approach, the following models were used:

    1. Linear Regression.
    2. Random Forest Regressor.
    3. Neural Network.
'''

st.write(' # 3 Results')
st.write('# 3.1 Linear Regression Model')
'''
Max accuracy reached: 1.9103 MAE
'''
error_trend= [1.836831055227187, 1.837839995904934, 1.842251172537868, 1.8457155152896294, 1.8466976458058515, 1.8462208377337994, 1.8463396118782003, 1.845603543359826, 1.84697885246783, 1.8473300579223264, 1.8469422189200535, 1.8468010737680225, 1.8485790917190466, 1.848895727214192, 1.849098826607906, 1.8695114438955358, 1.8697802846592548, 1.8685361884692195, 1.8686891621715285, 1.870059351493661, 1.870995187688226, 1.8716343722247646, 1.8720270832006074, 1.8726375377295041, 1.8739351768289267, 1.8762955608809353, 1.8859287568988043, 1.8865082909357274, 1.8845585300776926, 1.8864618798722514, 1.8864972259135913, 1.8882197322683805, 1.8887805924521144, 1.8832544158527451, 1.9136730344197752, 1.9103810893508069, 1.9103302259333386]
st.line_chart(error_trend)
'''
As can be seen with increasing number of parameters, the score gradually increases. A decision can be made on how good a score we want to have.
'''


st.write('# 3.2 Random Forest Model')
'''
Max accuracy reached: 1.919 MAE
'''

st.write('# 3.3 Neural Network Model')
'''
Max Accuracy reached after 200 epochs of training by 4 layer model : 1.96531 MAE
'''

'''
The values are still far from required in order to be used.  A lot of improvement could be done to the neural network model
We aim for MAE being around 0.8.
'''

st.write('# 4 Future works')
'''
    1. Select Team based on values predicted.
    2. Register Team and test it against other player around the world.
    3. Apply model to other areas.
'''
