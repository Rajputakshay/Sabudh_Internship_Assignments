import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request


def dataPreprocessing(data):
    
    dataframe = pd.DataFrame(columns=[
       'club_Bournemouth', 'club_Brighton+and+Hove', 'club_Burnley',
       'club_Chelsea', 'club_Crystal+Palace', 'club_Everton',
       'club_Huddersfield', 'club_Leicester+City', 'club_Liverpool',
       'club_Manchester+City', 'club_Manchester+United',
       'club_Newcastle+United', 'club_Southampton', 'club_Stoke+City',
       'club_Swansea', 'club_Tottenham', 'club_Watford', 'club_West+Brom',
       'club_West+Ham'], data=[[0]*19])

    dataframe[['position_CB', 'position_CF', 'position_CM',
           'position_DM', 'position_GK', 'position_LB', 'position_LM',
           'position_LW', 'position_RB', 'position_RM', 'position_RW',
           'position_SS']] = pd.DataFrame([[0]*12], index=dataframe.index)

    dataframe[['position_cat_Defenders', 'position_cat_Goalkeeper',
           'position_cat_Midfielders']] = pd.DataFrame([[0]*3], index=dataframe.index)

    dataframe[['region_EU', 'region_England',
           'region_Rest of World']] = pd.DataFrame([[0]*3], index = dataframe.index)
    
    dataframe['age'] = [int(data['age'])]
    dataframe['page_views'] = [int(data['page_views'])]
    dataframe['fpl_value'] = [float(data['fpl_value'])]
    dataframe['fpl_sel'] = [float(data['fpl_sel'])]
    dataframe['fpl_points'] = [int(data['fpl_points'])]
    dataframe['new_foreign'] = [int(data['new_foreign'])]
    dataframe['new_signing'] = [int(data['new_signing'])]
    dataframe['big_club'] = [int(data['big_club'])]
    
    
    if 'club_'+data['club'] in dataframe.columns:
        dataframe['club_'+data['club']] = 1
        
    if 'position_'+data['position'] in dataframe.columns:
        dataframe['position_'+data['position']] = 1
        
    if 'position_cat_'+data['position_cat'] in dataframe.columns:
        dataframe['position_cat_'+data['position_cat']] = 1
        
    if 'region_'+data['region'] in dataframe.columns:
        dataframe['region_'+data['region']] = 1
        

    dataframe['page_views'] = dataframe['page_views'].apply(np.log)
    dataframe = dataframe[['age', 'page_views', 'fpl_value', 'fpl_sel', 'fpl_points',
       'new_foreign', 'big_club', 'new_signing', 'club_Bournemouth',
       'club_Brighton+and+Hove', 'club_Burnley', 'club_Chelsea',
       'club_Crystal+Palace', 'club_Everton', 'club_Huddersfield',
       'club_Leicester+City', 'club_Liverpool', 'club_Manchester+City',
       'club_Manchester+United', 'club_Newcastle+United', 'club_Southampton',
       'club_Stoke+City', 'club_Swansea', 'club_Tottenham', 'club_Watford',
       'club_West+Brom', 'club_West+Ham', 'position_CB', 'position_CF',
       'position_CM', 'position_DM', 'position_GK', 'position_LB',
       'position_LM', 'position_LW', 'position_RB', 'position_RM',
       'position_RW', 'position_SS', 'position_cat_Defenders',
       'position_cat_Goalkeeper', 'position_cat_Midfielders', 'region_EU',
       'region_England', 'region_Rest of World']]
    return dataframe


app = Flask(__name__ ,template_folder='templates')

@app.route('/crash')
def main():
    raise Exception()

@app.route('/')
def index():
    return flask.render_template('webpage.html')


    
def ValuePredictor(to_predict):
    scaler = pickle.load(open('.\model\scaler.pkl', 'rb'))
    x = scaler.transform(to_predict)
    loaded_model = pickle.load(open('.\model\model.pkl','rb'))
    result = loaded_model.predict(x)
    
    return result



@app.route('/predict',methods = ['POST'])
def result():
    if request.method == 'POST':
        data = request.form.to_dict()
        pre_processed_data = dataPreprocessing(data)
        
        result = ValuePredictor(pre_processed_data)
        prediction = str(round(result[0],3))
    
    return render_template('predict.html',prediction=prediction)

if __name__ == '__main__':
    app.debug = True
    app.run(use_reloader=False)
