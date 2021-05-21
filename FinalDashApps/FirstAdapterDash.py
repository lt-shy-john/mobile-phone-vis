import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import *

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#==========================================================Get Data ================================================
#Read data file from excel spreadsheet
dataFileName = 'Mobile Device Data for Assignment 2.xlsx'
dataFrame = pd.read_excel(dataFileName)
dataArray = dataFrame.to_numpy().T
attributeArray = dataArray[4:]
attributeNames = dataFrame.columns.values[4:]
releaseYear = dataArray[2].astype(float)
modelNames = dataFrame['Model'].astype(str)
useLogScale = [True, True, True, False, False, False, False, False, False, False, False, False]

#Data to be used in scatter plots
scatterDatNames = np.array(['RAM', 'Storage', 'CPU', 'Diplay Diagonal', 'Pixel Density'])
scatterData = np.stack((attributeArray[0], attributeArray[1], attributeArray[2], attributeArray[3], attributeArray[11]), axis=1).T
epsilon = np.array([0.02, 0.01, 0.007, 0.002, 0.002])

#Get data about phone companies
deviceAndCompanySheet = pd.read_csv('USE THIS DATASET!!! Mobile Device Data Aligned with Company Name and ID.csv')
companyId = np.array(deviceAndCompanySheet['Company_ID']).astype(int)
companyNames = np.array(deviceAndCompanySheet['Company_real']).astype(str)
dropDownOptions, dropData, usedNames, count = [], [], [], 0
for i, name in enumerate(companyNames):
    if name in usedNames:
        j = usedNames.index(name)
        dropData[j].append(i)
    else:
        usedNames.append(name)
        dropDownOptions.append({'label': name, 'value': count})
        dropData.append([i])
        count += 1
dropData = [np.array(i).astype(int) for i in dropData]

#==========================================Create Dash App======================================================================
app = dash.Dash(__name__)

#Create a card for our controls
controls = dbc.Card([
    dbc.Row([
    dbc.Col(
        dbc.FormGroup([
            html.H1('Selected Attribute'),
            dcc.Dropdown(
                id = 'attOptions',
                options = [{'label': attName, 'value': i} for i, attName in enumerate(scatterDatNames)],
                value = 0)]),
        width=6),
    
    dbc.Col(
        dbc.FormGroup([
            html.H1('Select Company'),
            dcc.Dropdown(
                id = 'companySelection',
                options = dropDownOptions,
                multi=True
                )]),
        width=6)]),
    
    dbc.FormGroup([
        html.H1('Select Year Range:'),
        dcc.RangeSlider(
            id='yearSlider',
            min=releaseYear[0], 
            max=releaseYear[-1], step=0.1, 
            value=[releaseYear[0], releaseYear[-1]],
            marks={1991: '1991', 
                1995: '1995',
                2000: '2000', 
                2005: '2005',
                2010: '2010',
                2012: '2012'}),
        html.H1('Select Visualization Method:'),
        dcc.RadioItems(options=[
            {'label': 'Earliest Adapters', 'value': 'discrete'},
            {'label': 'Beating the Trend', 'value': 'continuous'},
            {'label': 'Closest to Median', 'value': 'median'}],
            value='discrete')
    ])
])

app.layout = html.Div([
    html.Div(controls),
    html.Div([
        dcc.Graph(id="scatter",  figure={'layout': {"height": 700}})]),
        dcc.Graph(id="bar-chart", figure={'layout': {"height": 700}}),
    html.Div(id='DebugText')
])

@app.callback(
    Output("scatter", "figure"),
    Output("bar-chart", "figure"),
    Output('DebugText', 'children'),
    [Input("attOptions", "value"),
    Input('yearSlider', 'value'),
    Input('companySelection', 'value')])
def update_bar_chart(i, yearRange, selectedCompanies):
    debugTex = ''
    
    #Filter out data into specified year range
    isInYearRange = ((releaseYear >= yearRange[0]) * (releaseYear <= yearRange[1])).astype(bool)
    years = releaseYear[isInYearRange]
    att = scatterData[i]
    att = att[isInYearRange]
    names = modelNames[isInYearRange].astype(str)
    
    #Do one dimensional clustering
    att = att.astype(float)
    if useLogScale[i]:
        att[att==0] = np.min(att[att!=0])
        attLog = normalizeList(np.log(att))
        debugTex = np.min(attLog)
        clusterID = DBSCAN(eps=epsilon[i]).fit(attLog.reshape(-1, 1))
    else:
        clusterID = DBSCAN(eps=epsilon[i]).fit(att.reshape(-1, 1))
    labels = clusterID.labels_
    
    firstInClusters, firstInClusterIDs = [], []
    arrayIndex = np.arange(len(att))
    for idx in np.unique(labels):
        isInCluster = (labels == idx)
        firstInClusterID = np.min(arrayIndex[isInCluster])
        if firstInClusters == []:
            firstInClusters.append([releaseYear[firstInClusterID], att[firstInClusterID]])
            firstInClusterIDs.append(firstInClusterID)
        if firstInClusters[-1][1] < att[firstInClusterID]:
            firstInClusters.append([releaseYear[firstInClusterID], att[firstInClusterID]])
            firstInClusterIDs.append(firstInClusterID)
    firstInClusters = np.array(firstInClusters)
    
    np.random.seed(seed=1)
    uniqueLabels = list(np.unique(labels))
    rand = np.random.rand(len(uniqueLabels))
    colors = np.array([rand[uniqueLabels.index(labe)] for labe in labels])
    colors[labels==-1] = 0
    
    #Create the scatter plot
    figScat = go.Figure()
    figScat.add_trace(go.Scatter(
        x=years, 
        y=att, mode='markers', 
        hovertemplate=names, 
        name=scatterDatNames[i],
        marker=dict(size=8, color=colors, colorscale='Turbo')))
    
    figScat.add_trace(go.Scatter(
        x = years[firstInClusterIDs],
        y = att[firstInClusterIDs],
        hovertemplate=names[firstInClusterIDs],
        mode='markers',
        marker=dict(size=20, color=colors[firstInClusterIDs], colorscale='Turbo')))
    
    figScat.update_layout(
        title="Earliest Adapters",
        font_size = 30,
        xaxis_title="Year",
        yaxis_title=scatterDatNames[i],
        transition_duration=500, 
        showlegend=False)
    
    compNames = []
    companyScores = []
    for idx in firstInClusterIDs:
        compName = companyNames[idx]
        if compName in compNames:
            j = compNames.index(compName)
            companyScores[j] += 1
        else:
            compNames.append(compName)
            companyScores.append(1)
    
    figBar = go.Figure()
    if (selectedCompanies == None) or (selectedCompanies == []):
        figBar.add_trace(go.Bar(x=compNames, y=companyScores))
        
    #Add title and specify other properties of the bar graph
    figBar.update_layout(
        title="Company Scores",
        font_size = 30,
        xaxis_title="Companies",
        yaxis_title="Score",
        transition_duration=500, 
        showlegend=False, 
        hovermode="x unified")
    
    if useLogScale[i]:
        figScat.update_yaxes(type="log")
    
    return figScat, figBar, debugTex#, figBar, debugTex
    

if __name__ == "__main__":
    app.run_server(debug=True, port=8052)

