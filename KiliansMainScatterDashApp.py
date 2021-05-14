import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

#========================================================== General Functions ================================================
#Brings values to desired range (default being 0 and 1)
def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#Returns coordinates of a line that fits the data linearly and detrended attributes
def getLinearFit(years, att, useLog):
    
    #Use log scale if necessary
    att = att.astype(float)
    if useLog:
        att[att==0] = 0.0001
        att = np.log(att)
    
    #We get fit coefficients and create y values for a fitted line
    fitCoefficients = np.polyfit(years, att, 1)
    fitLeft = years[0] * fitCoefficients[0] + fitCoefficients[1]
    fitRight = years[-1] * fitCoefficients[0] + fitCoefficients[1]
    
    #If using an exponential fit, we convert our fit back to the exponential scale
    if useLog:
        fitLeft = np.exp(fitLeft)
        fitRight = np.exp(fitRight)
    
    #Create the final lists to return
    linearFitX = [years[0], years[-1]]
    linearFitY = [fitLeft, fitRight]
    detrendedAttributes = att - (fitCoefficients[0] * years + fitCoefficients[1])
    return linearFitX, linearFitY, detrendedAttributes

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

#Create the 'Screen to Body Ratio' attribute
width = np.copy(attributeArray[6]).astype(float)
length = np.copy(attributeArray[7]).astype(float)
width[width==0] = min(width[width!=0])
length[length==0] = min(length[length!=0])
screenToBodyRatio = attributeArray[4] * attributeArray[5] / (width * length)
screenToBodyRatio = normalizeList(screenToBodyRatio)

#Data to be used in scatter plots
scatterDatNames = np.array(['RAM', 'Storage', 'CPU', 'Pixel Density', 'Screen to Body Ratio'])
scatterData = np.stack((attributeArray[0], attributeArray[1], attributeArray[2], attributeArray[11], screenToBodyRatio), axis=1).T

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
            dbc.Label('Selected Attribute'),
            dcc.Dropdown(
                id = 'attOptions',
                options = [{'label': attName, 'value': i} for i, attName in enumerate(scatterDatNames)],
                value = 0)]),
        width=6),
    
    dbc.Col(
        dbc.FormGroup([
            dbc.Label('Select Company'),
            dcc.Dropdown(
                id = 'companySelection',
                options = dropDownOptions,
                multi=True
                )]),
        width=6)]),
    
    dbc.FormGroup([
        html.H3('Select Year Range:'),
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
        html.H3('Select Visualization Method:'),
        dcc.RadioItems(options=[
            {'label': 'Earliest Adapters', 'value': 'discrete'},
            {'label': 'Beating the Trend', 'value': 'continuous'}],
            value='continuous')
    ])
])

app.layout = html.Div([
    html.Div(controls),
    html.Div([
        #html.H3('Devices Beating the Trend'),
        dcc.Graph(id="bar-chart", figure={'layout': {"height": 700}}),
        #html.H3('Company Scores'),
        dcc.Graph(id="scatter",  figure={'layout': {"height": 700}})]),
    html.Div(id='DebugText')
])

@app.callback(
    Output("bar-chart", "figure"),
    Output("scatter", "figure"),
    Output('DebugText', 'children'),
    [Input("attOptions", "value"),
    Input('yearSlider', 'value'),
    Input('companySelection', 'value')])
def update_bar_chart(i, yearRange, selectedCompanies):
    
    debugTex = 'Blah'
    
    #Filter out data into specified year range
    isInYearRange = ((releaseYear >= yearRange[0]) * (releaseYear <= yearRange[1])).astype(bool)
    years = releaseYear[isInYearRange]
    att = scatterData[i]
    att = att[isInYearRange]
    names = modelNames[isInYearRange].astype(str)
    
    #Calculate linear fit based on filtered data
    linearFitX, linearFitY, detrendedAttributes = getLinearFit(years, att, useLogScale[i])
    
    numOfUniqueCompanies = len(np.unique(companyId))
    companyScores = []
    companyPhoneCount = []
    compNames = []

    for k, score in enumerate(detrendedAttributes):
        compName = companyNames[k]
        if compName in compNames:
            j = compNames.index(compName)
            companyScores[j] += score
            companyPhoneCount[j] += 1
        else:
            compNames.append(compName)
            companyScores.append(score)
            companyPhoneCount.append(1)

    companyScores = np.array(companyScores)
    companyPhoneCount = np.array(companyPhoneCount)
    compNames = np.array(compNames)
    companyFinalScore = companyScores# / companyPhoneCount
    
    sortedZip = sorted(zip(companyFinalScore, compNames), reverse=True)
    namesSorted = [name for _, name in sortedZip]
    scoreSorted = [score for score, _ in sortedZip]
    
    figBar = go.Figure()
    if (selectedCompanies == None) or (selectedCompanies == []):
        figBar.add_trace(go.Bar(x=namesSorted, y=scoreSorted))
    else:
        isSelected = np.zeros(len(namesSorted))
        for comp in selectedCompanies:
            name = usedNames[comp]
            isSelected[namesSorted.index(name)] = 1
        debugTex = isSelected
        figBar.add_trace(go.Bar(x=namesSorted, y=scoreSorted,
            marker=dict(color=isSelected, colorscale='Blugrn')))
    #figBar.update_layout(hovermode="x unified")
    figBar.update_layout(
        title="Company Scores",
        xaxis_title="Companies",
        yaxis_title="Sum of Residuals",
        transition_duration=500, 
        showlegend=False, 
        hovermode="x unified")
    figBar.update_xaxes(showticklabels=False)
    
    
    #Scatterplot behaviour
    figScat = go.Figure()
    if (selectedCompanies == None) or (selectedCompanies == []):
        figScat.add_trace(go.Scatter(
            x=years, 
            y=att, mode='markers', 
            hovertemplate=names, 
            name=scatterDatNames[i],
            marker=dict(size=8, color=detrendedAttributes, colorscale='Turbo')))
    else:
        boolArray = np.zeros(len(releaseYear))
        for comp in selectedCompanies:
            for c in dropData[comp]:
                boolArray[c] = 1
        boolArray = boolArray.astype(bool)
        boolArray = boolArray[isInYearRange]
        
        figScat.add_trace(go.Scatter(
            x=years[~boolArray], 
            y=att[~boolArray], 
            mode='markers',
            hoverinfo='skip',
            marker=dict(size=6, color='lightblue')))
            
        figScat.add_trace(go.Scatter(
            x=years[boolArray], 
            y=att[boolArray], 
            mode='markers', 
            hovertemplate=names[boolArray], 
            name=scatterDatNames[i],
            marker=dict(size=8, color=detrendedAttributes[boolArray], colorscale='Bluered')))
    
    figScat.add_trace(go.Scatter(x=linearFitX, y=linearFitY, name='Fit'))
    figScat.update_layout(
        title="Devices Beating the Trend",
        xaxis_title="Year",
        yaxis_title=scatterDatNames[i],
        transition_duration=500, 
        showlegend=False)
    
    #Use log scale if necessary
    if useLogScale[i]:
        figScat.update_yaxes(type="log")
    
    return figScat, figBar, debugTex

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)