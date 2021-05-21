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

#Given the scores of each model, we calculate the score of each company
def getCompanyScores(scores, companyNames):
    companyScores, compNames = [], []
    for k, score in enumerate(scores):
        compName = companyNames[k]
        if compName in compNames:
            j = compNames.index(compName)
            companyScores[j] += score
        else:
            compNames.append(compName)
            companyScores.append(score)
    
    #Convert from list to numpy arrays
    companyScores = np.array(companyScores)
    compNames = np.array(compNames)
    
    #Sort arrays based on score
    sortedZip = sorted(zip(companyScores, compNames), reverse=True)
    namesSorted = [name for _, name in sortedZip]
    scoreSorted = [score for score, _ in sortedZip]
    return namesSorted, scoreSorted

#==========================================================Get Data ================================================
#Read data file from excel spreadsheet
dataFileName = 'Mobile Device Data for Assignment 2.xlsx'
dataFrame = pd.read_excel(dataFileName)
dataArray = dataFrame.to_numpy().T
attributeArray = dataArray[4:]
attributeNames = dataFrame.columns.values[4:]
releaseYear = dataArray[2].astype(float)
modelNames = dataFrame['Model'].astype(str)


#Select attributes to be used in scatter plots
indexOfDataToUse = np.array([3, 4, 5, 6, 7, 8, 9, 10])
scatterDatNames = attributeNames[indexOfDataToUse]
scatterData = attributeArray[indexOfDataToUse]

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
    
    #Create dropdown for attribute selection
    dbc.Col(
        dbc.FormGroup([
            html.H1('Select Attribute'),
            dcc.Dropdown(
                id = 'attOptions',
                options = [{'label': attName, 'value': i} for i, attName in enumerate(scatterDatNames)],
                value = 0)]),
        width=6),
    
    #Create dropdown for company selection
    dbc.Col(
        dbc.FormGroup([
            html.H1('Select Company'),
            dcc.Dropdown(
                id = 'companySelection',
                options = dropDownOptions,
                multi=True
                )]),
        width=6)]),
    
    #Create year selection slider
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
        
        #Create radio buttons for selecting visualization methods
        html.H1('Select Visualization Method:'),
        dcc.RadioItems(options=[
            {'label': 'Earliest Adapters', 'value': 'discrete'},
            {'label': 'Beating the Trend', 'value': 'continuous'},
            {'label': 'Closest to Median', 'value': 'median'}],
            value='median')
    ])
])

#Add controls and plots to the main app layout
app.layout = html.Div([
    html.Div(controls),
    html.Div([
        dcc.Graph(id="bar-chart", figure={'layout': {"height": 700}}),
        dcc.Graph(id="scatter",  figure={'layout': {"height": 700}})]),
    html.Div(id='DebugText')
])

#Function for updating plots based on user input
#This function gets called by the dash app library and is not called anywhere in this python code file
@app.callback(
    Output("bar-chart", "figure"),
    Output("scatter", "figure"),
    Output('DebugText', 'children'),
    [Input("attOptions", "value"),
    Input('yearSlider', 'value'),
    Input('companySelection', 'value')])
def update_bar_chart(i, yearRange, selectedCompanies):
    debugTex = 'Blah'
    
    #Filter out data into specified year range and prepare data for plot
    isInYearRange = ((releaseYear >= yearRange[0]) * (releaseYear <= yearRange[1])).astype(bool)
    years = releaseYear[isInYearRange]
    att = scatterData[i]
    att = att[isInYearRange]
    names = modelNames[isInYearRange].astype(str)
    
    #Calculate scores based on distance to median
    median = np.median(att)
    distToMedian = np.abs(median - att)
    distToMedian = normalizeList(distToMedian).astype(float)
    scores = 2 / (1 + np.exp(4 * distToMedian))
    
    #Add main scatter data
    figScat = go.Figure()
    hoverStrings = np.array(['{}<br>Score: {:.2f}'.format(name, scores[i]) for i, name in enumerate(names)])
    if (selectedCompanies == None) or (selectedCompanies == []):
        figScat.add_trace(go.Scatter(
            x=years, 
            y=att, mode='markers', 
            hovertemplate=hoverStrings, 
            name=scatterDatNames[i],
            marker=dict(size=8, color=scores, colorscale='Turbo')))
    
    #If any company is selected, only highligh data points corresponding to selected companies
    else:
        #Get selected Companies
        selectedComps = np.zeros(len(releaseYear))
        for comp in selectedCompanies:
            for c in dropData[comp]:
                selectedComps[c] = 1
        selectedComps = selectedComps.astype(bool)
        selectedComps = selectedComps[isInYearRange]
        
        #Add the non-highlighted points to plot
        figScat.add_trace(go.Scatter(
            x=years[~selectedComps], 
            y=att[~selectedComps], 
            mode='markers',
            hoverinfo='skip',
            marker=dict(size=6, color='lightblue')))
        
        #Add highlighted points to plot
        figScat.add_trace(go.Scatter(
            x=years[selectedComps], 
            y=att[selectedComps], 
            mode='markers', 
            hovertemplate=hoverStrings[selectedComps], 
            name=scatterDatNames[i],
            marker=dict(size=8, color=scores[selectedComps], colorscale='Bluered')))
    
    #Add line representing the median
    figScat.add_trace(go.Scatter(
        x = [np.min(years), np.max(years)], 
        y = [median, median], 
        name = 'Median'))
    
    #Specify plot title and other plot attributes
    figScat.update_layout(
        title="Closest to Median",
        font_size = 30,
        xaxis_title="Year",
        yaxis_title=scatterDatNames[i],
        transition_duration=500, 
        showlegend=False)
    
    #Create company scores bar graph
    figBar = go.Figure()
    namesSorted, scoreSorted = getCompanyScores(scores, companyNames)
    if (selectedCompanies == None) or (selectedCompanies == []):
        figBar.add_trace(go.Bar(x=namesSorted, y=scoreSorted))
    
    #If any companies are selected, highlight the selected companies
    else:
        isSelected = np.zeros(len(namesSorted))
        for comp in selectedCompanies:
            name = usedNames[comp]
            isSelected[namesSorted.index(name)] = 1
        figBar.add_trace(go.Bar(
            x=namesSorted, 
            y=scoreSorted,
            marker=dict(
                color=isSelected, 
                colorscale='Blugrn')))
    
    #Add title and specify other properties of the bar graph
    figBar.update_layout(
        title="Company Scores",
        font_size = 30,
        xaxis_title="Companies",
        yaxis_title="Sum of Residuals",
        transition_duration=500, 
        showlegend=False, 
        hovermode="x unified")
    figBar.update_xaxes(showticklabels=False)
    
    #Return figures to dash app
    return figScat, figBar, debugTex

if __name__ == "__main__":
    app.run_server(debug=True, port=8053)