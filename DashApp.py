
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

#Read data file from excel spreadsheet
dataFileName = 'Mobile Device Data for Assignment 2.xlsx'
dataFrame = pd.read_excel(dataFileName)
dataArray = dataFrame.to_numpy().T
attributeArray = dataArray[4:]
attributeNames = dataFrame.columns.values[4:]
releaseYear = dataArray[2].astype(float)
modelNames = dataFrame['Model'].astype(str)
useLogScale = [True, True, True, False, False, False, False, False, False, False, False, False]

#Initiate a a dash app object
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#==================================================Code for Scatter Plot Page =========================================
#This header specifies which dashboard objects are considered as input and output for our next function
@app.callback(
    [Output('scatterPlot', 'figure'),
    Output('table','data')],
    [Input('attOptions', 'value'),
    Input('yearSlider', 'value')])
def update_figure(i, yearRange):

    #Filter out data into specified year range
    isInYearRange = ((releaseYear >= yearRange[0]) * (releaseYear <= yearRange[1])).astype(bool)
    years = releaseYear[isInYearRange]
    att = attributeArray[i]
    att = att[isInYearRange]
    names = modelNames[isInYearRange].astype(str)

    #Calculate linear fit based on filtered data
    linearFitX, linearFitY, detrendedAttributes = getLinearFit(years, att, useLogScale[i])

    #We define the main scatter plot to be shown in our dashboard
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=att, mode='markers',
        hovertemplate=names,
        name=attributeNames[i],
        marker=dict(size=8, color=detrendedAttributes, colorscale='Turbo')))

    #Define the fitted line to be included in our scatter plot
    fig.add_trace(go.Scatter(x=linearFitX, y=linearFitY, name='Fit'))

    #Use log scale if necessary
    if useLogScale[i]:
        fig.update_yaxes(type="log")

    #We identify the market leader based on the max value in detrended attribute, and then create table data
    namee = np.copy(names)
    marketLeaderIds = (-detrendedAttributes).argsort()
    marketLeaderName = namee[marketLeaderIds]
    marketLeaderData = [{'phoneName': name} for name in marketLeaderName[:10]]

    #Run transition animation and return values to dashboard objects
    fig.update_layout(transition_duration=500, showlegend=False)
    return fig, marketLeaderData

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

#Brings values to desired range (default being 0 and 1)
def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#Create a card for our controls
controls = dbc.Card([
    dbc.FormGroup([
        dbc.Label('Selected Attribute'),
        dcc.Dropdown(
            id = 'attOptions',
            options = [{'label': attName, 'value': i} for i, attName in enumerate(attributeNames)],
            value = 0),
        html.Label('Year:'),
        dcc.RangeSlider(
            id='yearSlider',
            min=releaseYear[0],
            max=releaseYear[-1], step=0.1,
            value=[releaseYear[0], releaseYear[-1]],
            marks={1991: '1991', 2012: '2012'}),
        dash_table.DataTable(id='table', columns=[{'name': 'Phone Name', 'id': 'phoneName'}])
    ])
])

#We define the layout for the app
scatterContent = dbc.Container([
    html.H1('Models Beating the Trend'),
    html.Hr(),
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dcc.Graph(id='scatterPlot'), md=8)
    ])
])

#========================================== Code for Parallel Coordinates Page ======================================

# @app.callback(
#     [Output('paracoorContent', 'figure')])

cols = ['RAM Capacity (Mb)', 'Storage (Mb)', 'CPU Clock (MHz)', 'Display Diagonal (in)', 'Volume (cubic cm)', 'Mass (grams)']

dataFrame['Release Year 01'] = pd.DatetimeIndex(dataFrame['Release Date']).year

pc_fig = px.parallel_coordinates(dataFrame, dimensions=cols, color='Release Year 01',      color_continuous_scale=px.colors.sequential.Rainbow)

paracoorContent = dbc.Container([
    html.H1('Features of Phones through Years'),
    html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=pc_fig), md=8)
    ])
])


#================================Main App and Sidebar  =========================================================
#Define the style of the sidebar used for navigation
sidebarStyle = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

#Define the page margins
pageMargins = {
    "margin-left": "2rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}

#Define the sidebar navigation layout
sidebar = html.Div([
    html.H2('Visualization Pages'),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink('Home', href='/', active='exact'),
        dbc.NavLink('Scatter Plots', href='/scatter', active='exact'),
        dbc.NavLink('Parallel Coordinate Plot', href='/paracoor', active='exact')
    ])
], style=sidebarStyle)

#Add sidebar to app (which also adds everything else to app)
content = html.Div(id='pageContent', style=pageMargins)
app.layout = html.Div([dcc.Location(id='url'), sidebar, content])

#Define the functionality of the sidebar
@app.callback(
Output('pageContent', 'children'),
[Input('url', 'pathname')])
def renderPageContent(pathname):
    if pathname == '/':
        return dbc.Container([html.P('This is the home page')])
    elif pathname == '/scatter':
        return scatterContent
    elif pathname == '/paracoor':
        return paracoorContent

    #Return error if pathname does not exist
    return dbc.Jumbotron([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {pathname} was not recognised...")
    ])

if __name__ == "__main__":
    app.run_server()
