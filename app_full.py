from flask import Flask, request, render_template, Markup
import joblib
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from flask import Markup
from scipy.stats import linregress
import plotly.graph_objs as go


app = Flask(__name__)
models = {
    'model1': joblib.load('linear_regressor_model.pkl'),
    'model2': joblib.load('rf_regressor_model.pkl'),
    'model3': joblib.load('xgb_regressor_model.pkl')
}

@app.route('/', methods=['GET'])
def home():
    ## Michael
    ## EDA 1
    ## Data Manipulation for EDA 1
    data = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')
    
    ## Remove some unused columns
    columns_to_drop = ['StratificationCategory2', 'Stratification2', 'StratificationCategory3', 'Stratification3', 'Geolocation', 'YearEnd', 'Response',
                       'DataValueFootnoteSymbol', 'LowConfidenceLimit', 'HighConfidenceLimit', 'LocationID', 'TopicID', 'QuestionID', 'ResponseID',
                       'DataValueTypeID', 'StratificationCategoryID1', 'StratificationID1', 'StratificationCategoryID2', 'StratificationID2',
                       'StratificationCategoryID3', 'StratificationID3']
    data =  data.drop(columns=columns_to_drop)

    data = data.rename(columns={
        'LocationDesc': 'State',
        'LocationAbbr': 'StateAbbr',
        'YearStart': 'Year',
        'DataSource': 'Source',
    })
    
    data_obesity = data[data['Question'].str.contains('Obesity among adults')]
    data_obesity = data_obesity[data_obesity['DataValueType'] == 'Crude Prevalence']
    data_obesity = data_obesity.groupby(['State', 'Year', 'StateAbbr'])['DataValue'].mean()
    data_obesity = data_obesity.reset_index()

    print(data_obesity.head())

    data_obesity = data_obesity.rename(columns={
        'DataValue': 'Obesity Rate',
    })

    print(data_obesity.head())

    state_abbr = dict(zip(data_obesity['State'], data_obesity['StateAbbr']))

    print(state_abbr)

    obesity_2019 = data_obesity[data_obesity['Year'] == 2019]
    obesity_2020 = data_obesity[data_obesity['Year'] == 2020]
    obesity_2021 = data_obesity[data_obesity['Year'] == 2021]
    obesity_2022 = data_obesity[data_obesity['Year'] == 2022]
    
    # Create traces for each DataFrame
    trace_2019 = go.Choropleth(
        locations=obesity_2019['StateAbbr'],
        z=obesity_2019['Obesity Rate'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar=dict(title='Obesity Rate (%)'),
        visible=True,
        name='2019',
        geo='geo'
    )

    trace_2020 = go.Choropleth(
        locations=obesity_2020['StateAbbr'],
        z=obesity_2020['Obesity Rate'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar=dict(title='Obesity Rate (%)'),
        visible=False,
        name='2020',
        geo='geo'
    )

    trace_2021 = go.Choropleth(
        locations=obesity_2021['StateAbbr'],
        z=obesity_2021['Obesity Rate'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar=dict(title='Obesity Rate (%)'),
        visible=False,
        name='2021',
        geo='geo'
    )

    trace_2022 = go.Choropleth(
        locations=obesity_2022['StateAbbr'],
        z=obesity_2022['Obesity Rate'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar=dict(title='Obesity Rate (%)'),
        visible=False,
        name='2022',
        geo='geo'
    )

    # Create the data list with all traces
    data_1 = [trace_2019, trace_2020, trace_2021, trace_2022]

    # Create the layout with a dropdown menu for toggling between years
    layout = go.Layout(
        title='Obesity Rates by State',
        geo=dict(
            scope='usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            projection=dict(type='albers usa'),
            showcoastlines=True,
            coastlinewidth=0.5,
            coastlinecolor='rgb(0, 0, 0)'
        ),
        updatemenus=[
            dict(
                buttons=[
                    {'label': '2019', 'method': 'update', 'args': [{'visible': [True, False, False, False]}]},
                    {'label': '2020', 'method': 'update', 'args': [{'visible': [False, True, False, False]}]},
                    {'label': '2021', 'method': 'update', 'args': [{'visible': [False, False, True, False]}]},
                    {'label': '2022', 'method': 'update', 'args': [{'visible': [False, False, False, True]}]}
                ],
                direction='down',
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.1,
                yanchor='top'
            ),
        ]
    )

    # Create the figure object
    fig1 = go.Figure(data=data_1, layout=layout)

    
    # EDA 2
    ## Data Manipulation for EDA 2
    data_oral_health = data[data['Question'].str.contains('Visited dentist or dental clinic in the past year among adults')]
    data_teeth = data[data['Question'].str.contains('Six or more teeth lost among adults aged 65 years and older')]


    data_oral_health = data_oral_health[data_oral_health['DataValueType'] == 'Crude Prevalence']
    data_oral_health = data_oral_health.groupby(['State', 'Year', 'StateAbbr'])['DataValue'].mean()
    data_oral_health = data_oral_health.reset_index()
    data_teeth = data_teeth[data_teeth['DataValueType'] == 'Crude Prevalence']
    data_teeth = data_teeth.groupby(['State', 'Year', 'StateAbbr'])['DataValue'].mean()
    data_teeth = data_teeth.reset_index()


    data_oral_health = data_oral_health.rename(columns={
        'DataValue': 'Dentist Visit Rate',
    })

    data_teeth = data_teeth.rename(columns={
        'DataValue': 'lost 6 or more teeth by age 65',
    })

    ##merge data_ordal_health and data_obesity then merge teeth to it
    data_obesity_oral = pd.merge(data_oral_health, data_obesity, on=['State', 'Year', 'StateAbbr'], how='outer')
    data_obesity_teeth = pd.merge(data_obesity, data_teeth, on=['State', 'Year', 'StateAbbr'], how='outer')

    ##drop NaN values
    data_obesity_oral = data_obesity_oral.dropna()
    data_obesity_teeth = data_obesity_teeth.dropna()

    ## select only data from 2020
    data_obesity_oral = data_obesity_oral[data_obesity_oral['Year'] == 2020]
    data_obesity_teeth = data_obesity_teeth[data_obesity_teeth['Year'] == 2020]
    
    # Calculate linear regression for Dentist Visit Rate (Oral Health)
    slope_dentist_oral, intercept_dentist_oral, _, _, _ = linregress(data_obesity_oral['Dentist Visit Rate'], data_obesity_oral['Obesity Rate'])

    # Calculate linear regression for Teeth Lost Rate (Teeth Health)
    slope_teeth_teeth, intercept_teeth_teeth, _, _, _ = linregress(data_obesity_teeth['lost 6 or more teeth by age 65'], data_obesity_teeth['Obesity Rate'])

    # Create scatter plot with trend line for Dentist Visit Rate (Oral Health)
    fig2_1 = px.scatter(data_obesity_oral, x='Dentist Visit Rate', y='Obesity Rate', title='Obesity Rate vs. Oral Health by State (2020)')
    fig2_1.add_trace(go.Scatter(x=data_obesity_oral['Dentist Visit Rate'], y=slope_dentist_oral * data_obesity_oral['Dentist Visit Rate'] + intercept_dentist_oral, mode='lines', name='Negative Trend'))

    # Create scatter plot with trend line for Teeth Lost Rate (Teeth Health)
    fig2_2 = px.scatter(data_obesity_teeth, x='lost 6 or more teeth by age 65', y='Obesity Rate', title='Obesity Rate vs. Teeth Health by State (2020)')
    fig2_2.add_trace(go.Scatter(x=data_obesity_teeth['lost 6 or more teeth by age 65'], y=slope_teeth_teeth * data_obesity_teeth['lost 6 or more teeth by age 65'] + intercept_teeth_teeth, mode='lines', name='Positive Trend'))

    fig2_1.update_layout(title={'text': 'Obesity Relation to Dental Visits', 'x': 0.5})
    fig2_2.update_layout(title={'text': 'Obesity Relation to Teeth Lost', 'x': 0.5}, xaxis_title='Lost 6 or More Teeth by Age 65')

    
        # EDA 3
    ## Set up a function to get the data you want
    def process_question_data(data_frame, column_name):
        # Filter by 'DataValueType' and calculate mean by grouping
        data_frame = data_frame[data_frame['DataValueType'] == 'Crude Prevalence']  ##use this since less missingness
        data_frame = data_frame.groupby(['Year', 'State'])['DataValue'].mean()
        data_frame = data_frame.reset_index()

        # Rename columns
        data_frame = data_frame.rename(columns={'DataValue': column_name})

        return data_frame

    ## Pick the variables to test
    data_obesity = data[data['Question'].str.contains('Obesity among adults')]
    data_aerobic = data[data['Question'].str.contains('Met aerobic physical activity guideline for substantial health benefits, adults')]
    data_disability = data[data['Question'].str.contains('Adults with any disability')]
    data_health_status = data[data['Question'].str.contains('Fair or poor self-rated health status among adults')]
    data_veggies = data[data['Question'].str.contains('Consumed vegetables less than one time daily among adults')]
    data_fruit = data[data['Question'].str.contains('Consumed fruit less than one time daily among adults')]
    data_alcohol_percap = data[data['Question'].str.contains('Per capita alcohol consumption among people aged 14 years and older')]
    data_smoking = data[data['Question'].str.contains('Current cigarette smoking among adults')]
    data_sleep = data[data['Question'].str.contains('Short sleep duration among adults')]
    data_depression = data[data['Question'].str.contains('Depression among adults')]
    data_diabetes = data[data['Question'].str.contains('Diabetes among adults')]
    data_poverty = data[data['Question'].str.contains('Living below 150% of the poverty threshold among all people')]

    ## Run the functino to reformat
    data_obesity = process_question_data(data_obesity, 'Obesity Rate')
    data_aerobic = process_question_data(data_aerobic, 'met aerobic fitness level')
    data_disability = process_question_data(data_disability, 'disability rate')
    data_health_status = process_question_data(data_health_status, 'fair or poor health rate')
    data_veggies = process_question_data(data_veggies, 'veggie consumption rate')
    data_fruit = process_question_data(data_fruit, 'fruit consumption rate')
    data_alcohol_percap = process_question_data(data_alcohol_percap, 'per capita alcohol consumption')
    data_smoking = process_question_data(data_smoking, 'smoking rate')
    data_sleep = process_question_data(data_sleep, 'short sleep duration rate')
    data_depression = process_question_data(data_depression, 'depression rate')
    data_diabetes = process_question_data(data_diabetes, 'diabetes rate')
    data_poverty = process_question_data(data_poverty, 'poverty rate')

    # merge the dataframes together
    data_frames = [data_obesity, data_aerobic, data_disability, data_depression, data_diabetes, data_health_status, data_veggies, 
                    data_fruit, data_alcohol_percap, data_smoking, data_sleep, data_poverty]
    merged_data = data_frames[0]
    for df in data_frames[1:]:
        merged_data = pd.merge(merged_data, df, on=['Year', 'State'], how='outer')


    ##make 4 data frames, one forer 2019, one for 2020, 2021, 2022
    data_2019 = merged_data[merged_data['Year'] == 2019]
    data_2020 = merged_data[merged_data['Year'] == 2020]
    data_2021 = merged_data[merged_data['Year'] == 2021]
    data_2022 = merged_data[merged_data['Year'] == 2022]

    ##drop columns that have over 80% nan
    data_2019 = data_2019.dropna(thresh=0.8*len(data_2019), axis=1)
    data_2020 = data_2020.dropna(thresh=0.8*len(data_2020), axis=1)
    data_2021 = data_2021.dropna(thresh=0.8*len(data_2021), axis=1)
    data_2022 = data_2022.dropna(thresh=0.8*len(data_2022), axis=1)

    ## drop year column
    data_2019 = data_2019.drop(columns=['Year'])
    data_2020 = data_2020.drop(columns=['Year'])
    data_2021 = data_2021.drop(columns=['Year'])
    data_2022 = data_2022.drop(columns=['Year'])

    ## drop nan
    data_2019 = data_2019.dropna()
    data_2020 = data_2020.dropna()
    data_2021 = data_2021.dropna()
    data_2022 = data_2022.dropna()

    data_2019 = data_2019.drop(columns=['State'])
    data_2020 = data_2020.drop(columns=['State'])
    data_2021 = data_2021.drop(columns=['State'])
    data_2022 = data_2022.drop(columns=['State'])

    ## create correlation matrixes
    correlation_matrix_2019 = data_2019.corr()
    correlation_matrix_2020 = data_2020.corr()
    correlation_matrix_2021 = data_2021.corr()
    correlation_matrix_2022 = data_2022.corr()

    fig3 = go.Figure()

    correlation_matrices = {
        '2019': correlation_matrix_2019,
        '2020': correlation_matrix_2020,
        '2021': correlation_matrix_2021,
        '2022': correlation_matrix_2022
    }


    def create_heatmap(year):
        matrix = correlation_matrices[year]
        return go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Inferno',
            showscale=True
        )

    initial_year = '2019'
    fig3 = go.Figure(data=create_heatmap(initial_year))

    fig3.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=[
                    dict(
                        args=[
                            {'z': [correlation_matrices[year].values],
                                'x': [correlation_matrices[year].columns],
                                'y': [correlation_matrices[year].index]},
                            {'title': f'Correlation Heatmap ({year})'}
                        ],
                        label=year,
                        method="update"
                    ) for year in correlation_matrices.keys()
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ],
        title_text=f'Correlation Heatmap ({initial_year})',
        title_x=0.5,
    )

    
    # EDA 4
    ## Data Manipulation for EDA 4
    # data_mental = data[data['Question'].str.contains('Depression among adults')]
    # data_mental = data_mental[data_mental['DataValueType'] == 'Crude Prevalence']  ##due to lack of availiabel data

    # #Frequent mental distress among adults
    # data_mental_distress = data[data['Question'].str.contains('Frequent mental distress among adults')]
    # data_mental_distress = data_mental_distress[data_mental_distress['DataValueType'] == 'Crude Prevalence']


    # ## grouping by location, year, and LocationAbbr -> create 2 columns, 1 for each question
    # group_cols = ['State', 'Year', 'StateAbbr']
    # value_col = 'DataValue'

    # data_mental = data_mental.pivot_table(index=group_cols, columns='Question', values=value_col, aggfunc='mean').reset_index()

    # data_mental = data_mental.rename(columns={
    #     'DataValue': 'Depression Rate',
    # })

    # data_mental_distress = data_mental_distress.pivot_table(index=group_cols, columns='Question', values=value_col, aggfunc='mean').reset_index()

    # data_mental_distress = data_mental_distress.rename(columns={
    #     'DataValue': 'Frequent Mental Distress Rate',
    # })

    # ## merge the data frames
    # data_eda4 = pd.merge(data_mental, data_obesity, on=['State', 'Year', 'StateAbbr'], how='outer')
    # data_eda4 = pd.merge(data_eda4, data_mental_distress, on=['State', 'Year', 'StateAbbr'], how='outer')

    # ## average the data by year
    # data_eda4 = data_eda4.groupby(['Year']).mean().reset_index()
    
    # # Filter the data for all variables
    # selected_variables = ['Depression among adults', 'Frequent mental distress among adults', 'Obesity Rate']
    # data_filtered = data_eda4[['Year'] + selected_variables]

    # # Reshape the data to long format for Plotly Express
    # data_long = pd.melt(data_filtered, id_vars=['Year'], value_vars=selected_variables,
    #                     var_name='Variable', value_name='Rate')

    # # Create the bar chart using Plotly Express with side-by-side bars
    # fig4 = px.bar(data_long, x='Year', y='Rate', color='Variable',
    #              title='Obesity Rate and Mental Health Indicators by Year',
    #              labels={'Rate': 'Rate', 'Variable': 'Variable', 'Year': 'Year'},
    #              barmode='group')  # Use barmode='group' for side-by-side bars
    # fig4.update_layout(title={'text': 'Obesity Rate and Mental Health Indicators by Year', 'x': 0.5})
    
    # Convert the plot to a DIV element (HTML string)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    plot_div2_1 = plot(fig2_1, output_type='div', include_plotlyjs=False)
    plot_div2_2 = plot(fig2_2, output_type='div', include_plotlyjs=False)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)
    # plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)
    
    
    ## Kaixian
    disease = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")
    df = disease[disease.TopicID == "NPAW"]
    
    # EDA 1
    q1 = df[df.Question == 'Consumed fruit less than one time daily among adults']
    q1 = q1[q1['DataValueType'] == 'Crude Prevalence']
    q1['DataValue'].fillna(q1['DataValue'].mean(), inplace=True)
    q1 = q1.groupby(['LocationAbbr', 'YearStart'])['DataValue'].mean().reset_index()

    q1_2019 = q1[q1.YearStart == 2019]
    q1_2021 = q1[q1.YearStart == 2021]

    trace_2019 = go.Choropleth(
        locations=q1_2019['LocationAbbr'],
        z=q1_2019['DataValue'],
        locationmode='USA-states',
        colorscale='Viridis',
        name='2019'
    )

    trace_2021 = go.Choropleth(
        locations=q1_2021['LocationAbbr'],
        z=q1_2021['DataValue'],
        locationmode='USA-states',
        colorscale='Viridis',
        name='2021',
        visible=False
    )


    fig5 = go.Figure()

    fig5.add_trace(trace_2019)
    fig5.add_trace(trace_2021)

    buttons = [
        dict(
            label='2019',
            method='update',
            args=[{'visible': [True, False]}]
        ),
        dict(
            label='2021',
            method='update',
            args=[{'visible': [False, True]}]
        )
    ]

    fig5.update_layout(
        title_text='Consumed fruit less than one time daily among adults by state',
        geo_scope='usa',
        updatemenus=[
            dict(
                x=0.5,
                xanchor='center',
                yanchor='top',
                y=1.1,
                buttons=buttons
            )
        ]
    )
  
    # EDA 2
    data_activity = df[df.Question == 'No leisure-time physical activity among adults']
    data_activity = data_activity[data_activity['DataValueType'] == 'Crude Prevalence']
    data_activity['DataValue'].fillna(data_activity['DataValue'].mean(), inplace=True)
    data_activity = data_activity.groupby(['LocationAbbr', 'YearStart'])['DataValue'].mean().reset_index()

    combined_df = pd.merge(q1, data_activity, on=['LocationAbbr', 'YearStart'], suffixes=('_consumption', '_activity'))
    combined_df.rename(columns={'DataValue_consumption': 'consumption', 'DataValue_activity': 'activity'}, inplace=True)

    fig6 = go.Figure()

    filtered_df = combined_df[combined_df['YearStart'].isin([2019, 2021])]

    for year in filtered_df['YearStart'].unique():
        year_data = filtered_df[filtered_df['YearStart'] == year]
        fig6.add_trace(go.Scatter(
            x=year_data['consumption'],
            y=year_data['activity'],
            mode='markers',
            name=str(year),
            text=year_data['LocationAbbr']
        ))

    fig6.update_layout(
        title='Relationship Between Fruit Consumption Less Than Once/Day and No Leisure-time Physical Activity',
        xaxis_title='Fruit Consumption Less Than One Time Daily (%)',
        yaxis_title='No Leisure-time Physical Activity (%)',
        legend_title_text='Year'
    )

    
    # EDA 3
    q2 = df[df.Question == 'Consumed vegetables less than one time daily among adults']
    q2 = q2[q2['DataValueType'] == 'Crude Prevalence']
    q2['DataValue'].fillna(q2['DataValue'].mean(), inplace=True)
    q2 = q2.groupby(['LocationAbbr', 'YearStart'])['DataValue'].mean().reset_index()

    q2_2019 = q2[q2['YearStart'] == 2019]
    q2_2021 = q2[q2['YearStart'] == 2021]
    combined = pd.merge(q2_2019, q2_2021, on='LocationAbbr', suffixes=('_2019', '_2021'))
    combined['Diff'] = ((combined['DataValue_2021'] / combined['DataValue_2019']) - 1)

    fig7 = px.bar(combined, x='LocationAbbr', y='Diff')
    fig7.update_layout(
        title='Difference in Average Vegetable Consumption Less Than One Time Daily Among Adults by State and Year',
        xaxis_title="State",
        yaxis_title="Percentage Change",
        xaxis={'tickangle': 45},
        yaxis_tickformat='%',
        showlegend=False
    )
        
    # EDA 4
    data_obesity = df[df.Question == 'Obesity among adults']
    data_obesity = data_obesity[data_obesity['DataValueType'] == 'Crude Prevalence']
    data_obesity['DataValue'].fillna(data_obesity['DataValue'].mean(), inplace=True)
    data_obesity = data_obesity.groupby(['LocationAbbr', 'YearStart'])['DataValue'].mean().reset_index()

    combined_df = pd.merge(q2, data_obesity, on=['LocationAbbr', 'YearStart'], suffixes=('_consumption', '_obesity'))
    combined_df.rename(columns={'DataValue_consumption': 'consumption', 'DataValue_obesity': 'obesity'}, inplace=True)

    fig_2019 = go.Scatter(
        x=combined_df[combined_df['YearStart'] == 2019]['consumption'],
        y=combined_df[combined_df['YearStart'] == 2019]['obesity'],
        mode='markers',
        hoverinfo='text',
        text=combined_df[combined_df['YearStart'] == 2019]['LocationAbbr'],
        name='2019'
    )

    fig_2021 = go.Scatter(
        x=combined_df[combined_df['YearStart'] == 2021]['consumption'],
        y=combined_df[combined_df['YearStart'] == 2021]['obesity'],
        mode='markers',
        hoverinfo='text',
        text=combined_df[combined_df['YearStart'] == 2021]['LocationAbbr'],
        name='2021',
        visible=False  # Start with this trace invisible
    )

    fig8 = go.Figure()

    fig8.add_trace(fig_2019)
    fig8.add_trace(fig_2021)

    buttons = [
        dict(
            label='2019',
            method='update',
            args=[{'visible': [True, False]},
                  {'title': "Relationship Between Vegetable Consumption and Obesity (2019)"}]
        ),
        dict(
            label='2021',
            method='update',
            args=[{'visible': [False, True]},
                  {'title': "Relationship Between Vegetable Consumption and Obesity (2021)"}]
        )
    ]

    fig8.update_layout(
        title="Relationship Between Vegetable Consumption and Obesity",
        xaxis_title="Vegetable Consumption Less Than One Time Daily (%)",
        yaxis_title="Obesity Rate (%)",
        updatemenus=[
            dict(
                active=0,
                x=0.5,
                xanchor='center',
                yanchor='top',
                y=1.2,
                buttons=buttons
            )
        ]
    )

    
    # Convert the plot to a DIV element (HTML string)
    plot_div5 = plot(fig5, output_type='div', include_plotlyjs=False)
    plot_div6 = plot(fig6, output_type='div', include_plotlyjs=False)
    plot_div7 = plot(fig7, output_type='div', include_plotlyjs=False)
    plot_div8 = plot(fig8, output_type='div', include_plotlyjs=False)


    ## Jianxiong
    # EDA 1
    original_df = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')
    df = original_df.loc[original_df['Topic'] == 'Nutrition, Physical Activity, and Weight Status', :]
    df = df.dropna(axis=1, how='all')

    fig9 = px.choropleth(
    df,
    locations='LocationAbbr',  
    locationmode='USA-states',  
    color='DataValue',  
    color_continuous_scale=px.colors.sequential.Viridis,  # color scale
    scope="usa",  
    labels={'DataValue': 'Percentage Meeting Guidelines'},
    title='Percentage of Children Meeting Aerobic Guidelines by State'
    )

    fig9.update_layout(
    geo=dict(
        lakecolor='white'  # changes the lake color to white
    ),
    margin={"r":0,"t":0,"l":0,"b":0}  
    )
    
    # EDA 2
    # Filter to the specific question of interest
    filtered_df = df[df["Question"] == "Children and adolescents aged 6-13 years meeting aerobic physical activity guideline"]

    fig10 = px.bar(filtered_df, x='LocationAbbr', y='DataValue', title='Percentage of Children Meeting Aerobic Guidelines by State', labels={'LocationAbbr': 'State', 'DataValue': 'Percentage'})

    # EDA 3
    # Histogram of Data Values
    fig11 = px.histogram(filtered_df, x='DataValue', nbins=10, title='Distribution of Percentages of Children Meeting Aerobic Guidelines')
    fig11.update_xaxes(title_text='Percentage')
    fig11.update_yaxes(title_text='Frequency')
    
    # EDA 4
    # Boxplot by Race Group (ensure race_cat is properly filtered)
    race_cat = filtered_df[filtered_df['StratificationCategory1'] == 'Race/Ethnicity']
    fig12 = px.box(race_cat, x='Stratification1', y='DataValue', title='Variation in Meeting Aerobic Guidelines by Race Group', labels={'Stratification1': 'Race Group', 'DataValue': 'Percentage'})
    
    # Convert the plot to a DIV element (HTML string)
    plot_div9 = plot(fig9, output_type='div', include_plotlyjs=False)
    plot_div10 = plot(fig10, output_type='div', include_plotlyjs=False)
    plot_div11 = plot(fig11, output_type='div', include_plotlyjs=False)
    plot_div12 = plot(fig12, output_type='div', include_plotlyjs=False)
    
    
    ## Mengxue
    df = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")
    npaw = df[df['TopicID'] == 'NPAW']
    
    # EDA 1
    # adults no leisure time 
    adult_no_leisure_time = npaw[npaw['QuestionID'] == 'NPW06']
    adult_no_leisure_eda = adult_no_leisure_time[['YearStart',
                               'LocationAbbr', 'LocationDesc', 'DataValueType', 'DataValue', 'LowConfidenceLimit',
                               'HighConfidenceLimit', 'StratificationCategory1', 'Stratification1']]
    adult_no_leisure_eda = adult_no_leisure_eda.dropna()
    
    # overall
    adult_no_leisure_overall = adult_no_leisure_eda[adult_no_leisure_eda['StratificationCategory1'] == 'Overall']
    
    # Find the Overall percentage of US adults who had no physical activity within the previous month from 2019-2022
    # Filter the DataFrame to have only age-adjusted and crude data values
    age_adjusted_data = adult_no_leisure_overall[adult_no_leisure_overall['DataValueType'] == 'Age-adjusted Prevalence']
    crude_data = adult_no_leisure_overall[adult_no_leisure_overall['DataValueType'] == 'Crude Prevalence']

    # Create the figure for the initial view (age-adjusted data)
    fig13 = px.choropleth(age_adjusted_data, 
                        locations="LocationAbbr", 
                        locationmode="USA-states", 
                        color="DataValue", 
                        animation_frame="YearStart",
                        scope="usa",
                        color_continuous_scale='Viridis',
                        title="Overall percentage of US adults who had no physical activity within<br>the previous month (2019-2022)")

    # Update color axis title
    fig13.update_coloraxes(colorbar_title="State-wise Percentage")

    # Update layout
    fig13.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)'))  # Set background and lake colors to transparent

    # Create dropdown menu
    dropdown_menu = [{'label': 'Age-Adjusted Prevalence', 'method': 'update',
                      'args': [{'z': [age_adjusted_data['DataValue']], 'colorbar.title': 'Age-Adjusted Percentage'}]},
                     {'label': 'Crude Prevalence', 'method': 'update',
                      'args': [{'z': [crude_data['DataValue']], 'colorbar.title': 'Crude Percentage'}]}]

    # Add dropdown menu to the figure
    fig13.update_layout(updatemenus=[{'buttons': dropdown_menu,
                                    'direction': 'down',
                                    'showactive': True, 
                                    'x': 0.00,
                                    'xanchor': 'left',
                                    'y': 1.15,
                                    'yanchor': 'top'}])
    
    # EDA 2
    # Find the percentage of adults having no leisure time in the past month in year 2022 stratified by sex
    adult_no_leisure_2022 = adult_no_leisure_eda[adult_no_leisure_eda['YearStart'] == 2022]  
    no_leisure_sex = adult_no_leisure_2022[adult_no_leisure_2022['StratificationCategory1'] == 'Sex']
    
    # show the same thing with line plot
    # Use this as plot2 as this is clearer
    age_adjusted_sex_2022 = no_leisure_sex[no_leisure_sex['DataValueType'] == 'Age-adjusted Prevalence']
    crude_sex_2022 = no_leisure_sex[no_leisure_sex['DataValueType'] == 'Crude Prevalence']

    # Sort the DataFrames by 'LocationAbbr' to avoid crossing lines
    age_adjusted_sex_2022_sorted = age_adjusted_sex_2022.sort_values(by='LocationAbbr')
    crude_sex_2022_sorted = crude_sex_2022.sort_values(by='LocationAbbr')

    # print(age_adjusted_sex_2022[age_adjusted_sex_2022['LocationAbbr'] == 'VA'])
    # print(age_adjusted_sex_2022_sorted[age_adjusted_sex_2022_sorted['LocationAbbr'] == 'VA'])

    # Create the line plot for the initial view (age-adjusted data)
    fig14 = px.line(age_adjusted_sex_2022_sorted, x='LocationAbbr', y='DataValue', color='Stratification1', facet_col='YearStart',
                    category_orders={'LocationAbbr': sorted(age_adjusted_sex_2022_sorted['LocationAbbr'].unique())},
                    facet_col_wrap=1, labels={'DataValue': 'Percentage', 'Stratification1': 'Gender'})

    fig14.update_layout(title='''
    Percentage of adults who had no physical activity within<br>the previous month in 2022 stratified by gender''',
                        xaxis_title='Location',
                        margin=dict(t=90))  # Adjust top margin to make room for the title)

    # Create dropdown menu
    dropdown_menu14 = [{'label': 'Age-Adjusted Prevalence', 'method': 'update',
                        'args': [{'y': [age_adjusted_sex_2022_sorted[age_adjusted_sex_2022_sorted['Stratification1'] == strat]['DataValue'] for strat in age_adjusted_sex_2022_sorted['Stratification1'].unique()],
                                  'colorbar.title': 'Age-Adjusted Percentage'}]},
                       {'label': 'Crude Prevalence', 'method': 'update',
                        'args': [{'y': [crude_sex_2022_sorted[crude_sex_2022_sorted['Stratification1'] == strat]['DataValue'] for strat in crude_sex_2022_sorted['Stratification1'].unique()],
                                  'colorbar.title': 'Crude Percentage'}]}]

    # Add dropdown menu to the figure
    fig14.update_layout(updatemenus=[{'buttons': dropdown_menu14,
                                      'direction': 'down',
                                      'showactive': True,  
                                      'x': 1.36,
                                      'xanchor': 'right',
                                      'y': 0.4,
                                      'yanchor': 'top'}])
    
    # EDA 3
    # Find the percentage of adults having no leisure time in the past month in year 2022 stratified by race/ethnicity
    no_leisure_race = adult_no_leisure_2022[adult_no_leisure_2022['StratificationCategory1'] == 'Race/Ethnicity']
    age_adjusted_race_2022 = no_leisure_race[no_leisure_race['DataValueType'] == 'Age-adjusted Prevalence']
    crude_race_2022 = no_leisure_race[no_leisure_race['DataValueType'] == 'Crude Prevalence']

    fig15 = px.bar(age_adjusted_race_2022, x='LocationAbbr', y='DataValue', color='Stratification1', facet_col='YearStart', 
                 category_orders={'LocationAbbr': sorted(age_adjusted_race_2022['LocationAbbr'].unique())},
                 facet_col_wrap=1, labels={'DataValue': 'Percentage', 'Stratification1': 'Race/Ethnicity'})

    fig15.update_layout(title='''
    Percentage of US adults who had no physical activity<br>within the previous month in 2022
    stratified by race/ethnicity''',
                       xaxis_title='Location',
                      margin=dict(t=90))  # Adjust top margin to make room for the title)
    # Create dropdown menu
    dropdown_menu15 = [{'label': 'Age-Adjusted Prevalence', 'method': 'restyle',
                      'args': [{'y': [age_adjusted_race_2022['DataValue']], 'colorbar.title': 'Age-Adjusted Percentage'}]},
                     {'label': 'Crude Prevalence', 'method': 'restyle',
                      'args': [{'y': [crude_race_2022['DataValue']], 'colorbar.title': 'Crude Percentage'}]}]

    # Add dropdown menu to the figure
    fig15.update_layout(updatemenus=[{'buttons': dropdown_menu15,
                                    'direction': 'down',
                                    'showactive': True, 
                                    'x': 1.5,
                                    'xanchor': 'right',
                                    'y': 0.15,
                                    'yanchor': 'top'}])
    
    # EDA 4
    # For highschool student, find the overall percentage of soda consumption at least once daily in all locations in 2019 and 2021.Then, find the locations which reported higher overall percentage in 2021 vs 2019. For these locations, find the percentage of soda consumption at least once daily by grade.
    npaw_soda = npaw[npaw['QuestionID'] == 'NPW05']
    npaw_soda_eda = npaw_soda[['YearStart',
                           'LocationAbbr', 'LocationDesc', 'DataValue', 'LowConfidenceLimit',
                           'HighConfidenceLimit', 'StratificationCategory1', 'Stratification1']]
    # overall
    npaw_soda_overall = npaw_soda_eda[npaw_soda_eda['StratificationCategory1'] == 'Overall']
    npaw_soda_overall = npaw_soda_overall.dropna()
    
    # choose the states with both records in 2019 and 2021
    npaw_soda_overall_over_time = npaw_soda_overall.groupby('LocationDesc').filter(
        lambda x: (2019 in x['YearStart'].values) and (2021 in x['YearStart'].values))
    
    # Find the locations reporting higher overall percentage 2021 vs 2019
    soda_higher_over_time = npaw_soda_overall_over_time.groupby("LocationDesc").filter(
        lambda x: (x[x['YearStart'] == 2021]['DataValue'].mean() > x[x['YearStart'] == 2019]['DataValue'].mean()))

    soda_lower_over_time = npaw_soda_overall_over_time.groupby("LocationDesc").filter(
        lambda x: (x[x['YearStart'] == 2021]['DataValue'].mean() < x[x['YearStart'] == 2019]['DataValue'].mean()))
    print(len(soda_higher_over_time))  # 15 locations
    print(len(soda_lower_over_time))  # 22 locations
    
    # Among the locations which report higher overall percentages 2021 vs 2019, look at difference by grade
    npaw_soda_grade = npaw_soda_eda[npaw_soda_eda['StratificationCategory1'] == 'Grade']
    npaw_soda_increase_grade = npaw_soda_grade[npaw_soda_grade['LocationAbbr'].isin(soda_higher_over_time['LocationAbbr'].unique())]
    npaw_soda_increase_grade = npaw_soda_increase_grade.dropna()
    
    # Define a mapping from grade to numeric value
    grade_order = {'Grade 9': 9, 'Grade 10': 10, 'Grade 11': 11, 'Grade 12': 12}

    # Map the grades to numeric values
    npaw_soda_increase_grade['GradeNumeric'] = npaw_soda_increase_grade['Stratification1'].map(grade_order)

    # Split the dataset into two separate DataFrames for years 2021 and 2019
    soda_increase_2021 = npaw_soda_increase_grade[npaw_soda_increase_grade['YearStart'] == 2021]
    soda_increase_2019 = npaw_soda_increase_grade[npaw_soda_increase_grade['YearStart'] == 2019]

    # Sort the data on location and numeric grade for both datasets
    soda_increase_2021_sorted = soda_increase_2021.sort_values(by=['LocationAbbr', 'GradeNumeric'])
    soda_increase_2019_sorted = soda_increase_2019.sort_values(by=['LocationAbbr', 'GradeNumeric'])

    # Create the bar plot for the initial view with data for year 2021
    fig16 = px.bar(soda_increase_2021_sorted, x='LocationAbbr', y='DataValue', color='Stratification1', 
                  category_orders={'LocationAbbr': sorted(npaw_soda_increase_grade['LocationAbbr'].unique())},
                  facet_col_wrap=1, labels={'DataValue': 'Percentage', 'Stratification1': 'Grade'})

    fig16.update_layout(title='''
    Percentage of high school students who consumed at least one soda per day
    <br>among locations which reported an increase in overall high school student soda
    <br>consumption 2021 vs 2019, stratified by grade''',
                       xaxis_title='Location', margin=dict(t=120))  # Adjust top margin to make room for the title)

    # Create dropdown menu to switch between datasets
    dropdown_menu16 = [{'label': 'Data for 2021', 'method': 'update',
                        'args': [{'y': [soda_increase_2021_sorted[soda_increase_2021_sorted['Stratification1'] == 'Grade 9']['DataValue'].tolist(),
                                        soda_increase_2021_sorted[soda_increase_2021_sorted['Stratification1'] == 'Grade 10']['DataValue'].tolist(),
                                        soda_increase_2021_sorted[soda_increase_2021_sorted['Stratification1'] == 'Grade 11']['DataValue'].tolist(),
                                        soda_increase_2021_sorted[soda_increase_2021_sorted['Stratification1'] == 'Grade 12']['DataValue'].tolist()],
                                  'colorbar.title': 'Percentage'}]},
                       {'label': 'Data for 2019', 'method': 'update',
                        'args': [{'y': [soda_increase_2019_sorted[soda_increase_2019_sorted['Stratification1'] == 'Grade 9']['DataValue'].tolist(),
                                        soda_increase_2019_sorted[soda_increase_2019_sorted['Stratification1'] == 'Grade 10']['DataValue'].tolist(),
                                        soda_increase_2019_sorted[soda_increase_2019_sorted['Stratification1'] == 'Grade 11']['DataValue'].tolist(),
                                        soda_increase_2019_sorted[soda_increase_2019_sorted['Stratification1'] == 'Grade 12']['DataValue'].tolist()],
                                  'colorbar.title': 'Percentage'}]}]

    # Add dropdown menu to the figure
    fig16.update_layout(updatemenus=[{'buttons': dropdown_menu16,
                                      'direction': 'down',
                                      'showactive': True,
                                      'x': 1.23,
                                      'xanchor': 'right',
                                      'y': 0.3,
                                      'yanchor': 'top'}])

    # Convert the plot to a DIV element (HTML string)
    plot_div13 = plot(fig13, output_type='div', include_plotlyjs=False)
    plot_div14 = plot(fig14, output_type='div', include_plotlyjs=False)
    plot_div15 = plot(fig15, output_type='div', include_plotlyjs=False)
    plot_div16 = plot(fig16, output_type='div', include_plotlyjs=False)

    return render_template('index.html', plot_div1=Markup(plot_div1), plot_div2_1=Markup(plot_div2_1), plot_div2_2=Markup(plot_div2_2),plot_div3=Markup(plot_div3), \
                             plot_div5=Markup(plot_div5), plot_div6=Markup(plot_div6),\
                             plot_div7=Markup(plot_div7), plot_div8=Markup(plot_div8), plot_div9=Markup(plot_div9), \
                             plot_div10=Markup(plot_div10), plot_div11=Markup(plot_div11), plot_div12=Markup(plot_div12),\
                             plot_div13=Markup(plot_div13), plot_div14=Markup(plot_div14), plot_div15=Markup(plot_div15), plot_div16=Markup(plot_div16))  # Render the home page
    # removed plot_div3=Markup(plot_div3) plot_div4=Markup(plot_div4),

@app.route('/details', methods=['GET'])
def get_details():
    return render_template('details.html')

@app.route('/analysis', methods=['GET'])
def get_analysis():
    return render_template('analysis.html')

@app.route('/predict_model', methods=['POST'])
def predict_model():
    # Collecting all inputs
    linear_features = ['aerobic', 'disability', 'depression', 'distress', 'drinking', 'sleep', 
                     'viggie', 'fruit', 'chronic', 'health', 'physical', 'diabetes', 'asthma', 
                     'dentist', 'blood', 'pain', 'inactivity', 'cholesterol', 'activity', 
                     'unemployment', 'copd', 'checkup', 'smoking', 'medication', 'poverty', 'teeth']
    
    # xgboost_features = linear_features + ['unhealthy', 'alcohol', 'life', 'limitation', 'phyunhealthy', 'food', 'transport', 'support', 'bills']
    model_key = request.form['model_selection']
    feature_names = linear_features
    
    features = []
    for feature in feature_names:
        value = request.form.get(feature, type=float)  # Using get to avoid errors if any field is somehow missing
        if value is None:
            return render_template('error.html', error=f"Missing value for {feature}")
        features.append(value)

    model = models[model_key]
    

    prediction = model.predict([features])
    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
