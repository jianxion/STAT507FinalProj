from flask import Flask, request, render_template, Markup
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
from flask import Markup


app = Flask(__name__)
models = {
    'model1': joblib.load('linear_regressor_model.pkl'),
    # 'model2': joblib.load('random_forest_regressor_model.pkl'),
    'model3': joblib.load('xgb_regressor_model.pkl')
}

@app.route('/', methods=['GET'])
def home():
    original_df = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')
    df = original_df.loc[original_df['Topic'] == 'Nutrition, Physical Activity, and Weight Status', :]
    df = df.dropna(axis=1, how='all')

    fig1 = px.choropleth(
    df,
    locations='LocationAbbr',  
    locationmode='USA-states',  
    color='DataValue',  
    color_continuous_scale=px.colors.sequential.Viridis,  # color scale
    scope="usa",  
    labels={'DataValue': 'Percentage Meeting Guidelines'},
    title='Percentage of Children Meeting Aerobic Guidelines by State'
    )

    fig1.update_layout(
    geo=dict(
        lakecolor='white'  # changes the lake color to white
    ),
    margin={"r":0,"t":0,"l":0,"b":0}  
    )
    # Filter to the specific question of interest
    filtered_df = df[df["Question"] == "Children and adolescents aged 6-13 years meeting aerobic physical activity guideline"]

    fig2 = px.bar(filtered_df, x='LocationAbbr', y='DataValue', title='Percentage of Children Meeting Aerobic Guidelines by State', labels={'LocationAbbr': 'State', 'DataValue': 'Percentage'})

    # EDA 3
    # Histogram of Data Values
    fig3 = px.histogram(filtered_df, x='DataValue', nbins=10, title='Distribution of Percentages of Children Meeting Aerobic Guidelines')
    fig3.update_xaxes(title_text='Percentage')
    fig3.update_yaxes(title_text='Frequency')
    
    # EDA 4
    # Boxplot by Race Group (ensure race_cat is properly filtered)
    race_cat = filtered_df[filtered_df['StratificationCategory1'] == 'Race/Ethnicity']
    fig4 = px.box(race_cat, x='Stratification1', y='DataValue', title='Variation in Meeting Aerobic Guidelines by Race Group', labels={'Stratification1': 'Race Group', 'DataValue': 'Percentage'})
    
    # Convert the plot to a DIV element (HTML string)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    #Render the home page
    return render_template('index.html', plot_div1=Markup(plot_div1), plot_div2=Markup(plot_div2), plot_div3=Markup(plot_div3), \
                           plot_div4=Markup(plot_div4))  

@app.route('/predict_model', methods=['POST'])
def predict_model():
    # Collecting all inputs
    linear_features = ['aerobic', 'disability', 'depression', 'distress', 'drinking', 'sleep', 
                     'viggie', 'fruit', 'chronic', 'health', 'physical', 'diabetes', 'asthma', 
                     'dentist', 'blood', 'pain', 'inactivity', 'cholesterol', 'activity', 
                     'unemployment', 'copd', 'checkup', 'smoking', 'medication', 'poverty', 'teeth']
    
    xgboost_features = linear_features + ['unhealthy', 'alcohol', 'life', 'limitation', 'phyunhealthy', 'food', 'transport', 'support', 'bills']
    model_key = request.form['model_selection']
    print("selected", model_key)
    if model_key == 'model1':
        feature_names = linear_features
    else:
        feature_names = xgboost_features
    print(feature_names)
    
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
