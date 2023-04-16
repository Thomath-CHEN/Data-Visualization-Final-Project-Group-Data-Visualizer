'''
    Author Info:
        Dynamic Viewer: Yaqi CHEN
        Electricity Trends: Nan CHEN
        Electricity Source: Zeli PAN
        Emission Predictor: Taolue CHEN
        
'''

import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import altair as alt
import plotly.express as px
import geopandas as gpd


st.set_page_config(page_title='World CO2 Emission Inspector', layout='wide')

# You can change to your own directory here
path = "E:/硕士学习/第二年第二学期/Data Visualization/Assignment3/webpage files/"

def Dynamic_Viewer(path):
    
    # Author: Yaqi CHEN

    st.title("Welcome to Dynamic Viewer")

    helper = "This page visualizes the percentage of CO2 Emission in Transport, Manufacturing, "+\
             "Commerce and Other Sectors for each country in each year. The 4 world maps displayed below present the spatial "+\
             "distribution of these factors. To interact with the graph, hover the mouse pointer "+\
             "on the countries to view the detailed data. To zoom in or zoom out, pitch or spread with two "+\
             "fingers on touchpad or use mouse whell. Click ▶️ to play the animation and ▣ to stop it. Or you can slide the time bar to "+\
             "choose a specific year. Drag the map to view other locations."

    st.button("help", help = helper)

    with open(path+"CO2byTransportMap%.html", "r") as file:
        html_content1 = file.read()
        
    with open(path+"CO2byCommerceMap%.html", "r") as file:
        html_content2 = file.read()

    with open(path+"CO2byManufacturingMap%.html", "r") as file:
        html_content3 = file.read()

    with open(path+"CO2byOthergMap%.html", "r") as file:
        html_content4 = file.read()

    # Divide the screen into four columns using Streamlit's beta_columns() function
    col1, col2= st.columns(2)

    # Display each map in a separate IFrame in a 2x2 format
    with col1:
        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">% of CO2 Emission in Transport</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.components.v1.html(html_content1, width=700, height=600)
        
    with col2:
        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">% of CO2 Emission in Manufacturing</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.components.v1.html(html_content3, width=700, height=600)

    col3, col4= st.columns(2)
    # Display each map in a separate IFrame in a 2x2 format
    with col3:
        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">% of CO2 Emission in Commerce</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.components.v1.html(html_content2, width=700, height=600)
        
    with col4:
        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">% of CO2 Emission in Other Sectors</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.components.v1.html(html_content4, width=700, height=600)

def Electricity_Trends(path):

    # Author: Nan CHEN    

    #st.set_page_config(page_title="Emission Trends")

    st.title("Welcome to Electricity Consumption Trends")

    st.title("⬇ Longitudinal data on electrical energy consumption")

    image = Image.open(path+'Pic_2.png')

    st.image(image, caption='World electricity consumption by country in 2014')

    st.write("This table presents the electricity consumption per capita by country in 2014. The country with the highest electricity consumption per capita was Qatar, with a consumption of 42,164 kilowatt hours (kWh) per person. This is significantly higher than the second-highest country, Kuwait, which had a consumption of 17,545 kWh per person. The other countries in the top 10 also had high electricity consumption per capita levels, with most exceeding 10,000 kWh per person. It is interesting to note that some of the countries with the highest electricity consumption per capita, such as Qatar and Kuwait, are also major oil and gas exporters, which may contribute to their high levels of energy consumption.")

    st.title("⬇ Cross-sectional data on electricity consumption (major countries)")

    image = Image.open(path+'Pic_1.png')

    st.image(image, caption='Top 10 Highest Electric power consumption (kWh per capita) Over Time')
    st.write("This table presents the top 10 countries with the highest electric power consumption per capita over time, from 2006 to 2014. The data shows that Qatar had the highest electric power consumption per capita throughout the period, with a peak of 54,276 kWh in 2010. The other countries in the top 10 also had high consumption levels, with most exceeding 10,000 kWh per capita. It is worth noting that high electric power consumption per capita may not necessarily indicate high living standards, as it could also reflect the inefficiency of the electricity generation and distribution system in some countries. Nonetheless, the trend in the top 10 countries suggests a continued increase in electric power consumption per capita over time, driven by population growth, urbanization, and economic development.")

def Electricity_Source(path):
    
    # Author: Zeli_PAN
    
    st.title("Welocome to Electricity Source")
    
    # ==========
    
    df_raw = pd.read_csv(path+"FFG_Hackathon_Country_Level_Data.csv")
    # read raw data file
    
    df = df_raw[['Year', 'Country Name', 'Country Code', 'EG.ELC.FOSL.ZS', 'EG.ELC.HYRO.ZS', 'EG.ELC.NUCL.ZS', 'EG.ELC.RNWX.ZS', 'EG.USE.ELEC.KH.PC', 'SP.RUR.TOTL', 'SP.URB.TOTL']]
    # create a dataframe with only columns that will be used in data visualization
    
    df.dropna(inplace=True)
    # remove rows with NaN values because some data is not available for certain countries for a certain year (notably 2016)
    
    df['Total Electricity Generation (billion KWh)'] = df['EG.USE.ELEC.KH.PC'] * ( df['SP.RUR.TOTL'] + df['SP.URB.TOTL'] ) / 1000000000
    # calculate total electricity generation in billion KWh for the country from electricity consumption per capita * (rural population + urban population) / 1 billion
    
    df.drop(['EG.USE.ELEC.KH.PC', 'SP.RUR.TOTL', 'SP.URB.TOTL'], axis=1, inplace=True)
    # remove electricity consumption per capita, rural population, and urban population columns that are only used to facilitate the calculation of total electricity generation
    
    df.rename(columns={'EG.ELC.FOSL.ZS':'Fossil %', 'EG.ELC.HYRO.ZS':'Hydro %', 'EG.ELC.NUCL.ZS':'Nuclear %', 'EG.ELC.RNWX.ZS':'Other Renewable Excluding Hydro %'}, inplace=True)
    # remane columns
    
    df = pd.concat([df[['Year', 'Country Name', 'Country Code']], df[['Fossil %', 'Hydro %', 'Nuclear %', 'Other Renewable Excluding Hydro %', 'Total Electricity Generation (billion KWh)']].round(2)], axis=1)
    # round to 2 decimal places
    
    df["Year"]=df["Year"].apply(str)
    # format the Year column to string
    
    # ==========
    
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    gdf = gpd.read_file(world_filepath)
    gdf.rename(columns={"name":"Country Name"}, inplace=True)
    # get the coordinates of countries and rename the column
    
    gdf.loc[gdf['Country Name']=='United States of America', 'Country Name']='United States'
    gdf.loc[gdf['Country Name']=='Russia', 'Country Name']='Russian Federation'
    gdf.loc[gdf['Country Name']=='Dem. Rep. Congo', 'Country Name']='Congo, Dem. Rep.'
    gdf.loc[gdf['Country Name']=='Dominican Rep.', 'Country Name']='Dominican Republic.'
    gdf.loc[gdf['Country Name']=='Bosnia and Herz.', 'Country Name']='Bosnia and Herzegovina'
    gdf.loc[gdf['Country Name']=='Brunei', 'Country Name']='Brunei Darussalam'
    gdf.loc[gdf['Country Name']=="Côte d'Ivoire", 'Country Name']="Cote d'Ivoire"
    gdf.loc[gdf['Country Name']=="Egypt", 'Country Name']="Egypt, Arab Rep."
    gdf.loc[gdf['Country Name']=="South Korea", 'Country Name']="Korea, Rep."
    gdf.loc[gdf['Country Name']=="Venezuela", 'Country Name']="Venezuela, RB"
    # change the country names that are inconsistent
    
    gdf_joined = gdf.set_index("Country Name").join(df.set_index("Country Name"),how="right")
    # join our data with the external data to get the geographic cooridnates of each country
    
    gdf_joined.dropna(inplace=True)
    # remove rows with empty cells
    
    overview_map = px.choropleth(gdf_joined, \
                             hover_name=gdf_joined.index, \
                             geojson=gdf_joined.set_index("Country Code"), \
                             locations="Country Code", \
                             animation_frame="Year", \
                             color="Total Electricity Generation (billion KWh)", \
                             color_continuous_scale="oranges", \
                             hover_data=["Fossil %", "Hydro %", "Nuclear %", "Other Renewable Excluding Hydro %"], \
                             title="hover mouse over a country to see electricity generation % by source and total output")
    # generate the overview map
    
    overview_map.update_layout(width=1000, height=600)
    # make overview_map larger
    
    st.write(overview_map)
    # let Streamlit display the overview map
    
    # ==========
    
    country_list = sorted(list(set(df['Country Name'])))
    # obtain a country list from the Country Name column 
    
    country_selected = st.selectbox("Select Country", options=country_list)
    # create a dropdown list on Streamlit for user to select a country
    
    df_filtered = df[df['Country Name']==country_selected]
    # filter dataframe to the country selected
    
    # ==========
    
    df_source = df_filtered[['Year', 'Country Name', 'Fossil %', 'Hydro %', 'Nuclear %', 'Other Renewable Excluding Hydro %']]
    # select columns to build the "percentage of electricity generation by source" chart
    
    df_source = pd.melt(df_source, id_vars=['Year', 'Country Name'], \
                        value_vars=['Fossil %', 'Hydro %', 'Nuclear %', 'Other Renewable Excluding Hydro %'], \
                        var_name='Source Category', \
                        value_name='Source Percentage', \
                        ignore_index=True)
    # use pd.melt to unpivot the df_source dataframe so that Altair can assign different colors to different source categories
    
    base_source = alt.Chart(df_source).encode(
        x=alt.X('Year:T', scale=alt.Scale(padding=20)),
        y='Source Percentage:Q',
        color=alt.Color('Source Category:N', scale=alt.Scale(scheme='category10')),
        tooltip='Source Percentage:Q'
    )\
    .properties(width=450, height=300, title='percentage of electricity generation by source')
    # build a base for percentage of electricity generation by source
    
    source_chart = base_source.mark_line() + base_source.mark_circle()
    # generate line chart with data point marks on the base_source
    
    # ==========
    
    df_generation = df_filtered[['Year', 'Country Name', 'Total Electricity Generation (billion KWh)']]
    # select columns to build the "total electricity generation" chart
    
    electricity_generation_chart = alt.Chart(df_generation).mark_bar(size=30).encode(
        x=alt.X('Year:T', scale=alt.Scale(padding=20)),
        y='Total Electricity Generation (billion KWh):Q',
        tooltip='Total Electricity Generation (billion KWh):Q'
    )\
    .properties(width=450, height=300, title='total electricity generation')
    # generate the bar chart for electricity generation
    
    # ==========
    
    combined_chart = alt.vconcat(source_chart, electricity_generation_chart)
    
    st.write(combined_chart)
    # let Streamlit display both the "percentage of electricity generation by source" and the "total electricity generation" charts
    
    # ==========
    
    st.write(df_filtered)
    
    # let Streamlit display the data table


def Emission_Predictor(path):

    # Author: Taolue CHEN

    # Load the RandomForest model
    model = joblib.load(path+'rf_model.pkl')

    # Load the emission data
    database = pd.read_csv(path+"total_data.csv")

    # Define a function to make predictions
    def predict(input_values):
        return model.predict([input_values])[0]

    def similar(icon_values, database):
        
        target = np.array(icon_values[1:])
        
        distances = np.sqrt(np.sum((database.iloc[:,1:-2] - target)**2, axis=1))
        
        closest_indices = distances.argsort()[:3]
        
        return closest_indices.to_list()

    # Layout and app components

    #st.set_page_config(page_title='Emission Predictor', layout='wide')


    st.title("Welcome to Emission Predictor")

    helper = "This page Predicts, Visualizes and Compares CO2 emission of a given country"+\
        " or entity. With various interaction components deployed, the page empowers users’ "+\
            "exploration in the visualization. Users can enter values for each feature, and click"+\
                " “Predict” button to view predicted CO2 emission (in kilo tons). Meanwhile, the "+\
                    "emission is visualized by 3 representatives, # trees cut, # of cars’ exhaust and"+\
                        " degrees of temperature increased. On emission comparer, users hover the mouse"+\
                            " pointer on the bard and the line to view detailed data."

    st.button("help", help = helper)

    # Load icon images
    icon_names = ['year', 'clean', 'power', 'land', 'renewable', 'gdp', 'population', 'urban']
    value_names = ["Year", "Clean Energy Share", "Power Generation", "Land Area", "Renewable Energy Share", "GDP", "Population", "Urbanization"]
    icon_paths = [f'{name}.jpg' for name in icon_names]
    icons = [Image.open(path+img_path) for img_path in icon_paths]
    exp_values = [2007,5.89,3339.46,743532,30.46,1.73E+11,16530195,0.87]

    # Create 2x4 layout for icons and input boxes
    icon_values = [None] * 8
    for i in range(2):
        cols = st.columns(4)
        for j in range(4):
            icon_index = i * 4 + j
            with cols[j]:
                st.image(icons[icon_index], width=200) 
                if icon_index != 0:
                    icon_values[icon_index] = st.number_input(f'{value_names[icon_index]}', value=exp_values[icon_index])
                else:
                    icon_values[icon_index] = st.number_input(f'{value_names[icon_index]}', value=exp_values[0])
                
    all_entered = all(list(map(lambda x: x is not None, icon_values)))

    # Predict button and output display
    cols_pred = st.columns(4)
    prediction = None
    if cols_pred[0].button('Predict') and all_entered:
        prediction = round(predict(icon_values),2)
        cols_pred[1].write(f'Answer: {prediction} kilo tons')

    # Dropdown and image representation
    img_option = cols_pred[3].selectbox("Representation", ['trees', 'cars', 'thermometer'])
    represent_pic_path = ["tree.png", "car.png", "thermo.png"]
    represents = [Image.open(path+img_path) for img_path in represent_pic_path]
    with cols_pred[2]:  
        if prediction is not None:
            if img_option == 'trees':
                amount = str(int(prediction * 45872))  # number of trees cut
                st.image(represents[0], caption = amount + " Trees Cut",width = 200)
            elif img_option == 'cars':
                amount = str(int(prediction * 217))  # number os cars' exhaust
                st.image(represents[1], caption = amount + " Cars' Exhaust", width = 200)
            elif img_option == 'thermometer':
                amount = str(round(prediction * 8e-10, 5))  # temperature increased
                st.image(represents[2], caption = amount + " °C In creased", width = 200)
        else:
            st.write("Please predict the CO2 Emission first")
            

    if prediction is not None:
        # Plot similar countries
        idx = similar(icon_values, database)
        data = database.loc[idx,['CO2 Emission','Name','Year']].sort_values(by = "CO2 Emission",
                                                                     ascending = False)
        
        data.columns = ["CO2 Emission", "Country", "Year"]
        data['Prediction'] = prediction 
        
        # plot CO2 Emission of similar contries
        bar_chart = alt.Chart(data).mark_bar().encode(
            x='Country',
            y='CO2 Emission',
            color='Year:O', 
            tooltip=['Country', 'CO2 Emission']
        ).interactive()
        
        # Create a horizontal line at y = prediction
        line_chart = alt.Chart(data).mark_rule(color='red').encode(
            y = 'Prediction:Q',
            size = alt.value(2.5)
        ).interactive()
        
        combined_chart = bar_chart + line_chart
        
        st.altair_chart(combined_chart, use_container_width=True)

    else:
        st.write("Please predict the CO2 Emission first")


if __name__ == "__main__":
    
    # combine all the pages together
    
    page_names_to_funcs = {
        "Dynamic Viewer": Dynamic_Viewer,
        "Electricity Trends": Electricity_Trends,
        "Electricity Source": Electricity_Source,
        "Emission Predictor": Emission_Predictor
    }
    
    demo_name = st.sidebar.selectbox("Choose an App", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name](path)


