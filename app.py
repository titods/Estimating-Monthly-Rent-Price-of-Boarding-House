import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = None
sns.set_style('darkgrid')

import streamlit as st
import folium
import branca.colormap as cm
from streamlit_folium import folium_static
import pickle
from PIL import Image
import shap

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

# function to get the preprocessing pipeline
def get_feature_pipeline(numerical, nominal, ordinal, binary, algorithm, ordinal_category):
    # get every features as their data types
    get_numerical = FunctionTransformer(lambda x: x[numerical], validate = False)
    get_nominal = FunctionTransformer(lambda x: x[nominal], validate = False)
    get_ordinal = FunctionTransformer(lambda x: x[ordinal], validate = False)
    get_binary = FunctionTransformer(lambda x: x[binary], validate = False)
    
    # preprocessing pipeline for tree-based algorithm
    if algorithm == 'tree-based':
        pipeline_numerical = Pipeline([('numerical', get_numerical)])
        pipeline_nominal = Pipeline([('nominal', get_nominal),
                                     ('nominal-encoder', OneHotEncoder())])
        pipeline_ordinal = Pipeline([('ordinal', get_ordinal),
                                     ('ordinal-encoder', OrdinalEncoder(categories = ordinal_category))])
        pipeline_binary = Pipeline([('binary', get_binary)])
        feature_pipeline = FeatureUnion([('pipeline_numerical', pipeline_numerical),
                                         ('pipeline_nominal', pipeline_nominal),
                                         ('pipeline_ordinal', pipeline_ordinal),
                                         ('pipeline_binary',pipeline_binary)])
        return feature_pipeline
    
    # preprocessing pipeline for other than tree-based algorithm 
    elif algorithm == 'non-tree-based':
        pipeline_numerical = Pipeline([('numerical', get_numerical),
                                       ('scaler', MinMaxScaler())])  # scaler               
        pipeline_nominal = Pipeline([('nominal', get_nominal),
                                     ('nominal-encoder', OneHotEncoder())])
        pipeline_ordinal = Pipeline([('ordinal', get_ordinal),
                                     ('ordinal-encoder', OrdinalEncoder(categories = ordinal_category)),
                                     ('scaler', MinMaxScaler())])    # scaler
        pipeline_binary = Pipeline([('binary', get_binary)])
        feature_pipeline = FeatureUnion([('pipeline_numerical', pipeline_numerical),
                                         ('pipeline_nominal', pipeline_nominal),
                                         ('pipeline_ordinal', pipeline_ordinal),
                                         ('pipeline_binary',pipeline_binary)])
        return feature_pipeline
    else:
        print('error in algorithm input argument. try "tree-based" or "non-tree-based"')

def main():
    #### load and prepare everything ####
    # load the dataset
    df_EDA = pd.read_csv('dataset/Kost Data (For EDA).csv')
    df_model = pd.read_csv('dataset/Kost Data (For Modelling).csv')
    df_summary_model_eval = pd.read_csv('dataset/summary_model_eval.csv', index_col = [0, 1]).loc[['stacking_meta_LinReg', 'LGBM_no_outliers_tuned']].rename(index = {'stacking_meta_LinReg':'Stacking Regressor', 'LGBM_no_outliers_tuned':'Tuned LightGBM'})
    pred_interval = pd.read_csv('dataset/summary_pred_int.csv')
    
    # create dictionaries for mapping user input with the feature in the DataFrame
    room_facilities = { 'Water Heater':'fasilitas_kamar_pemanas air',
                        'Shower':'fasilitas_kamar_shower',
                        'Sink':'fasilitas_kamar_wastafel',
                        'AC':'fasilitas_kamar_ac',
                        'Chair':'fasilitas_kamar_kursi',
                        'Wardrobe':'fasilitas_kamar_lemari',
                        'Table':'fasilitas_kamar_meja',
                        'Bed Sheet':'fasilitas_kamar_sprei',
                        'TV':'fasilitas_kamar_tv',
                        'TV Cable':'fasilitas_kamar_tv kabel',
                        'Window to the Outside':'fasilitas_kamar_jendela kearah luar',
                        'Including Electricity':'fasilitas_kamar_termasuk listrik',
                        'In-room Bathroom':'fasilitas_kamar_kamar mandi dalam'
                        }
    building_facilities = { 'Dining Room':'area makan',
                            'Parking Area':'area parkir',
                            'CCTV':'cctv',
                            'Cleaning Service':'cleaning service',
                            'Kitchen':'dapur',
                            'Laundry':'laundry',
                            'Washing Machine':'mesin cuci',
                            'Microwave':'microwave',
                            'Living Room':'ruang santai / ruang tamu',
                            'Clothes Drying Area':'tempat jemur pakaian',
                            'Wifi':'wifi',
                            'Near Mall':'sekitar_gedung_mall',
                            'Near Supermarket':'sekitar_gedung_supermarket',
                            'Near Healtcare Access':'sekitar_gedung_akses kesehatan',
                            'Near School':'sekitar_gedung_sekolah',
                            'Near Transportation Access':'sekitar_gedung_akses transportasi',
                            'Stove':'kompor',
                            'Fridge':'kulkas',
                            'Water Dispenser':'dispenser',
                            'Mixed (Male and Female) Boarding House':'jenis_kost_campur',
                            'in Jakarta':'jakarta_check'
                            }
    others = {  'luas_kost':'Room Area', 
                'total_fasilitas_kamar':'Total Room Facilities', 
                'total_fasilitas_gedung':'Total Building Facilities', 
                'total_fasilitas':'Total (Room and Building) Facilities)',
                'fasilitas_kamar_jumlah kasur':'The Number of Bed(s)',
                'fasilitas_kamar_tipe kasur':'Bed Type',
                'kota_Jakarta Selatan': 'City_Jakarta Selatan',
                'kota_Jakarta Pusat': 'City_Jakarta Pusat',
                'kota_Jakarta Barat': 'City_Jakarta Barat',
                'kota_Jakarta Timur': 'City_Jakarta Timur',
                'kota_Tangerang': 'City_Tangerang',
                'kota_Jakarta Utara': 'City_Jakarta Utara',
                'kota_Depok': 'City_Depok',
                'kota_Bandung': 'City_Bandung'
             }
    features = ['luas_kost', 'total_fasilitas_kamar', 'total_fasilitas_gedung', 'total_fasilitas', 'kota_Bandung', 'kota_Depok', 'kota_Jakarta Barat', 'kota_Jakarta Pusat', 'kota_Jakarta Selatan', 'kota_Jakarta Timur', 'kota_Jakarta Utara', 'kota_Tangerang', 'fasilitas_kamar_jumlah kasur', 'fasilitas_kamar_tipe kasur', 'area makan', 'area parkir', 'cctv', 'cleaning service', 'dapur', 'laundry', 'mesin cuci', 'microwave', 'ruang santai / ruang tamu', 'tempat jemur pakaian', 'wifi', 'sekitar_gedung_mall', 'sekitar_gedung_supermarket', 'sekitar_gedung_akses kesehatan', 'sekitar_gedung_sekolah', 'sekitar_gedung_akses transportasi', 'fasilitas_kamar_pemanas air', 'fasilitas_kamar_shower', 'fasilitas_kamar_wastafel', 'fasilitas_kamar_ac', 'fasilitas_kamar_kursi', 'fasilitas_kamar_lemari', 'fasilitas_kamar_meja', 'fasilitas_kamar_sprei', 'fasilitas_kamar_tv', 'fasilitas_kamar_tv kabel', 'kompor', 'kulkas', 'dispenser', 'fasilitas_kamar_jendela kearah luar', 'jenis_kost_campur', 'fasilitas_kamar_termasuk listrik', 'fasilitas_kamar_kamar mandi dalam', 'jakarta_check']
    mapper = {}
    mapper.update(building_facilities)
    mapper.update(room_facilities)
    new_mapper = {}            
    for key, val in mapper.items():
        new_mapper[val] = key
    new_mapper.update(others)
    
    # load the models
    LGBM_final = pickle.load(open('model/LGBM_final.sav', 'rb'))
    LGBM_001 = pickle.load(open('model/LGBM_001.sav', 'rb'))
    LGBM_099 = pickle.load(open('model/LGBM_099.sav', 'rb'))
    LGBM_095 = pickle.load(open('model/LGBM_095.sav', 'rb'))
    LGBM_005 = pickle.load(open('model/LGBM_005.sav', 'rb'))
    
    #### Navigation ####
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.selectbox('Select Page', ['Home Page', 'Overview', 'Data Visualization', 'Monthly Rent Price Estimator'])
    
    ## sidebar: Home Page ##
    if selected_page == 'Home Page':
        st.title('Estimating Monthly Rent Price of Boarding House Project')
        st.write('Open the navigation sidebar on the left to select the pages')
        st.image('image/Boarding House.jpg')
        st.caption('Author: Tito Dwi Syahputra')
    
    ## siderbar: Overview ##    
    elif selected_page == 'Overview':
        st.title('Overview')
        
        st.header('Objective')
        st.markdown('The main objective of this project is to develop a machine learning regression model that can be used to estimate monthly rent price of boarding house, specifically in Jakarta, Tangerang, Bandung, and Depok.')
        st.markdown('The second objective is to know which factors mostly determine the monthly rent price of a boarding house.')
        
        st.header('Dataset')
        st.markdown('The dataset used to develop the model is scraped from [infokost.id](https://infokost.id/).')
        st.markdown('At the first iteration, I want to estimate the monthly rent price for region in Jabodetabek and Bandung, so I scraped for all of those regions, but unfortunately the website does not have enough data for region in Bogor and Bekasi. Therefore I only use boarding house data in Jakarta, Tangerang, Bandung, and Depok.')
        st.markdown('The very first scraped dataset consists of 5024 boarding house rooms, but there are lots of missing values. Since I cant simply impute those missing values with simple descriptive statistics (like mean, median, or mode), I have to drop them. After data cleaning and some considerations during the modeling process, In the end the dataset consists of 2535 boarding house rooms.')
        st.markdown('''41 features are used to develop the final machine learning model:
- 4 numerical features, i.e. "Room Area", "Total (Room and Building) Facilities", "Total Room Facilities", and "Total Building Facilities"
- 1 nominal feature, i.e. "City"
- 2 ordinal features, i.e. "The Number of Bed(s)" and "Bed Type"
- 34 binary features describing whether a facility is present (labeled as 1) or not (labeled as 0), i.e. "Water Heater", "AC", "TV", etc''')                                 
                    
        st.header('The Final Machine Learning Model')
        st.caption('Expand the table to read the full model description')
        df_summary_model_eval['Model Description'] = ''
        df_summary_model_eval['Model Description'].loc['Tuned LightGBM'] = 'Boosting Ensemble method tuned using RandomizedSearchCV'
        df_summary_model_eval['Model Description'].loc['Stacking Regressor'] = 'Stacking Ensemble method using "Tuned Random Forest" model and "Tuned LightGBM" model as the base-learner, then finally Linear regression as the meta-learner'
        st.write(df_summary_model_eval[['R^2', 'RMSE', 'MAE', 'Model Description']])
        st.markdown('I have tried several machine learning algorithms such as Linear Regression, Random Forest, and LightGBM. Also I have tried to use other ensemble methods such as Voting and Stacking Regressor. At the end of the day **I choose "tuned LightGBM" model to be my final model** to estimate the monthly rent price. In addition, this model is the second best performing model, meanwhile the first one is "Stacking Regressor".')
        st.markdown('Why I choose the second best performing model instead of the  first one? There are some considerations, the first is because the "Stacking Regressor" model is more overfitting the training set than a single model "tuned LightGBM". Moreover, I also want to generate the prediction interval, and I found that `lightgbm` library provides the easiest way to generate it. Furthermore, the gap of model performances between "Stacking Regressor" and "Tuned LightGBM" is relatively small (The difference on test set is only 0.006 on R-squared, around 5,409 IDR on RMSE, and around 3,681 IDR on MAE). Finally, "Tuned LightGBM" model is easier to be interpreted than "Stacking Regressor". Essentially the more complex the model is, the harder the model to be interpreted.')
        if st.checkbox('Do you want to see the prediction interval?', help = 'Please uncheck this if you want to see the plots of SHAP values clearly'):
            st.write(pred_interval.describe())
            fig, ax = plt.subplots(figsize = (12, 6), nrows = 1, ncols = 2)
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Quantile 0.05'], marker = 'x', label = 'lower bound', ax = ax[0], alpha = 0.5)
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Predicted Price'], label = 'pred', ax = ax[0])
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Quantile 0.95'], marker = 'x', label = 'upper bound', ax = ax[0], alpha = 0.5)
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Quantile 0.05']), ax = ax[0])
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Predicted Price']), ax = ax[0])
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Quantile 0.95']), ax = ax[0])
            ax[0].set_xlabel('Actual Price (IDR)')
            ax[0].set_ylabel('Predicted Price (IDR)')
            ax[0].set_title('95 % Prediction Interval')
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Quantile 0.01'], marker = 'x', label = 'lower bound', ax = ax[1], alpha = 0.5)
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Predicted Price'], label = 'pred', ax = ax[1])
            sns.scatterplot(pred_interval['Actual Price'], pred_interval['Quantile 0.99'], marker = 'x', label = 'upper bound', ax = ax[1], alpha = 0.5)
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Quantile 0.01']), ax = ax[1])
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Predicted Price']), ax = ax[1])
            sns.lineplot(sorted(pred_interval['Actual Price']), sorted(pred_interval['Quantile 0.99']), ax = ax[1])
            ax[1].set_xlabel('Actual Price (IDR)')
            ax[1].set_ylabel('Predicted Price (IDR)')
            ax[1].set_title('99 % Prediction Interval')
            plt.suptitle('Tuned LightGBM model')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.header('Model Interpretation')
        st.subheader('Feature Importance')
        feature_importance = pd.DataFrame({'Feature Name':features, 'Importance':LGBM_final.feature_importances_}).sort_values('Importance', ascending = False).reset_index(drop = True)                                           
        feature_importance['Feature Name'] = feature_importance['Feature Name'].map(new_mapper)
        feature_importance['Rank'] = np.arange(1, feature_importance.shape[0] + 1)
        st.caption('Feature Importance is ranked based on how many of these features are used to make splits within the LightGBM model. The more a feature is used to split the trees, the more important it is.')
        st.write(feature_importance)
        st.write('''Here are some insights that we can get from feature importance:
- The top 5 most important features are "Room Area", "Total (Room and Building) Facilities)", "Total Room Facilities", "Total Building Facilities", and "Bed Type".
- Surprisingly It looks like "the presence of other buildings around the boarding house" are at the top 10 of feature importances, but not all of them share the same importance. "Mall" , "School", and "Transportation Access" are more important than "Supermarket" and "Healthcare Access". Usually, a boarding house located 'strategically' has higher monthly rent price.
- It turns out that "Parking Area", "Table", "CCTV", "Wardrobe", and "Chair" are the facilities which are at the bottom 5 of feature importance. I am not surprised with this fact because most of boarding house commonly has those facilities, especially in Jakarta. That is why the presence of those facilities hardly affect on estimating the monthly rent price.''')
        
        st.subheader('SHAP values')
        st.caption('Unlike feature importance, SHAP values tell us how is the contribution of each feature to the output prediction. In other words, we can see how a model makes its output prediction based on the contribution of each feature to their output prediction.')
        select_shap =  st.radio('Select One', ['Plot all SHAP values on bee-swam-plot', 'Plot all of the mean absolute SHAP values on barplot'], help = 'Please uncheck the displayed prediction interval plots above in order to clearly see the plots of SHAP values')
        explainer = shap.Explainer(LGBM_final)
        shap_values = explainer(df_model.drop('harga_kost_per_bulan', axis = 1).rename(columns = new_mapper))
        if select_shap == 'Plot all SHAP values on bee-swam-plot':
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values, df_model.drop('harga_kost_per_bulan', axis = 1).rename(columns = new_mapper), max_display = 50))
            st.set_option('deprecation.showPyplotGlobalUse', True)
        elif select_shap == 'Plot all of the mean absolute SHAP values on barplot':
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.plots.bar(shap_values, max_display = 50))
            st.set_option('deprecation.showPyplotGlobalUse', True)
        st.write('''Here are some insights that we can get from SHAP values:
- According to the SHAP values, "Total Room Facilities" is the most impactful feature, followed by "Room Area" and "Water Heater". On average "Total Room Facilities" gives the highest contibution to estimate the increasing or decreasing the monthly rent price, followed by "Room Area" and "Water Heater".
- It seems that the most expensive facility that exist in boarding house is "Water Heater" and followed by "AC". We can see from the bee-warm-plot that those two facilities give higher contribution to the output prediction.
- The LightGBM also capture that "Parking Area" and "CCTV" are the two cheapest facilities in boarding house. In fact, It is a must for a boarding house to have those two facilities, specifically in Jakarta, and everybody usually wants that those facilites exist in her/his boarding house.
- "Washing Machine" and "Wifi" seem like  not included as luxurious facilities. For "Wifi" it is quite obvious that "internet" is a must for everyone to have it so the boarding house owner provides it. Meanwhile for "Washing Machine" probably is due to the fact in our data most of boarding house doesn't have "Washing Machine", and usually most of people who rent boarding house rely on public laundry so that "Washing Machine" is not necessary.''')
        
        st.header('Future Improvement')
        st.markdown('''We should scrap more data from other online platforms of boarding house. Our dataset is extracted from a single online platform which this could make our data quite skewed if the price in [infokost.id](https://infokost.id/) website is not representative in general for all boarding house units in each of their region. Also adding more data points usually decrease the variance error of the model hence it can reduce the overfitting issue which is the case in this project.''')
        
        
        ## siderbar: Data Visualization ##
    elif selected_page == 'Data Visualization': 
        st.title('Boarding House Data Visualization')
        
        st.header('The dataset')
        if st.checkbox('Display the dataset?', False):
            show_data = st.radio('Select One', ['Raw', 'Preprocessed'])
            if show_data == 'Raw':
                st.caption('The "raw" dataset is used to perform EDA')
                st.write(df_EDA)
                st.write(df_EDA.shape)
            elif show_data == 'Preprocessed':
                st.caption('The "preprocessed" dataset is used to train ML model')
                st.write(df_model)
                st.write(df_model.shape)
                
        st.header('Visualization')
        st.caption('Select the tabs below')
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Monthly Rent Price', 'Room Area', 'City', 'Bed Type', 'The Number of Bed(s)', 'Facility'])
        
        # Monthly Rent Price Visualization #
        with tab1:
            st.subheader('Monthly Rent Price Visualization')
            st.write(df_EDA[['harga_kost_per_bulan']].describe().rename({'harga_kost_per_bulan':'Monthly Rent Price'}, axis = 1).T)
            fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)
            sns.distplot(x = df_EDA['harga_kost_per_bulan'], ax = ax[0])
            sns.boxplot(x = df_EDA['harga_kost_per_bulan'], ax = ax[1])
            ax[0].set_title('Data Distribution')          
            ax[1].set_xlabel('Monthly Rent Price (IDR)')
            st.pyplot(fig)
                
        # Room Area Visualization #         
        with tab2:
            st.subheader('Room Area Visualization')
            area_plot = st.selectbox('Select Plot', ['KDEplot and Boxplot (Data Distribution)', 'Scatterplot (Room Area vs Monthly Rent Price)'])
            if area_plot == 'KDEplot and Boxplot (Data Distribution)':
                st.write(df_EDA[['luas_kost']].describe().rename({'luas_kost':'Room Area'}, axis = 1).T)
                fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)
                sns.distplot(x = df_EDA['luas_kost'], ax = ax[0])
                sns.boxplot(x = df_EDA['luas_kost'], ax = ax[1])
                ax[0].set_title('Data Distribution')
                ax[1].set_xlabel('Room Area (m^2)')
                st.pyplot(fig)
            elif area_plot == 'Scatterplot (Room Area vs Monthly Rent Price)':
                groupby = st.selectbox('Group by', ['None', 'The Number of Facilities', 'Type of Bed', 'The Number of Bed(s)', 'City', 'Facility'])
                if groupby == 'None':
                    fig, ax = plt.subplots()
                    sns.regplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost')
                    plt.title('Room Area vs Monthly Rent Price')
                    ax.set_xlabel('Room Area (m^2)')
                    ax.set_ylabel('Monthly Rent Price (IDR)')
                    st.pyplot(fig)
                elif groupby == 'The Number of Facilities':
                    st.caption('Two types of facilities: Room Facilities and Building Facilities')
                    facilities = st.radio('Select which Facility', ['Room', 'Building', 'Room + Building'])
                    if facilities == 'Room':
                        fig, ax = plt.subplots()
                        sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'total_fasilitas_kamar')
                        plt.title('Room Area vs Monthly Rent Price')
                        ax.set_xlabel('Room Area (m^2)')
                        ax.set_ylabel('Monthly Rent Price (IDR)')
                        ax.legend(title = 'Total (Room) Facilities')
                        st.pyplot(fig)
                    elif facilities == 'Building':
                        fig, ax = plt.subplots()
                        sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'total_fasilitas_gedung')
                        plt.title('Room Area vs Monthly Rent Price')
                        ax.set_xlabel('Room Area (m^2)')
                        ax.set_ylabel('Monthly Rent Price (IDR)')
                        ax.legend(title = 'Total (Building) Facilities')
                        st.pyplot(fig)
                    elif facilities == 'Room + Building':
                        fig, ax = plt.subplots()
                        sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'total_fasilitas')
                        plt.title('Room Area vs Monthly Rent Price')
                        ax.set_xlabel('Room Area (m^2)')
                        ax.set_ylabel('Monthly Rent Price (IDR)')
                        ax.legend(title = 'Total (Room + Building) Facilities')
                        st.pyplot(fig)
                elif groupby == 'Type of Bed':
                    fig, ax = plt.subplots()
                    sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'fasilitas_kamar_tipe kasur', hue_order = ['single bed', 'long bed', 'double bed', '≥queen bed'], palette = 'Reds')
                    plt.title('Room Area vs Monthly Rent Price')
                    ax.set_xlabel('Room Area (m^2)')
                    ax.set_ylabel('Monthly Rent Price (IDR)')
                    ax.legend(title = 'Type of Bed')
                    st.caption('The darker the color, The bigger the bed size')
                    st.pyplot(fig)
                elif groupby == 'The Number of Bed(s)':
                    fig, ax = plt.subplots()
                    sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'fasilitas_kamar_jumlah kasur', hue_order = ['0 bed', '1 bed', '2 bed'], palette = 'coolwarm')
                    plt.title('Room Area vs Monthly Rent Price')
                    ax.set_xlabel('Room Area (m^2)')
                    ax.set_ylabel('Monthly Rent Price (IDR)')
                    ax.legend(title = 'The number of Bed(s)')
                    st.pyplot(fig)
                elif groupby == 'City':
                    fig, ax = plt.subplots()
                    sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = 'kota')
                    plt.title('Room Area vs Monthly Rent Price')
                    ax.set_xlabel('Room Area (m^2)')
                    ax.set_ylabel('Monthly Rent Price (IDR)')
                    ax.legend(title = 'City')
                    st.pyplot(fig)
                elif groupby == 'Facility':
                    st.caption('Two types of facilities: Room Facilities and Building Facilities')
                    facilities = st.radio('Select which Facility', ['Room', 'Building'])
                    if facilities == 'Room':
                        room_fac = st.selectbox('Select Room Facility', list(room_facilities.keys()))
                        fig, ax = plt.subplots()
                        sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = room_facilities[room_fac], palette = ['#808080', '#DA0610'])
                        plt.title('Room Area vs Monthly Rent Price')
                        ax.set_xlabel('Room Area (m^2)')
                        ax.set_ylabel('Monthly Rent Price (IDR)')
                        ax.legend(title = room_fac)
                        st.pyplot(fig)
                    elif facilities == 'Building':
                        building_fac = st.selectbox('Select Building Facility', list(building_facilities.keys()))
                        fig, ax = plt.subplots()
                        sns.scatterplot(data = df_EDA, y = 'harga_kost_per_bulan', x = 'luas_kost', hue = building_facilities[building_fac], palette = ['#808080', '#DA0610'])
                        plt.title('Room Area vs Monthly Rent Price')
                        ax.set_xlabel('Room Area (m^2)')
                        ax.set_ylabel('Monthly Rent Price (IDR)')
                        ax.legend(title = building_fac)
                        st.pyplot(fig)
                        
        # City Visualization #
        with tab3:            
            st.subheader('City Visualization')
            city_plot = st.selectbox('Select Plot', ['Map', 'Countplot (Data Distribution)', 'Boxplot and Barplot (City vs Monthly Rent Price)'])
            if city_plot == 'Map':
                cities = st.multiselect('Select a City or Multiple Cities', list(df_EDA['kota'].unique()), default = list(df_EDA['kota'].unique()))
                df_city = df_EDA[df_EDA['kota'].isin(cities)].reset_index(drop = True)
                m = folium.Map(location = [df_city['latitude'].mean(), df_city['longitude'].mean()],
                               zoom_start = 10, min_zoom = 9, max_zoom = 12, tiles = 'cartodbpositron')
                folium.TileLayer('OpenStreetMap').add_to(m)
                folium.TileLayer('Stamen Terrain').add_to(m)
                folium.TileLayer('cartodbpositron').add_to(m)
                folium.TileLayer('cartodbdark_matter').add_to(m)
                folium.LayerControl().add_to(m)
                colormap = cm.LinearColormap(colors = ['white', 'blue'], vmin = df_city['harga_kost_per_bulan'].min(), vmax = df_city['harga_kost_per_bulan'].max())
                m.add_child(colormap) 
                for i in range(df_city.shape[0]):
                    text = '''Monthly Rent Price: Rp.{}'''.format(str(df_city.iloc[i]['harga_kost_per_bulan']))
                    folium.Circle(location = [df_city.iloc[i]['latitude'], df_city.iloc[i]['longitude']],
                                  radius = 100, fill = True, fill_opacity = 0.2, weight = 5,
                                  color = colormap(df_city.iloc[i]['harga_kost_per_bulan']), 
                                  popup = text,
                                  tooltip = 'click to see the details').add_to(m)
                folium_static(m)
            elif city_plot == 'Countplot (Data Distribution)':
                fig, ax = plt.subplots()
                sns.countplot(data = df_EDA, y = 'kota', order = df_EDA.kota.value_counts().index, palette = 'hls')
                for i, val in enumerate(df_EDA.kota.value_counts()):
                    plt.text(s = val, x = val, y = i)
                plt.title('The Number of Units for each City')
                ax.set_xlabel('Count')
                ax.set_ylabel('City')
                st.pyplot(fig)
            elif city_plot == 'Boxplot and Barplot (City vs Monthly Rent Price)':
                fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5.5))
                order = df_EDA.groupby('kota').agg({'harga_kost_per_bulan':'mean'}).sort_values(by = 'harga_kost_per_bulan', ascending = False).index
                sns.boxplot(data = df_EDA, x = 'kota', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[0])
                ax[0].set_title('Monthly Rent Price Distribution for each City', y = 1.05)
                ax[0].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[0].set_xlabel('City')
                ax[0].set_ylabel('Monthly Rent Price (IDR)')
                sns.barplot(data = df_EDA, x = 'kota', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[1])
                ax[1].set_title('Avg. Monthly Rent Price for each City', y = 1.05)
                ax[1].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[1].set_xlabel('City')
                ax[1].set_ylabel('Monthly Rent Price (IDR)')
                plt.tight_layout()
                st.pyplot(fig)
                
        # Bed Type Visualization #
        with tab4:
            st.subheader('Bed Type Visualization')
            st.caption('Basically bed type describes the size of bed: "Single Bed" < "Long Bed" < "Double Bed" < "≥Queen Bed"')
            bed_type_plot = st.selectbox('Select Plot', ['Countplot (Data Distribution)', 'Boxplot and Barplot (Bed Type vs Monthly Rent Price)'])
            if bed_type_plot == 'Countplot (Data Distribution)':
                fig, ax = plt.subplots()
                sns.countplot(data = df_EDA, y = 'fasilitas_kamar_tipe kasur', order = df_EDA['fasilitas_kamar_tipe kasur'].value_counts().index, palette = 'hls')
                for i, val in enumerate(df_EDA['fasilitas_kamar_tipe kasur'].value_counts()):
                    plt.text(s = val, x = val, y = i)
                plt.title('The Number of Units for each Bed Type')
                ax.set_xlabel('Count')
                ax.set_ylabel('Bed Type')
                st.pyplot(fig)
            elif bed_type_plot == 'Boxplot and Barplot (Bed Type vs Monthly Rent Price)':
                fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5.5))
                order = df_EDA.groupby('fasilitas_kamar_tipe kasur').agg({'harga_kost_per_bulan':'mean'}).sort_values(by = 'harga_kost_per_bulan', ascending = False).index
                sns.boxplot(data = df_EDA, x = 'fasilitas_kamar_tipe kasur', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[0])
                ax[0].set_title('Monthly Rent Price Distribution for each Bed Type', y = 1.05)
                ax[0].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[0].set_xlabel('Bed Type')
                ax[0].set_ylabel('Monthly Rent Price (IDR)')
                sns.barplot(data = df_EDA, x = 'fasilitas_kamar_tipe kasur', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[1])
                ax[1].set_title('Avg. Monthly Rent Price for each Bed Type', y = 1.05)
                ax[1].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[1].set_xlabel('Bed Type')
                ax[1].set_ylabel('Monthly Rent Price (IDR)')
                plt.tight_layout()
                st.pyplot(fig)
                
        # The Number of Bed(s) Visualization #        
        with tab5:
            st.subheader('The Number of Bed(s) Visualization')
            total_bed_plot = st.selectbox('Select Plot', ['Countplot (Data Distribution)', 'Boxplot and Barplot (The Number of Bed(s) vs Monthly Rent Price)'])
            if total_bed_plot == 'Countplot (Data Distribution)':
                fig, ax = plt.subplots()
                sns.countplot(data = df_EDA, y = 'fasilitas_kamar_jumlah kasur', order = df_EDA['fasilitas_kamar_jumlah kasur'].value_counts().index, palette = 'hls')
                for i, val in enumerate(df_EDA['fasilitas_kamar_jumlah kasur'].value_counts()):
                    plt.text(s = val, x = val, y = i)
                plt.title('The Number of Units for each Number of Bed(s)')
                ax.set_xlabel('Count')
                ax.set_ylabel('The Number of Bed(s)')
                st.pyplot(fig)
            elif total_bed_plot == 'Boxplot and Barplot (The Number of Bed(s) vs Monthly Rent Price)':
                fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5.5))
                order = df_EDA.groupby('fasilitas_kamar_jumlah kasur').agg({'harga_kost_per_bulan':'mean'}).sort_values(by = 'harga_kost_per_bulan', ascending = False).index
                sns.boxplot(data = df_EDA, x = 'fasilitas_kamar_jumlah kasur', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[0])
                ax[0].set_title('Monthly Rent Price Distribution for each Number of Bed(s)', y = 1.05)
                ax[0].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[0].set_xlabel('The Number of Bed(s)')
                ax[0].set_ylabel('Monthly Rent Price (IDR)')
                sns.barplot(data = df_EDA, x = 'fasilitas_kamar_jumlah kasur', y = 'harga_kost_per_bulan', palette = 'hls', order = order, ax = ax[1])
                ax[1].set_title('Avg. Monthly Rent Price for each Number of Bed(s)', y = 1.05)
                ax[1].set_xticklabels(order, rotation = 45, ha = 'right')
                ax[1].set_xlabel('The Number of Bed(s)')
                ax[1].set_ylabel('Monthly Rent Price (IDR)')
                plt.tight_layout()
                st.pyplot(fig)
                
        # Facility Visualization #
        with tab6:
            st.subheader('Facility Visualization')
            st.caption('Two types of facilities: Room Facilities and Building Facilities')
            fac = st.radio('Select which Facility', ['Building', 'Room'])
            if fac == 'Building':
                building_fac = st.selectbox('Select the Building Facility', list(building_facilities.keys()))
                fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 5.5))
                sns.countplot(data = df_EDA, x = building_facilities[building_fac], palette = ['#808080', '#DA0610'], ax = ax[0])
                ax[0].set_title('The Number of Units Distribution', y = 1.05)
                ax[0].set_ylabel('Count')
                ax[0].set(xlabel = None)
                sns.boxplot(data = df_EDA, x = building_facilities[building_fac], y = 'harga_kost_per_bulan', palette = ['#808080', '#DA0610'], ax = ax[1])
                ax[1].set_title('Monthly Rent Price Distribution', y = 1.05)
                ax[1].set_ylabel('Monthly Rent Price (IDR)')
                ax[1].set(xlabel = None)
                sns.barplot(data = df_EDA, x = building_facilities[building_fac], y = 'harga_kost_per_bulan', palette = ['#808080', '#DA0610'], ax = ax[2])
                ax[2].set_title('Avg. Monthly Rent Price', y = 1.05)
                ax[2].set_ylabel('Monthly Rent Price (IDR)')
                ax[2].set(xlabel = None)
                plt.suptitle(building_fac)
                plt.tight_layout()
                st.pyplot(fig)
            elif fac == 'Room':
                room_fac = st.selectbox('Select the Room Facility', list(room_facilities.keys()))
                fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 5.5))
                sns.countplot(data = df_EDA, x = room_facilities[room_fac], palette = ['#808080', '#DA0610'], ax = ax[0])
                ax[0].set_title('The Number of Units Distribution', y = 1.05)
                ax[0].set_ylabel('Count')
                ax[0].set(xlabel = None)
                sns.boxplot(data = df_EDA, x = room_facilities[room_fac], y = 'harga_kost_per_bulan', palette = ['#808080', '#DA0610'], ax = ax[1])
                ax[1].set_title('Monthly Rent Price Distribution', y = 1.05)
                ax[1].set_ylabel('Monthly Rent Price (IDR)')
                ax[1].set(xlabel = None)
                sns.barplot(data = df_EDA, x = room_facilities[room_fac], y = 'harga_kost_per_bulan', palette = ['#808080', '#DA0610'], ax = ax[2])
                ax[2].set_title('Avg. Monthly Rent Price', y = 1.05)
                ax[2].set_ylabel('Monthly Rent Price (IDR)')
                ax[2].set(xlabel = None) 
                plt.suptitle(room_fac)
                plt.tight_layout()
                st.pyplot(fig)
    
            

    ## siderbar: Monthly Rent Price Estimator ##
    elif selected_page == 'Monthly Rent Price Estimator':
        st.title('Boarding House Monthly Rent Price Estimator')
        st.markdown('Please input your desired boarding house unit and we will try to estimate the monthly rent price')
        
        # City
        st.subheader('Where is your boarding house unit?')
        city = st.selectbox('City', df_EDA['kota'].unique())
        
        # jakarta_check
        if 'Jakarta' in city:
            jakarta_check = 1
        else:
            jakarta_check = 0
        
        # Room Area
        st.subheader('How large is your boarding house room area?')
        df_area = df_EDA[df_EDA['kota'] == city][['luas_kost']]
        st.caption('For "%s" region, the boarding house area should be ranging from %.2f - %.2f m\u00b2 with an average of %.2f m\u00b2'%(city, df_area.min()[0], df_area.max()[0], df_area.mean()[0]))
        area = st.number_input('Room Area', min_value = df_area.min()[0], max_value = df_area.max()[0], help = 'Please input the value according to the suggested range above')
        
        # The Number of Bed(s)
        st.subheader('How many number of beds are there?')
        number_of_beds = st.selectbox('The Number of Bed(s)', ['0 bed', '1 bed', '2 bed'], index  = 1, help = 'Most boarding house units have only "1 bed"')
        
        # Bed Type
        st.subheader('What type of bed is there?')
        bed_type = st.selectbox('Bed Type', ['single bed', 'long bed', 'double bed', '≥queen bed'], help = 'Describing the size of the bed where "single bed" < "long bed" < "double bed" < "≥queen bed"')
        
        # jenis_kost_campur
        st.subheader('Is your boarding house for a particular gender only or mixed?')
        selected_value = st.radio('Type of Boarding House', ['male and female living together', 'male only / female only'])
        if selected_value == 'male and female living together':
            jenis_kost_campur = 1
        else:
            jenis_kost_campur = 0
            
        # Other buldings that are near the boarding house
        st.subheader('Are there other buildings around your boarding house?')
        mall = st.checkbox('Near Mall')
        supermarket = st.checkbox('Near Supermarket')
        healthcare = st.checkbox('Near Healtcare Access')
        school = st.checkbox('Near School')
        transportation = st.checkbox('Near Transportation Access')
        
        # Building facility
        st.subheader('Which building facilities does your boarding house have?')
        Dining_Room = st.checkbox('Dining Room')
        Parking_Area = st.checkbox('Parking Area')
        CCTV = st.checkbox('CCTV')
        Cleaning_Service = st.checkbox('Cleaning Service')
        Kitchen = st.checkbox('Kitchen')
        Laundry = st.checkbox('Laundry')
        Washing_Machine = st.checkbox('Washing Machine')
        Microwave = st.checkbox('Microwave')
        Living_Room = st.checkbox('Living Room')
        Clothes_Drying_Area = st.checkbox('Clothes Drying Area')
        Wifi = st.checkbox('Wifi')
        Stove = st.checkbox('Stove')
        Fridge = st.checkbox('Fridge')
        Water_Dispenser = st.checkbox('Water Dispenser')
        
        # Room facility
        st.subheader('Which room facilities does your boarding house have?')
        Water_Heater = st.checkbox('Water Heater')
        Shower = st.checkbox('Shower')
        Sink = st.checkbox('Sink')
        AC = st.checkbox('AC')
        Chair = st.checkbox('Chair')
        Wardrobe = st.checkbox('Wardrobe')
        Table = st.checkbox('Table')
        Bed_Sheet = st.checkbox('Bed Sheet')
        TV = st.checkbox('TV')
        TV_Cable = st.checkbox('TV Cable')
        Window = st.checkbox('Window to the Outside')
        Electricity = st.checkbox('Including Electricity')
        Bathroom = st.checkbox('In-room Bathroom')
        
        # Retrieve all the input user data
        df_input = pd.DataFrame([{
            'kota':city, 'jakarta_check':jakarta_check, 'luas_kost':area, 'fasilitas_kamar_jumlah kasur':number_of_beds, 'fasilitas_kamar_tipe kasur':bed_type, 'jenis_kost_campur':jenis_kost_campur,
            'sekitar_gedung_mall':mall, 'sekitar_gedung_supermarket':supermarket, 'sekitar_gedung_akses kesehatan':healthcare, 'sekitar_gedung_sekolah':school, 'sekitar_gedung_akses transportasi':transportation,
            'area makan':Dining_Room, 'area parkir':Parking_Area, 'cctv':CCTV, 'cleaning service': Cleaning_Service, 'dapur':Kitchen, 'laundry':Laundry, 'mesin cuci':Washing_Machine,
            'microwave':Microwave, 'ruang santai / ruang tamu':Living_Room, 'tempat jemur pakaian':Clothes_Drying_Area, 'wifi':Wifi, 'kompor':Stove, 'kulkas':Fridge, 'dispenser':Water_Dispenser,
            'fasilitas_kamar_pemanas air':Water_Heater, 'fasilitas_kamar_shower':Shower, 'fasilitas_kamar_wastafel':Sink, 'fasilitas_kamar_ac':AC, 'fasilitas_kamar_kursi':Chair,
            'fasilitas_kamar_lemari':Wardrobe, 'fasilitas_kamar_meja':Table, 'fasilitas_kamar_sprei':Bed_Sheet, 'fasilitas_kamar_tv':TV, 'fasilitas_kamar_tv kabel':TV_Cable, 
            'fasilitas_kamar_jendela kearah luar':Window, 'fasilitas_kamar_termasuk listrik':Electricity, 'fasilitas_kamar_kamar mandi dalam':Bathroom
            }])
        room_fac_cols = ['fasilitas_kamar_pemanas air', 'fasilitas_kamar_shower', 'fasilitas_kamar_wastafel', 'fasilitas_kamar_ac', 'fasilitas_kamar_kursi', 'fasilitas_kamar_lemari', 'fasilitas_kamar_meja', 'fasilitas_kamar_sprei', 'fasilitas_kamar_tv', 'fasilitas_kamar_tv kabel', 'fasilitas_kamar_jendela kearah luar', 'fasilitas_kamar_termasuk listrik', 'fasilitas_kamar_kamar mandi dalam']
        building_fac_cols = ['area makan', 'area parkir', 'cctv', 'cleaning service', 'dapur', 'laundry', 'mesin cuci', 'microwave', 'ruang santai / ruang tamu', 'tempat jemur pakaian', 'wifi', 'kompor', 'kulkas', 'dispenser', 'jenis_kost_campur', 'jakarta_check']
        df_input['total_fasilitas_gedung'] = df_input[building_fac_cols].sum(axis = 1)
        df_input['total_fasilitas_kamar'] = df_input[room_fac_cols].sum(axis = 1)
        df_input['total_fasilitas'] = df_input['total_fasilitas_kamar'] + df_input['total_fasilitas_gedung']

        binary_cols = ['area makan', 'area parkir', 'cctv', 'cleaning service', 'dapur', 'laundry', 'mesin cuci', 'microwave', 'ruang santai / ruang tamu', 'tempat jemur pakaian', 'wifi', 'sekitar_gedung_mall', 'sekitar_gedung_supermarket', 'sekitar_gedung_akses kesehatan', 'sekitar_gedung_sekolah', 'sekitar_gedung_akses transportasi', 'fasilitas_kamar_pemanas air', 'fasilitas_kamar_shower', 'fasilitas_kamar_wastafel', 'fasilitas_kamar_ac', 'fasilitas_kamar_kursi', 'fasilitas_kamar_lemari', 'fasilitas_kamar_meja', 'fasilitas_kamar_sprei', 'fasilitas_kamar_tv', 'fasilitas_kamar_tv kabel', 'kompor', 'kulkas', 'dispenser', 'fasilitas_kamar_jendela kearah luar', 'jenis_kost_campur', 'fasilitas_kamar_termasuk listrik', 'fasilitas_kamar_kamar mandi dalam', 'jakarta_check']
        pl_features = get_feature_pipeline(numerical = ['luas_kost', 'total_fasilitas_kamar', 'total_fasilitas_gedung', 'total_fasilitas'], 
                                           nominal = ['kota'], 
                                           ordinal = ['fasilitas_kamar_jumlah kasur', 'fasilitas_kamar_tipe kasur'], 
                                           binary = binary_cols, 
                                           algorithm = 'tree-based', 
                                           ordinal_category = [['0 bed', '1 bed', '2 bed'],
                                                               ['single bed', 'long bed', 'double bed', '≥queen bed']])
        
        for col in [col for col in binary_cols if col not in ['jakarta_check', 'jenis_kost_campur']]:
            df_input[col] = df_input[col].astype('int64')
        pl_features.fit(df_EDA)
        df_input = pd.DataFrame(pl_features.transform(df_input).toarray(), columns = features)
        
        # Show inputted user data
        st.subheader('Do you wanna see your input data?')
        if st.checkbox('Yes', False):
            st.write(df_input.rename(new_mapper, axis = 1))
            st.write(df_input.rename(new_mapper, axis = 1).dtypes)
        
        # Estimate the price
        if st.button('Estimate the monthly rent price'):
            price = LGBM_final.predict(df_input)[0]   
            price_001 = LGBM_001.predict(df_input)[0] 
            price_005 = LGBM_005.predict(df_input)[0]
            price_095 = LGBM_095.predict(df_input)[0]
            price_099 = LGBM_099.predict(df_input)[0]
            st.success('Estimated monthly rent price: {} IDR'.format(format(round(price), ',')))
            st.caption('Estimated 95% prediction interval for this unit: {} - {} IDR'.format(format(round(price_005), ','), format(round(price_095), ',')))
            st.caption('Estimated 99% prediction interval for this unit: {} - {} IDR'.format(format(round(price_001), ','), format(round(price_099), ',')))
        
if __name__ == '__main__':
    main()
