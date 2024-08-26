import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import requests
from streamlit_lottie import st_lottie
import json
import lzma
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import joblib

# Set up page configuration for Streamlit
st.set_page_config(page_title="Copper Modelling", page_icon='analytics.ico', layout="wide")

# Define the option class with encoding values
class Option:
    country_values_enc = ['Select an option...', '28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
    status_values = ['Select an option...', 'Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_encoded = {'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4, 'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}
    item_type_values = ['Select an option...', 'W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    item_type_encoded = {'W': 5.0, 'WI': 6.0, 'S': 3.0, 'Others': 1.0, 'PL': 2.0, 'IPL': 0.0, 'SLAWR': 4.0}
    application_values = ['Select an option...', 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0,
                          41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    product_ref_values = ['Select an option...', 611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                          164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642,
                          1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026,
                          1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://raw.githubusercontent.com/YABASEIMMANUEL/copper_modeling/main/business-analysis.json")

# Layout: left for menu and animation, right for content
left, right = st.columns([1, 3], gap="medium")

# Initialize session state for form inputs
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Function to reset form
def reset_form():
    st.session_state.reset = True

# Function to clear the reset flag
def clear_reset_flag():
    st.session_state.reset = False

# Left Column: Lottie Animation and Option Menu
with left:
    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="copper_animation")
    else:
        st.warning("Lottie animation could not be loaded. Please check the URL or your internet connection.")
    
    selected = option_menu("Menu",
                           options=["Home", "Predictive Analytics", "Observations"],
                           icons=["house", "info-circle", "bar-chart", "lightbulb"],
                           default_index=0,
                           orientation="vertical",
                           menu_icon="cast",
                           styles={
                               "container": {"padding": "5px", "background-color": "#C4C3FB"},
                               "icon": {"color": "#5751BB", "font-size": "25px"},
                               "nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#5751BB"},
                           })
    
# Right Column: Content based on selected menu option
with right:
    if selected == 'Home':
        # Home Tab Content
        title_text = '''<h3 style='font-size: 35px;color:#5751BB;text-align: center;'>What is Copper?</h3>'''
        st.markdown(title_text, unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1.5], gap="large")
        with col1:
            st.markdown('''<h5 style='color:#44454A;font-size:20px'> Copper is a versatile and highly conductive metal that has been used by humans for thousands of years. It has a distinctive reddish-brown color and is known for its excellent electrical and thermal conductivity, making it a key material in electrical wiring, electronics, and heat exchangers. Copper is also resistant to corrosion and has antimicrobial properties, which contribute to its use in plumbing, medical devices, and coins. Additionally, copper is essential in the production of alloys such as bronze and brass, and it plays a critical role in various industries, including construction, transportation, and renewable energy.''', unsafe_allow_html=True)
            st.write('')
            st.markdown('<a href="https://en.wikipedia.org/wiki/Copper" target="_blank" style="color: #5751BB;">Copper - Wiki</a>', unsafe_allow_html=True)
            st.write('')
        with col2:
            st.image('https://www.mining.com/wp-content/uploads/2023/12/AdobeStock_648674620-1024x683.jpeg', caption="Copper wires - google image", width=330)

        title_text = '''<h3 style='font-size: 35px;color:#5751BB;text-align: center;'>India's Copper Production and Consumption Trends</h3>'''
        st.markdown(title_text, unsafe_allow_html=True)
        left, right = st.columns([2, 2], gap="medium")
        with right:
            st.write('')
            st.markdown('''<h5 style='font-size:20px;color:#44454A'> From 2017 to 2024, India has experienced a significant shift in its copper trade dynamics. In 2017-18, India was a net exporter, with copper exports at around 335,000 tonnes, driven by robust domestic production of approximately 850,000 tonnes, and consumption at 670,000 tonnes. However, the closure of major smelting operations in 2018 led to a sharp decline in production, causing India to become a net importer from 2018-19 onward. By 2023-24, India's copper imports have surged to meet the growing domestic consumption, which is estimated to reach around 1,000,000 tonnes, while exports have drastically reduced, and production has stagnated at around 450,000 tonnes, leading to a substantial dependency on imports to bridge the gap. ''', unsafe_allow_html=True)
        with left:
            st.image('https://static.theprint.in/wp-content/uploads/2023/01/ANI-20230111130414.jpg', width=425)
        st.write('')
        title_text = '''<h3 style='font-size: 35px;color:#5751BB;text-align: center;'>What is Copper used for?</h3>'''
        st.markdown(title_text, unsafe_allow_html=True)
        left, right = st.columns([1.5, 1], gap="large")
        with right:
            st.image('https://sterlitecopper.com/blog/wp-content/uploads/2018/07/01.png', width=300)
        with left:
            st.markdown('''<h5 style='font-size:20px;color:#44454A'> Copper is utilized across nearly all industries. The image illustrates the percentage of copper consumption in various sectors in India.<br><br>
                        Currently, copper is employed in a wide range of areas, including: <br>''', unsafe_allow_html=True)
            st.markdown('''
                    - **Construction and infrastructure development** 
                    - **Power generation and electrical transmission** 
                    - **Production of industrial equipment** 
                    - **Manufacturing of electronic devices** 
                    - **Automotive and transportation industries** 
                    <br>''', unsafe_allow_html=True)
        st.write('')
        st.write('')
        st.markdown('''<h4 style='font-size: 30px;color:#5751BB;text-align: left;'>Copper's Versatile Uses</h4>''', unsafe_allow_html=True)
        st.markdown('''<h5 style='font-size:18px;color:grey'> Click on the expanders to learn more.''', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
        with c1:
            with st.container():
                with st.expander(':violet[**Electrical and Electronics**]'):
                    st.markdown('''Copper's high electrical conductivity makes it indispensable in the production of wiring, circuit boards, and connectors. It ensures efficient power transmission in electronics, appliances, and telecommunication systems, playing a crucial role in powering everyday devices and advanced technology.''', unsafe_allow_html=True)
        with c2:
            with st.container():
                with st.expander(':violet[**Construction**]'):
                    st.markdown('''Copper is widely used in construction for plumbing, electrical wiring, and roofing. Its resistance to corrosion and ability to withstand extreme weather conditions make it a preferred material for long-lasting infrastructure, ensuring the reliability and safety of buildings.''', unsafe_allow_html=True)

        with c3:
            with st.container():
                with st.expander(':violet[**Energy Sector**]'):
                    st.markdown(''' Copper is essential in the generation and distribution of electricity. It is used in power cables, transformers, and renewable energy systems like wind turbines and solar panels. Copper’s ability to efficiently conduct electricity makes it critical for minimizing energy loss in power grids and enhancing the efficiency of energy production.''', unsafe_allow_html=True)

        with c4: 
            with st.container():
                with st.expander(':violet[**Automotive Industry**]'):
                    st.markdown('''In the automotive sector, copper is used in vehicle wiring, electric motors, and battery systems, particularly in electric and hybrid vehicles. Its thermal and electrical conductivity contributes to the performance and safety of modern vehicles, supporting the development of energy-efficient and sustainable transportation.''', unsafe_allow_html=True)

        st.write('')  
        title_text = '''<h3 style='font-size: 30px;color:#5751BB;text-align: left;'>Video references on Copper</h3>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.write('')  
        col1, col2, col3 = st.columns(3)

        with col1:
            st.video('https://www.youtube.com/watch?v=gqmkiPPIsUQ&pp=ygUNIGFib3V0IGNvcHBlcg%3D%3D')
        with col2:
            st.video('https://www.youtube.com/watch?v=AgRYHT6WFV0&pp=ygUTIGNvcHBlciBpbiBpbmR1c3RyeQ%3D%3D')
        with col3:
            st.video('https://youtu.be/g8Nar1s5UgM?si=ALCKDdRJPgQunIhi')

    
    elif selected == 'Predictive Analytics':
        
        title_text = '''<h2 style='font-size: 28px;text-align: center;color:#5751BB;'>Copper Selling Price and Status Prediction</h2>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.write('')
        
        # Set up option menu for selling price and status menu
        select = option_menu('', options=["SELLING PRICE", "STATUS"],
                                        icons=["cash", "toggles"],
                                        orientation='horizontal',
                                        styles={
                               "container": {"padding": "5px", "background-color": "#C4C3FB"},
                               "icon": {"color": "#5751BB", "font-size": "25px"},
                               "nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#5751BB"},
                           })


        if select == 'SELLING PRICE':
            st.markdown('##### ***<span style="color:#5751BB">Fill all the fields and Press the below button to view the :red[predicted price] of copper</span>***', unsafe_allow_html=True)
            st.write('')
    
            # Create form to get the user input 
            with st.form('Price Prediction'):
                col1, col2 = st.columns(2)
                with col1:
                    item_date = st.date_input(label='Item Date', key='item_date' if not st.session_state.reset else 'new_item_date')
                    country = st.selectbox(label='Country', options=Option.country_values_enc, key='country' if not st.session_state.reset else 'new_country')
                    item_type = st.selectbox(label='Item Type', options=Option.item_type_values, key='item_type' if not st.session_state.reset else 'new_item_type')
                    customer = st.number_input('Customer ID', min_value=10000, key='customer' if not st.session_state.reset else 'new_customer')
                    thickness = st.number_input(label='Thickness', min_value=0.1, key='thickness' if not st.session_state.reset else 'new_thickness')
                    quantity = st.number_input(label='Quantity', min_value=0.1, key='quantity' if not st.session_state.reset else 'new_quantity')
                    
                with col2:
                    delivery_date = st.date_input(label='Delivery Date', key='delivery_date' if not st.session_state.reset else 'new_delivery_date')
                    status = st.selectbox(label='Status', options=Option.status_values, key='status' if not st.session_state.reset else 'new_status')
                    product_ref = st.selectbox(label='Product Ref', options=Option.product_ref_values, key='product_ref' if not st.session_state.reset else 'new_product_ref')
                    application = st.selectbox(label='Application', options=Option.application_values, key='application' if not st.session_state.reset else 'new_application')
                    width = st.number_input(label='Width', min_value=1.0, key='width' if not st.session_state.reset else 'new_width')
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                # Add submit and clear buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button('PREDICT PRICE')
                with col2:
                    clear_button = st.form_submit_button('CLEAR FORM', on_click=reset_form)

            if submit_button:
                clear_reset_flag()  # Reset the reset state after submission
                if any(val == 'Select an option...' for val in [country, item_type, application, product_ref, status]):
                    st.error("Please fill in all required fields.")
                else:
                    try:
                        # Attempt to load the model using LZMA compression
                        with lzma.open('RandomForestRegressor_compressed.pkl', 'rb') as files:
                            predict_model = pickle.load(files)
                    except lzma.LZMAError:
                        # Fallback to standard pickle loading if LZMA fails
                        with open('RandomForestRegressor_compressed.pkl', 'rb') as file:
                            predict_model = pickle.load(file)

                    # Check if the status is in the dictionary before encoding
                    if status in Option.status_encoded:
                        status = Option.status_encoded[status]
                    else:
                        st.error(f"Status '{status}' is not recognized. Please select a valid status.")
                        st.stop()

                    item_type = Option.item_type_encoded[item_type]

                    delivery_time_taken = abs((item_date - delivery_date).days)

                    quantity_log = np.log(quantity)
                    thickness_log = np.log(thickness)

                    user_data = np.array([[customer, country, status, item_type, application, width, product_ref,
                                        delivery_time_taken, quantity_log, thickness_log ]])
                    
                    pred = predict_model.predict(user_data)

                    selling_price = np.exp(pred[0])

                    st.subheader(f":green[Predicted Selling Price :] {selling_price:.2f}") 

        if select == 'STATUS':
            st.markdown('##### ***<span style="color:#5751BB">Fill all the fields and Press the below button to view the status :green[WON] / :red[LOST] of copper in the desired time range</span>***', unsafe_allow_html=True)
            st.write('')

            with st.form('Status Classifier'):
                col1, col2 = st.columns(2)

                with col1:
                    item_date = st.date_input(label='Item Date', key='status_item_date' if not st.session_state.reset else 'new_status_item_date')
                    country = st.selectbox(label='Country', options=Option.country_values_enc, key='status_country' if not st.session_state.reset else 'new_status_country')
                    item_type = st.selectbox(label='Item Type', options=Option.item_type_values, key='status_item_type' if not st.session_state.reset else 'new_status_item_type')
                    thickness = st.number_input(label='Thickness', min_value=0.1, key='status_thickness' if not st.session_state.reset else 'new_status_thickness')
                    application = st.selectbox(label='Application', options=Option.application_values, key='status_application' if not st.session_state.reset else 'new_status_application')
                    product_ref = st.selectbox(label='Product Ref', options=Option.product_ref_values, key='status_product_ref' if not st.session_state.reset else 'new_status_product_ref')

                with col2:
                    delivery_date = st.date_input(label='Delivery Date', key='status_delivery_date' if not st.session_state.reset else 'new_status_delivery_date')
                    customer = st.number_input('Customer ID', min_value=10000, key='status_customer' if not st.session_state.reset else 'new_status_customer')
                    quantity = st.number_input(label='Quantity', min_value=0.1, key='status_quantity' if not st.session_state.reset else 'new_status_quantity')
                    width = st.number_input(label='Width', min_value=1.0, key='status_width' if not st.session_state.reset else 'new_status_width')
                    selling_price = st.number_input(label='Selling Price', min_value=0.1, key='status_selling_price' if not st.session_state.reset else 'new_status_selling_price')
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                # Add submit and clear buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button('PREDICT STATUS')
                with col2:
                    clear_button = st.form_submit_button('CLEAR FORM', on_click=reset_form)

            if submit_button:
                clear_reset_flag()  # Reset the reset state after submission
                if any(val == 'Select an option...' for val in [country, item_type, application, product_ref]):
                    st.error("Please fill in all required fields.")
                else:
                    item_type = Option.item_type_encoded[item_type]
                    delivery_time_taken = abs((item_date - delivery_date).days)
                    quantity_log = np.log(quantity)
                    thickness_log = np.log(thickness)
                    selling_price_log = np.log(selling_price)

                    user_data = np.array([[customer, country, item_type, application, width, product_ref,
                                        delivery_time_taken, quantity_log, thickness_log, selling_price_log]])

                    try:
                        # Load the partitioned model parts using LZMA
                        with lzma.open('ExtraTreesClassifier_part1.pkl', 'rb') as file1:
                            part1 = pickle.load(file1)

                        with lzma.open('ExtraTreesClassifier_part2.pkl', 'rb') as file2:
                            part2 = pickle.load(file2)

                        # Combine the parts to reconstruct the model
                        model = ExtraTreesClassifier(n_estimators=len(part1) + len(part2))
                        model.estimators_ = part1 + part2

                        # Manually set the necessary attributes
                        model.n_classes_ = 2  # Number of classes (for binary classification)
                        model.classes_ = np.array([0, 1])  # The class labels
                        model.n_features_in_ = len(user_data[0])  # Set the number of features
                        model.n_outputs_ = 1  # Single target output

                    except lzma.LZMAError:
                        st.error("Failed to load the model due to LZMA compression error.")
                        st.stop()

                    status = model.predict(user_data)

                    if status == 1:
                        st.subheader(f":green[Status of the copper : ] Won")
                    else:
                        st.subheader(f":red[Status of the copper :] Lost")

    elif selected == 'Observations':
        # Inferences Tab Content
        title_text = '''<h2 style='font-size: 35px;text-align: center;color:#5751BB;'>Observations</h2>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.markdown('''In the previous section of Predictive Analytics, we analyzed the price and status predictions based on user inputs. While we obtained results for both predictions, there are notable concerns, particularly with the status prediction.<br>''', 
                    unsafe_allow_html=True)
        
        st.markdown('''<h4 style='color:#5751BB;'>Selling Price Prediction:</h4>''', unsafe_allow_html=True)
        st.markdown('''
                    - **Data Inconsistency:** Despite having a large dataset with **181,673 entries**, the data lacks consistency, especially in some variables 
                    that are :red[noisy] and scattered, requiring extensive cleaning.
                    - **High Variance in Selling Price:** The selling price field exhibited high variance, including negative values, which complicated the modeling process.
                    - **Use of Tree-Based Regressors:** Given the skewness and noise in the data, we opted for :red[tree-based regressors] over :red[Linear regression], 
                    as they can better handle data irregularities.
                    - **Model Performance:** The :red[RandomForestRegressor] performed well, achieving an accuracy of about **92%**, demonstrating its effectiveness 
                    in managing the complexities of the dataset.
                    <br>''', unsafe_allow_html=True)
        
        st.markdown('''<h4 style='color:#5751BB;'>Status Prediction:</h4>''', unsafe_allow_html=True)
        st.markdown('''
                    - **Focus on Two Statuses:** We concentrated on predicting only two statuses, :red[Lost] and :green[Won], to simplify the model and improve accuracy.
                    - **Data Balancing with SMOTE:** **Oversampling** with SMOTE was used to balance the status classes, addressing the issue of class imbalance.
                    - **Ensemble Modeling:** We utilized an ensemble of models including Random Forest, Extra Trees, and XGB Classifier. The :red[Extra Trees Classifier] 
                    narrowly outperformed Random Forest, achieving a high accuracy of roughly **98%**.
                    - **Potential Overfitting:** The high accuracy might indicate overfitting, as the model predominantly predicts the :green[Won] status.
                    - **Training Data Limitations:** The model's focus on predicting :green[Won] status highlights the need for a more balanced and extensive training dataset 
                    to improve the prediction accuracy for the :red[Lost] status.
                    <br>''', unsafe_allow_html=True)


# Footer with icons
footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #C4C3FB;
        color: black;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 100;
    }
    .footer a {
        color: black;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #007BFF;
    }
    .footer i {
        margin-right: 5px;
    }
    </style>
    <div class="footer">
        <p>© 2024 Copper Modelling Project | Created by YABASE IMMANUEL</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
