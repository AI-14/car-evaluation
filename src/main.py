import pickle
import streamlit as st
import numpy as np

def init_info_content():
    # Method for initializing the main content such as instructions, title etc.

    title = '''
        <div style="background-color:#3f93d9;padding:10px;border-radius:20px">
            <h2 style="color:white;text-align:center;font-size:40px">
                Car Evaluation - A machine learning prediction model
            </h2>
        </div>
    '''

    info = '''
    ***Instructions:***
    - Select your choices from the dropdown boxes.
    - Hit the predict button and get the results.
    
    #### Developer: Amaan Izhar [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/AI-14)
    '''

    design = '''
        <div style="background-color:#6bd6bf;padding:10px;border-radius:20px">
        </div>
    '''

    st.markdown(title, unsafe_allow_html=True)
    st.write('\n\n')
    st.markdown(info)
    st.write('\n\n')
    st.markdown(design, unsafe_allow_html=True)
    st.write('\n\n')

def load_model():
    # Method that loads the pickle file.

    filename = 'src//model//careval_finalized_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    return model

def make_prediction():
    # Method for making prediction.

    buying_choice = st.selectbox('Buying', ['vhigh', 'high', 'med', 'low'])
    maint_choice = st.selectbox('Maintenance', ['vhigh', 'high', 'med', 'low'])
    doors_choice = st.selectbox('Number of doors', [2, 3, 4, '5-more'])
    persons_choice = st.selectbox('Number of persons', [2, 4, 'more'])
    lug_boot_choice = st.selectbox('Luggage Boot', ['big',  'med', 'small'])
    safety_choice = st.selectbox('Safety Measure', ['high', 'med', 'low'])

    buying_dict = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    maint_dict = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    doors_dict = {2: 2, 3: 3, 4: 4, '5-more': 5}
    persons_dict = {2: 2, 4: 4, 'more': 5}
    lug_boot_dict = {'big': 2, 'med': 1, 'small': 0}
    safety_dict = {'high': 2, 'med': 1, 'low': 0}

    buying = buying_dict[buying_choice]
    maint = maint_dict[maint_choice]
    doors = doors_dict[doors_choice]
    persons = persons_dict[persons_choice]
    lug_boot = lug_boot_dict[lug_boot_choice]
    safety = safety_dict[safety_choice]

    input_features = np.array([buying, maint, doors, persons, lug_boot, safety])
    pred_dict = {0: 'Acceptable', 1: 'Good', 2: 'Unacceptable', 3: 'Very Good'}

    car_ml_model = load_model()

    if st.button('Predict'):
        prediction = car_ml_model.predict(input_features.reshape(1,-1))
        st.success(pred_dict[prediction[0]])

def main():
    # Method to handle the main functionalities.

    init_info_content()
    make_prediction()

# Application starts here.
if __name__ == '__main__':
    main()