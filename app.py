import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime,date
import pickle


def get_age(open_date):
    difference = date(2015,12,31) - open_date
    return difference.days


def get_results(input_data):
    df = pd.DataFrame(input_data)
    df['Year'] = pd.DatetimeIndex(df['Open Date']).year  # year
    df['Month'] = pd.DatetimeIndex(df['Open Date']).month  # month
    df['Day_of_week'] = pd.DatetimeIndex(
        df['Open Date']).dayofweek  # dayofweek
    df['Weekend'] = df.apply(
        lambda row: 1 if row['Day_of_week'] >= 5 else 0, axis=1)  # weekend
    df['Age_of_Restaurent'] = df.apply(
        lambda row: get_age(row["Open Date"]), axis=1)
    # one hot encoder for citygroup and type-train
    df = pd.get_dummies(df, columns=['City Group', 'Type'])
    # drop open date
    df = df.drop("Open Date", axis=1)
    with open("model/city_label_encoder.pickle", "rb") as f:
        city_label_encoder = pickle.load(f)
        df['City'] = city_label_encoder.transform(df['City'])

    xgb_result = 0
    rf_result = 0
    with open("model/xgbr_best.pickle", "rb")as f:
        xgbr_best = pickle.load(f)
        model_cols = list(xgbr_best.get_booster().feature_names)
        missing_cols = set(model_cols)-set(list(df.columns))
        missing_df = pd.DataFrame(columns=missing_cols)
        df = pd.concat([df,missing_df],axis=1)
        df = df.fillna(0)
        print(df.columns)
        xgb_result = xgbr_best.predict(df)

    with open("model/rf_best.pickle", "rb")as f:
        rf_best = pickle.load(f)
        rf_result = rf_best.predict(df)

    print("XGB Result", xgb_result)
    print("Randomforest Result", rf_result)
    return xgb_result,rf_result

if __name__ == '__main__':
    st.title("Annual Restaurant Sales")
    st.markdown("Open Date")
    OpenDate = st.date_input(
        "When restaurent started",
        value=date(2015, 12, 9),
        min_value=date(1994, 1, 1),
        max_value=date(2015, 12, 31))
    st.markdown("City")
    City = st.selectbox(
        'what city',
        ('İstanbul', 'Ankara,İzmir', 'Samsun', 'Bursa', 'Sakarya', 'Antalya', 'Diyarbakır', 'Adana', 'Eskişehir',
         'Kayseri', 'Tekirdağ', 'Konya', 'Aydın', 'Trabzon', 'Muğla', 'Osmaniye', 'Balıkesir', 'Elazığ', 'Karabük',
         'Şanlıurfa', 'Kırklareli', 'Kocaeli', 'Edirne', 'Bolu', 'Tokat', 'Gaziantep', 'Isparta', 'Denizli', 'Uşak',
         'Amasya', 'Afyonkarahisar', 'Kastamonu', 'Kütahya'))

    st.markdown("CityGroup")
    CityGroup = st.radio(
        "What citygroup",
        ('Big Cities', 'Other'))
    st.markdown("Type")
    Type = st.radio(
        "What type",
        ('FC', 'IL', 'DT', 'MB'))
    st.markdown("#P values")
    st.markdown("""**P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23,P24,P25,26,P27,P28,P29,P30,P31,P32,P33,P34,P35,P36 P37 -
     There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
     **""")
    P1 = st.slider('P1', 0.0, 1.0, step=0.01)
    st.write("p1 ranges-", P1)

    P2 = st.slider('P2', 0.0, 1.0, step=0.01)
    st.write("p2 ranges-", P2)

    P3 = st.slider('P3', 0.0, 1.0, step=0.01)
    st.write("p3 ranges-", P3)

    P4 = st.slider('P4', 0.0, 1.0, step=0.01)
    st.write("p4 ranges-", P4)

    P5 = st.slider('P5', 0.0, 1.0, step=0.01)
    st.write("p5 ranges-", P5)

    P6 = st.slider('P6', 0.0, 1.0, step=0.01)
    st.write("p6 ranges-", P6)

    P7 = st.slider('P7', 0.0, 1.0, step=0.01)
    st.write("p7 ranges-", P7)

    P8 = st.slider('P8', 0.0, 1.0, step=0.01)
    st.write("p8 ranges-", P8)

    P9 = st.slider('P9', 0.0, 1.0, step=0.01)
    st.write("p9 ranges-", P9)

    P10 = st.slider('P10', 0.0, 1.0, step=0.01)
    st.write("p10 ranges-", P10)

    P11 = st.slider('P11', 0.0, 1.0, step=0.01)
    st.write("p11 ranges-", P11)

    P12 = st.slider('P12', 0.0, 1.0, step=0.01)
    st.write("p12 ranges-", P12)

    P13 = st.slider('P13', 0.0, 1.0, step=0.01)
    st.write("p13 ranges-", P13)

    P14 = st.slider('P14', 0.0, 1.0, step=0.01)
    st.write("p14 ranges-", P14)

    P15 = st.slider('P15', 0.0, 1.0, step=0.01)
    st.write("p15 ranges-", P15)

    P16 = st.slider('P16', 0.0, 1.0, step=0.01)
    st.write("p16 ranges-", P16)

    P17 = st.slider('P17', 0.0, 1.0, step=0.01)
    st.write("p17 ranges-", P17)

    P18 = st.slider('P18', 0.0, 1.0, step=0.01)
    st.write("p18 ranges-", P18)

    P19 = st.slider('P19', 0.0, 1.0, step=0.01)
    st.write("p19 ranges-", P19)

    P20 = st.slider('P20', 0.0, 1.0, step=0.01)
    st.write("p20 ranges-", P20)

    P21 = st.slider('P21', 0.0, 1.0, step=0.01)
    st.write("p21 ranges-", P21)

    P22 = st.slider('P22', 0.0, 1.0, step=0.01)
    st.write("p22 ranges-", P22)

    P23 = st.slider('P23', 0.0, 1.0, step=0.01)
    st.write("p23 ranges-", P23)

    P24 = st.slider('P24', 0.0, 1.0, step=0.01)
    st.write("p24 ranges-", P24)

    P25 = st.slider('P25', 0.0, 1.0, step=0.01)
    st.write("p25 ranges-", P25)

    P26 = st.slider('P26', 0.0, 1.0, step=0.01)
    st.write("p26 ranges-", P26)

    P27 = st.slider('P27', 0.0, 1.0, step=0.01)
    st.write("p27 ranges-", P27)

    P28 = st.slider('P28', 0.0, 1.0, step=0.01)
    st.write("p28 ranges-", P28)

    P29 = st.slider('P29', 0.0, 1.0, step=0.01)
    st.write("p29 ranges-", P29)

    P30 = st.slider('P30', 0.0, 1.0, step=0.01)
    st.write("p30 ranges-", P30)

    P31 = st.slider('P31', 0.0, 1.0, step=0.01)
    st.write("p31 ranges-", P31)

    P32 = st.slider('P32', 0.0, 1.0, step=0.01)
    st.write("p32 ranges-", P32)

    P33 = st.slider('P33', 0.0, 1.0, step=0.01)
    st.write("p33 ranges-", P33)

    P34 = st.slider('P34', 0.0, 1.0, step=0.01)
    st.write("p34 ranges-", P34)

    P35 = st.slider('P35', 0.0, 1.0, step=0.01)
    st.write("p35 ranges-", P35)

    P36 = st.slider('P36', 0.0, 1.0, step=0.01)
    st.write("p36 ranges-", P36)

    P37 = st.slider('P37', 0.0, 1.0, step=0.01)
    st.write("p37 ranges-", P37)

    if st.button("predict"):
        input_data = [{'Id': 0,
                      'Open Date': OpenDate,
                      'City': City,
                      'City Group': CityGroup,
                      'Type': Type,
                      'P1': P1,
                      'P2': P2,
                      'P3': P3,
                      'P4': P4,
                      'P5': P5,
                      'P6': P6,
                      'P7': P7,
                      'P8': P8,
                      'P9': P9,
                      'P10': P10,
                      'P11': P11,
                      'P12': P12,
                      'P13': P13,
                      'P14': P14,
                      'P15': P15,
                      'P16': P16,
                      'P17': P17,
                      'P18': P18,
                      'P19': P19,
                      'P20': P20,
                      'P21': P21,
                      'P22': P22,
                      'P23': P23,
                      'P24': P24,
                      'P25': P25,
                      'P26': P26,
                      'P27': P27,
                      'P28': P28,
                      'P29': P29,
                      'P30': P30,
                      'P31': P31,
                      'P32': P32,
                      'P33': P33,
                      'P34': P34,
                      'P35': P35,
                      'P36': P36,
                      'P37': P37}]
        print(input_data)
        xgb_res,rf_res = get_results(input_data)
        st.text(f"XG Boost predicted_revenue {xgb_res} Millions Annually")
        st.text(f"Random Forest predicted_revenue {rf_res} Millions Annually")
