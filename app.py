import streamlit as st
import pandas as pd
import requests
import json


# Flask API URL
API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(page_title="Price Prediction", page_icon="📊", layout="wide")
st.markdown("""
<style>
    [data-testid=stHeader] {
        display:none;
    }
    .block-container{
        padding-top: 20px;
    }
    img {
        width:300px;
        height:200px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("House Price Prediction")

st.write("Select housing features...")

inputs = st.columns(3)

with inputs[0]:
    with st.expander(label="Lot Information"):
        st.image("app_src/frontage.png")
        lot_inputs = st.columns(2)
        lot_inputs[0].number_input(
            label="Lot Area (Square Feet)",
            value=1300,
            min_value=1000,
            max_value=220000,
            key="Lot Area"
        )

        lot_inputs[1].number_input(
            label="Lot Frontage (Feet)",
            value=50,
            min_value=20,
            max_value=350,
            step=1,
            key="Lot Frontage"
        )

        lot_inputs[0].selectbox(
            label="Lot Shape",
            options=['IR1', 'Reg', 'IR2', 'IR3'],
            key="Lot Shape"
        )

        lot_inputs[1].selectbox(
            label="Lot Configuration",
            options=['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3'],
            key="Lot Config"
        )

        lot_inputs[0].selectbox(
            label="Street Type",
            options=['Pave', 'Grvl'],
            key="Street"
        )

        lot_inputs[1].selectbox(
            label="Alley Type",
            options=['Pave', 'Grvl', 'NA'],
            key="Alley"
        )

        lot_inputs[0].selectbox(
            label="Utilities",
            options=['AllPub', 'NoSewr', 'NoSeWa'],
            key="Utilities"
        )

        lot_inputs[1].selectbox(
            label="Land Slope",
            options=['Gtl', 'Mod', 'Sev'],
            key="Land Slope"
        )

        lot_inputs[0].selectbox(
            label="First Proximity",
            options=['Norm', 'Feedr', 'PosN', 'RRNe', 'RRAe', 'Artery', 'PosA', 'RRAn', 'RRNn'],
            key="Condition 1"
        )

        lot_inputs[1].selectbox(
            label="Second Proximity",
            options=['Norm', 'Feedr', 'PosA', 'PosN', 'Artery', 'RRNn', 'RRAe', 'RRAn'],
            key="Condition 2"
        )

        lot_inputs[0].selectbox(
            label="Zoning Classification",
            options=['RL', 'RH', 'FV', 'RM', 'C (all)', 'I (all)', 'A (agr)'],
            key="MS Zoning"
        )

        lot_inputs[1].selectbox(
            label="Land Contour",
            options=['Lvl', 'HLS', 'Bnk', 'Low'],
            key="Land Contour"
        )

        lot_inputs[0].selectbox(label="Neighborhood",
                                options=['NAmes', 'Gilbert', 'StoneBr', 'NWAmes', 'Somerst', 'BrDale', 'NPkVill', 'NridgHt', 'Blmngtn', 'NoRidge', 'SawyerW', 'Sawyer',
                                         'Greens', 'BrkSide', 'OldTown', 'IDOTRR', 'ClearCr', 'SWISU', 'Edwards', 'CollgCr', 'Crawfor', 'Blueste', 'Mitchel', 'Timber',
                                         'MeadowV', 'Veenker', 'GrnHill', 'Landmrk'],
                                key="Neighborhood")


    with st.expander(label="Outdoor Space"):
        st.image("app_src/outdoor.png")
        outdoor_inputs = st.columns(2)

        outdoor_inputs[0].number_input(
            label="Wood Deck Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=500,
            key="Wood Deck SF"
        )

        outdoor_inputs[1].number_input(
            label="Open Porch Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=300,
            key="Open Porch SF"
        )

        outdoor_inputs[0].number_input(
            label="Enclosed Porch Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=300,
            key="Enclosed Porch"
        )

        outdoor_inputs[1].number_input(
            label="3 Season Porch Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=300,
            key="3Ssn Porch"
        )

        outdoor_inputs[0].number_input(
            label="Screen Porch Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=300,
            key="Screen Porch"
        )

        outdoor_inputs[1].number_input(
            label="Pool Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=800,
            key="Pool Area"
        )

        outdoor_inputs[0].selectbox(
            label="Pool Quality",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
            key="Pool QC"
        )

        outdoor_inputs[1].selectbox(
            label="Fence Quality",
            options=['GdWo', 'GdPrv', 'MnPrv', 'MnWw', 'NA'],
            key="Fence"
        )

        outdoor_inputs[0].selectbox(
            label="Miscellaneous Feature",
            options=['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
            key="Misc Feature"
        )

        outdoor_inputs[1].number_input(
            label="Miscellaneous Feature Value",
            value=0,
            min_value=0,
            max_value=10000,
            key="Misc Val"
        )


    with st.expander(label="Heating and Cooling"):
        st.image("app_src/heating.png")
        temperature_inputs = st.columns(2)
        temperature_inputs[0].selectbox(
            label="Heating Type",
            options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
            key="Heating"
        )

        temperature_inputs[1].selectbox(
            label="Heating Quality",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            key="Heating QC"
        )

        temperature_inputs[0].selectbox(
            label="Central Air Conditioning",
            options=['Y', 'N'],
            key="Central Air"
        )

        temperature_inputs[1].selectbox(
            label="Electrical System",
            options=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
            key="Electrical"
        )


with inputs[1]:
    with st.expander(label="Building Information"):
        st.image("app_src/dwelling.png")
        building_inputs = st.columns(2)

        building_inputs[0].number_input(
            label="Overall Quality (1-10)",
            value=5,
            min_value=1,
            max_value=10,
            key="Overall Qual"
        )

        building_inputs[1].number_input(
            label="Overall Condition (1-10)",
            value=5,
            min_value=1,
            max_value=10,
            key="Overall Cond"
        )

        building_inputs[0].number_input(
            label="Original Construction Year",
            value=2000,
            min_value=1800,
            max_value=2024,
            key="Year Built"
        )

        building_inputs[1].number_input(
            label="Remodeled/Additions Year",
            value=2000,
            min_value=1900,
            max_value=2024,
            key="Year Remod/Add"
        )

        building_inputs[0].selectbox(
            label="Building Type",
            options=['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
            key="Bldg Type"
        )

        building_inputs[1].selectbox(
            label="House Style",
            options=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
            key="House Style"
        )

        building_inputs[0].selectbox(
            label="Roof Style",
            options=['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
            key="Roof Style"
        )

        building_inputs[1].selectbox(
            label="Roof Material",
            options=['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
            key="Roof Matl"
        )

        building_inputs[0].selectbox(
            label="Exterior Covering (1st)",
            options=['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shngl'],
            key="Exterior 1st"
        )

        building_inputs[1].selectbox(
            label="Exterior Covering (2nd)",
            options=['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shngl'],
            key="Exterior 2nd"
        )

        building_inputs[0].selectbox(
            label="Foundation Type",
            options=['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            key="Foundation"
        )

        building_inputs[1].selectbox(
            label="Masonry Veneer Type",
            options=['None', 'BrkCmn', 'BrkFace', 'Stone'],
            key="Mas Vnr Type"
        )

        building_inputs[0].number_input(
            label="Masonry Veneer Area (sq-f)",
            value=0,
            min_value=0,
            max_value=2000,
            key="Mas Vnr Area"
        )

        building_inputs[1].number_input(
            label="Dwelling Area (Square Feet)",
            value=20,
            min_value=20,
            max_value=200,
            step=5,
            key="MS SubClass"
        )

        building_inputs[0].selectbox(
            label="Exterior Quality",
            options=['TA', 'Gd', 'Ex', 'Fa'],
            key="Exter Qual"
        )

        building_inputs[1].selectbox(
            label="Exterior Condition",
            options=['TA', 'Gd', 'Fa', 'Po', 'Ex'],
            key="Exter Cond"
        )


    with st.expander(label="Basement Information"):
        st.image("app_src/basement.png")
        basement_inputs = st.columns(2)

        basement_inputs[0].selectbox(
            label="Basement Quality",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
            key="Bsmt Qual"
        )

        basement_inputs[1].selectbox(
            label="Basement Condition",
            options=['Gd', 'TA', 'Po', 'Fa', 'Ex', 'NA'],
            key="Bsmt Cond"
        )

        basement_inputs[0].selectbox(
            label="Basement Exposure (Walkout or Garden Level Walls)",
            options=['Gd', 'Av', 'Mn', 'No', 'NA'],
            key="Bsmt Exposure"
        )

        basement_inputs[1].selectbox(
            label="Basement Finished Type 1 (Rating of Finished Area)",
            options=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
            key="BsmtFin Type 1"
        )

        basement_inputs[0].number_input(
            label="Basement Finished Area 1 (Square Feet of Type 1)",
            value=0,
            min_value=0,
            max_value=2000,
            key="BsmtFin SF 1"
        )

        basement_inputs[1].selectbox(
            label="Basement Finished Type 2 (Rating of Second Finished Area)",
            options=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
            key="BsmtFin Type 2"
        )

        basement_inputs[0].number_input(
            label="Basement Finished Area 2 (Square Feet of Type 2)",
            value=0,
            min_value=0,
            max_value=2000,
            key="BsmtFin SF 2"
        )

        basement_inputs[1].number_input(
            label="Unfinished Basement AreA",
            value=0,
            min_value=0,
            max_value=2000,
            key="Bsmt Unf SF"
        )

        basement_inputs[0].number_input(
            label="Total Basement Area",
            value=0,
            min_value=0,
            max_value=4000,
            key="Total Bsmt SF"
        )

        basement_inputs[1].number_input(
            label="Full Basement Bathrooms",
            value=0,
            min_value=0,
            max_value=3,
            key="Bsmt Full Bath"
        )

        basement_inputs[0].number_input(
            label="Half Basement Bathrooms",
            value=0,
            min_value=0,
            max_value=3,
            key="Bsmt Half Bath"
        )


    with st.expander(label="Sale Information"):
        st.image("app_src/sale.png")
        sale_inputs = st.columns(2)

        sale_inputs[0].selectbox(
            label="Sale Type (Transaction Type)",
            options=['WD', 'New', 'Con', 'CWD', 'ConLD', 'ConLI', 'ConLw', 'Oth'],
            key="Sale Type"
        )

        sale_inputs[1].selectbox(
            label="Sale Condition",
            options=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
            key="Sale Condition"
        )

        sale_inputs[0].selectbox(
            label="Selling Month",
            options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            key="Mo Sold"
        )

        sale_inputs[1].number_input(
            label="Selling Year",
            value=2020,
            min_value=2000,
            max_value=2024,
            key="Yr Sold"
        )


with inputs[2]:
    with st.expander(label="Room and Living Space Information"):
        st.image("app_src/living_space.png")
        room_inputs = st.columns(2)

        room_inputs[0].number_input(
            label="1st Floor Area (Square Feet)",
            value=800,
            min_value=0,
            max_value=4000,
            key="1st Flr SF"
        )

        room_inputs[1].number_input(
            label="2nd Floor Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=4000,
            key="2nd Flr SF"
        )

        room_inputs[0].number_input(
            label="Low Quality Finished Area (Square Feet)",
            value=0,
            min_value=0,
            max_value=500,
            key="Low Qual Fin SF"
        )

        room_inputs[1].number_input(
            label="Above Grade Living Area (Square Feet)",
            value=1200,
            min_value=0,
            max_value=5000,
            key="Gr Liv Area"
        )

        room_inputs[0].number_input(
            label="Full Bathrooms Above Grade",
            value=1,
            min_value=0,
            max_value=4,
            key="Full Bath"
        )

        room_inputs[1].number_input(
            label="Half Bathrooms Above Grade",
            value=1,
            min_value=0,
            max_value=4,
            key="Half Bath"
        )

        room_inputs[0].number_input(
            label="Number of Bedrooms Above Grade",
            value=3,
            min_value=0,
            max_value=10,
            key="Bedroom AbvGr"
        )

        room_inputs[1].number_input(
            label="Number of Kitchens Above Grade",
            value=1,
            min_value=0,
            max_value=3,
            key="Kitchen AbvGr"
        )

        room_inputs[0].selectbox(
            label="Kitchen Quality",
            options=['Ex', 'Gd', 'TA', 'Fa'],
            key="Kitchen Qual"
        )

        room_inputs[1].number_input(
            label="Total Rooms Above Grade",
            value=6,
            min_value=0,
            max_value=14,
            key="TotRms AbvGrd"
        )

        room_inputs[0].selectbox(
            label="Functional Condition",
            options=['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
            key="Functional"
        )

        room_inputs[1].number_input(
            label="Number of Fireplaces",
            value=1,
            min_value=0,
            max_value=4,
            step=1,
            key="Fireplaces"
        )

        room_inputs[0].selectbox(
            label="Fireplace Quality",
            options=['Gd', 'TA', 'Po', 'Ex', 'Fa'],
            key="Fireplace Qu"
        )


    with st.expander(label="Garage Information"):
        st.image("app_src/garage.png")
        garage_inputs = st.columns(2)

        garage_inputs[0].selectbox(
            label="Garage TypE",
            options=['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', 'NA'],
            key="Garage Type"
        )

        garage_inputs[1].number_input(
            label="Year Garage Built",
            value=2000,
            min_value=1900,
            max_value=2024,
            key="Garage Yr Blt"
        )

        garage_inputs[0].selectbox(
            label="Garage Finish (Interior Finish of the Garage)",
            options=['RFn', 'Unf', 'Fin', 'NA'],
            key="Garage Finish"
        )

        garage_inputs[1].number_input(
            label="Number of Cars",
            value=2,
            min_value=0,
            max_value=4,
            key="Garage Cars"
        )

        garage_inputs[0].number_input(
            label="Garage Area (Size of Garage in Square Feet)",
            value=500,
            min_value=0,
            max_value=1500,
            key="Garage Area"
        )

        garage_inputs[1].selectbox(
            label="Garage Quality",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
            key="Garage Qual"
        )

        garage_inputs[0].selectbox(
            label="Garage Condition",
            options=['Gd', 'TA', 'Po', 'Fa', 'Ex', 'NA'],
            key="Garage Cond"
        )

        garage_inputs[1].selectbox(
            label="Paved Drive",
            options=['P', 'Y', 'N'],
            key="Paved Drive"
        )


def get_prediction(input_data):
    """
    Function to call the API and get the prediction
    """
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()['predictions']
            return prediction
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Error occurred: {e}")



input_data = pd.DataFrame.from_dict(st.session_state, orient="index").transpose()

st.dataframe(input_data, use_container_width=True)

if st.button("Predict"):
    st.write("Sending data to API...")
    prediction = get_prediction(input_data.to_dict(orient='records'))

    if prediction:
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
