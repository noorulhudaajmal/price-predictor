import pandas as pd


def dynamic_importer() -> str:
    """
    Dynamically imports the data for testing out the model

    (eqv to real world API call)
    """

    data = {
        "Order": [1, 2],
        "PID": [526301100, 526301120],
        "MS SubClass": [60, 20],
        "MS Zoning": ["RL", "RL"],
        "Lot Frontage": [65.0, 80.0],
        "Lot Area": [8450, 9600],
        "Street": ["Pave", "Pave"],
        "Alley": [None, None],
        "Lot Shape": ["Reg", "IR1"],
        "Land Contour": ["Lvl", "Lvl"],
        "Utilities": ["AllPub", "AllPub"],
        "Lot Config": ["Inside", "Corner"],
        "Land Slope": ["Gtl", "Gtl"],
        "Neighborhood": ["CollgCr", "Veenker"],
        "Condition 1": ["Norm", "Norm"],
        "Condition 2": ["Norm", "Norm"],
        "Bldg Type": ["1Fam", "1Fam"],
        "House Style": ["2Story", "1Story"],
        "Overall Qual": [7, 6],
        "Overall Cond": [5, 8],
        "Year Built": [2003, 1976],
        "Year Remod/Add": [2004, 1976],
        "Roof Style": ["Gable", "Gable"],
        "Roof Matl": ["CompShg", "CompShg"],
        "Exterior 1st": ["VinylSd", "MetalSd"],
        "Exterior 2nd": ["VinylSd", "MetalSd"],
        "Mas Vnr Type": ["BrkFace", "None"],
        "Mas Vnr Area": [196, 0],
        "Exter Qual": ["Gd", "TA"],
        "Exter Cond": ["TA", "TA"],
        "Foundation": ["PConc", "CBlock"],
        "Bsmt Qual": ["Gd", "TA"],
        "Bsmt Cond": ["TA", "TA"],
        "Bsmt Exposure": ["No", "Gd"],
        "BsmtFin Type 1": ["GLQ", "ALQ"],
        "BsmtFin SF 1": [706, 978],
        "BsmtFin Type 2": ["Unf", "Unf"],
        "BsmtFin SF 2": [0, 0],
        "Bsmt Unf SF": [150, 284],
        "Total Bsmt SF": [856, 1262],
        "Heating": ["GasA", "GasA"],
        "Heating QC": ["Ex", "Ex"],
        "Central Air": ["Y", "Y"],
        "Electrical": ["SBrkr", "SBrkr"],
        "1st Flr SF": [856, 1262],
        "2nd Flr SF": [854, 0],
        "Low Qual Fin SF": [0, 0],
        "Gr Liv Area": [1710, 1262],
        "Bsmt Full Bath": [1, 0],
        "Bsmt Half Bath": [0, 1],
        "Full Bath": [2, 2],
        "Half Bath": [1, 0],
        "Bedroom AbvGr": [3, 3],
        "Kitchen AbvGr": [1, 1],
        "Kitchen Qual": ["Gd", "TA"],
        "TotRms AbvGrd": [8, 6],
        "Functional": ["Typ", "Typ"],
        "Fireplaces": [1, 1],
        "Fireplace Qu": ["TA", "TA"],
        "Garage Type": ["Attchd", "Attchd"],
        "Garage Yr Blt": [2003, 1976],
        "Garage Finish": ["RFn", "RFn"],
        "Garage Cars": [2, 2],
        "Garage Area": [548, 460],
        "Garage Qual": ["TA", "TA"],
        "Garage Cond": ["TA", "TA"],
        "Paved Drive": ["Y", "Y"],
        "Wood Deck SF": [210, 140],
        "Open Porch SF": [60, 0],
        "Enclosed Porch": [0, 0],
        "3Ssn Porch": [0, 0],
        "Screen Porch": [0, 120],
        "Pool Area": [0, 0],
        "Pool QC": [None, None],
        "Fence": ["MnPrv", "MnPrv"],
        "Misc Feature": [None, None],
        "Misc Val": [0, 0],
        "Mo Sold": [5, 6],
        "Yr Sold": [2010, 2010],
        "Sale Type": ["WD", "WD"],
        "Sale Condition": ["Normal", "Normal"],
        "SalePrice": [208500, 181500]
    }

    df = pd.DataFrame(data)

    json_data = df.to_json(orient="split")

    return json_data
