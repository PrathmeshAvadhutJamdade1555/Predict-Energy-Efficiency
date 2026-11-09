import pandas as pd
from train import train_and_evaluate

def predict_building(
    model,
    scaler,
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    overall_height,
    orientation,
    glazing_area,
    glazing_area_dist,
):
    feature_columns = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ]
    new_building = pd.DataFrame(
        [[
            relative_compactness,
            surface_area,
            wall_area,
            roof_area,
            overall_height,
            orientation,
            glazing_area,
            glazing_area_dist,
        ]],
        columns=feature_columns,
    )
    new_building_scaled = scaler.transform(new_building)

    # Make prediction
    prediction = model.predict(new_building_scaled)

    return prediction[0][0], prediction[0][1]
if __name__ == "__main__":
    model, scaler = train_and_evaluate("data/ENB2012_data_CSV.csv")

    heating_load, cooling_load = predict_building(
        model,
        scaler,
        relative_compactness=float(input("enter  relative compactness")),
        surface_area=float(input("enter  surface_area")),
        wall_area=float(input("enter  wall_area")),
        roof_area=float(input("enter  roof_area")),
        overall_height=float(input("enter  overall_heigh")),
        orientation=float(input("enter  orientation")),
        glazing_area=float(input("enter  glazing_area")),
        glazing_area_dist=float(input("enter  glazing_area_dist")),
    )
    print("\n--- Prediction for New Building ---")
    print(f"Predicted Heating Load: {heating_load:.2f}")
    print(f"Predicted Cooling Load: {cooling_load:.2f}")
