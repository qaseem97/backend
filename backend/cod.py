from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import onnxruntime as rt
import pandas as pd
import numpy as np
import json
import io

app = FastAPI(title="Air Quality AI Backend API")

# ---------------------------------------------------------
# 1. LOAD MODELS AND DATA
# ---------------------------------------------------------
try:
    with open('dashboard_data.json', 'r') as f:
        dashboard_data = json.load(f)
        
    # Only the 3 models present in the folder will be loaded
    sess_km = rt.InferenceSession("kmeans_model.onnx")
    sess_lr = rt.InferenceSession("linear_regression_model.onnx")
    sess_knn = rt.InferenceSession("knn_model.onnx")
    
    print("✅ JSON Data and ONNX Models successfully loaded!")
except Exception as e:
    print(f"❌ Error loading files: {e}")

# ---------------------------------------------------------
# 2. INPUT SCHEMA AND HELPER FUNCTIONS
# ---------------------------------------------------------
class SensorData(BaseModel):
    PM25: float
    PM10: float
    hourVal: float

def predict_with_onnx(session, input_array):
    """Passes a NumPy array to the ONNX model and returns the result."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Converting data to float32 is mandatory for ONNX
    input_array = input_array.astype(np.float32)
    result = session.run([output_name], {input_name: input_array})
    return result[0]

# ---------------------------------------------------------
# 3. API ENDPOINTS
# ---------------------------------------------------------

# Endpoint A: For historical dashboard graphs
@app.get("/api/dashboard-data")
def get_dashboard_data():
    if not dashboard_data:
        raise HTTPException(status_code=404, detail="JSON data not found")
    return dashboard_data

# Endpoint B: Single Live Prediction (Manual Entry)
@app.post("/api/predict-live")
def predict_live_aqi(data: SensorData):
    try:
        input_array = np.array([[data.PM25, data.PM10, data.hourVal]])
        
        cluster_pred = predict_with_onnx(sess_km, input_array)
        lr_pred = predict_with_onnx(sess_lr, input_array)
        knn_pred = predict_with_onnx(sess_knn, input_array)
        
        return {
            "status": "success",
            "live_predictions": {
                "kmeans_cluster": int(cluster_pred[0]),
                "linear_regression_pm25": round(float(lr_pred[0][0]), 2),
                "knn_pm25": round(float(knn_pred[0][0]), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint C: BATCH PREDICTION (Excel File Upload)
@app.post("/api/predict-file")
async def predict_from_excel(file: UploadFile = File(...)):
    """
    This endpoint takes an Excel file from the user, extracts PM25, PM10, 
    and Time/Hour, runs the ONNX models on the entire file, and returns the results.
    """
    if not file.filename.endswith(('.xls', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed.")
    
    try:
        # 1. Read the file into memory
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # 2. Check if the required columns are present
        required_cols = ['PM25', 'PM10']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in the file.")
        
        # 3. Handle Time/Hour
        if 'Time' in df.columns:
            # If 'Time' column exists, extract the hour from it
            df['Time'] = pd.to_datetime(df['Time'])
            df['hourVal'] = df['Time'].dt.hour
        elif 'hourVal' not in df.columns:
            raise HTTPException(status_code=400, detail="The file must contain either a 'Time' or 'hourVal' column.")
            
        # 4. Extract data and remove NaN values
        feature_df = df[['PM25', 'PM10', 'hourVal']].dropna()
        input_array = feature_df.values
        
        # 5. Batch Predictions
        cluster_preds = predict_with_onnx(sess_km, input_array)
        lr_preds = predict_with_onnx(sess_lr, input_array)
        knn_preds = predict_with_onnx(sess_knn, input_array)
        
        # 6. Add the results back to the DataFrame
        feature_df['Predicted_Cluster'] = cluster_preds
        feature_df['Predicted_LR_PM25'] = lr_preds
        feature_df['Predicted_KNN_PM25'] = knn_preds
        
        # Send the data back to the dashboard in JSON format
        return {
            "status": "success",
            "total_rows_processed": len(feature_df),
            "data": feature_df.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Air Quality AI Backend with File Upload is running!"}