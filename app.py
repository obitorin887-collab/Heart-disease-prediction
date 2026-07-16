from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Heart Disease Prediction")

# Templates directory
templates = Jinja2Templates(directory="templates")

# Load trained model
model = joblib.load("heart_disease_rf.pkl")

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Weight",
    "Height",
    "BMI",
    "Smoking",
    "Physical_Activity",
    "Diet",
    "Stress_Level",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total"
]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    Gender: str = Form(...),
    Weight: float = Form(...),
    Height: float = Form(...),
    BMI: float = Form(...),
    Smoking: str = Form(...),
    Physical_Activity: str = Form(...),
    Diet: str = Form(...),
    Stress_Level: str = Form(...),
    Hypertension: int = Form(...),
    Diabetes: int = Form(...),
    Hyperlipidemia: int = Form(...),
    Family_History: int = Form(...),
    Previous_Heart_Attack: int = Form(...),
    Systolic_BP: int = Form(...),
    Diastolic_BP: int = Form(...),
    Heart_Rate: int = Form(...),
    Blood_Sugar_Fasting: float = Form(...),
    Cholesterol_Total: float = Form(...)
):

    input_data = pd.DataFrame(
        [[
            Age,
            Gender,
            Weight,
            Height,
            BMI,
            Smoking,
            Physical_Activity,
            Diet,
            Stress_Level,
            Hypertension,
            Diabetes,
            Hyperlipidemia,
            Family_History,
            Previous_Heart_Attack,
            Systolic_BP,
            Diastolic_BP,
            Heart_Rate,
            Blood_Sugar_Fasting,
            Cholesterol_Total
        ]],
        columns=FEATURE_ORDER
    )

    prediction = model.predict(input_data)[0]

    result = "❤️ High Risk" if prediction == 1 else "💚 Low Risk"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": result
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )