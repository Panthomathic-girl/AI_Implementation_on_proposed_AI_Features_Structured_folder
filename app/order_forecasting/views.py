# app/order_pattern_forecasting/customer_forecasting/views.py
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from datetime import datetime
import shutil
from pathlib import Path

# Local imports
from .schema import CustomerForecastResponse
from .schema import RetrainResponse
from .schema import PredictRequest
from .service import predict_raw
from .service import fine_tune_with_new_csv
from .service import train_pro_independent_model  # Now from service.py

router = APIRouter(prefix="/order", tags=["Customer-Level Forecasting"])

@router.post("/train_from_csv")
async def train_from_uploaded_csv(file: UploadFile = File(...)):
    """
    FULLY DYNAMIC: Upload CSV → saved → trained → model deployed
    No hardcoded paths anywhere
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    # Temporary upload
    temp_path = Path(f"temp_upload_{datetime.now():%Y%m%d_%H%M%S}.csv")
    final_dataset_path = Path("data/Order_Pattern_Forecasting_dataset.csv")

    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Replace master dataset
        if final_dataset_path.exists():
            final_dataset_path.unlink()  # Clean replace

        temp_path.replace(final_dataset_path)

        # Train using the file we just saved
        model_path = train_pro_independent_model(data_path=final_dataset_path)

        size_mb = Path(model_path).stat().st_size / (1024 * 1024)

        return {
            "status": "success",
            "new_model_path": str(model_path),
            "dataset_used": str(final_dataset_path),
            "model_size_mb": round(size_mb, 2),
            "message": (
                f"FULL TRAINING SUCCESS!\n"
                f"Model: {model_path}\n"
                f"Trained on: {final_dataset_path.name}\n"
                f"Size: {size_mb:.2f} MB\n"
                f"Check console for detailed stats"
            ),
            "trained_at": datetime.now().isoformat()
        }

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict", response_model=CustomerForecastResponse)
async def predict_customer_orders(request: PredictRequest):
    year, month = request.year, request.month
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be 1–12")

    raw = predict_raw(year, month)

    customer_dict = {}
    for p in raw["predictions"]:
        cust = p["customer_id"]
        prod = p["product_id"]
        qty = p["predicted_quantity"]
        customer_dict.setdefault(cust, {})[prod] = qty

    customer_orders = [
        {"customer_id": c, "products": p} for c, p in customer_dict.items()
    ]

    return {
        "forecast_month": f"{year}-{month:02d}",
        "total_predicted_orders": int(raw["total_predicted_orders"]),
        "total_customers_expected_to_order": len(customer_orders),
        "customer_orders": customer_orders[:100],  # limit for response size
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"Customer forecast for {month:02d}/{year}"
    }


# views.py

@router.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    temp_path = Path(f"temp_retrain_{datetime.now():%Y%m%d_%H%M%S}.csv")
    master_path = Path("data/Order_Pattern_Forecasting_dataset.csv")  # This is the only place it appears

    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # FULLY DYNAMIC CALL — both paths passed explicitly
        stats = fine_tune_with_new_csv(
            uploaded_csv_path=temp_path,
            master_dataset_path=master_path
        )

        return {
            "success": True,
            "message": stats["message"],
            "new_model_path": stats["new_model_path"],
            "model_version": stats["model_version"],
            "historical_records": stats["historical_records"],
            "uploaded_valid_rows": stats["uploaded_valid_rows"],
            "total_records_used": stats["total_records_used"],
            "duplicates_removed": stats["duplicates_removed"],
            "training_samples": stats["training_samples"],
            "positive_orders": stats["positive_orders"],
            "duration_seconds": stats["duration_seconds"],
            "retrained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


