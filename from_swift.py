from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
from predict_on_img import ModelInit
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model = ModelInit(path_checkpoint="lds-weights/model_fold_4.pth")

@app.post("/upload-images/")
async def upload_images(left_image: UploadFile = File(...), right_image: UploadFile = File(...)):
    try:
        logger.info("Received image upload request")
        
        # Read and process the left image
        left_image_data = await left_image.read()
        logger.info("Left image data read successfully")
        
        try:
            left_img = Image.open(io.BytesIO(left_image_data))
            left_img.load()  # Ensure the image is properly loaded
            logger.info("Left image processed successfully")
        except Exception as e:
            logger.error(f"Invalid left image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid left image format: {str(e)}")

        # Read and process the right image
        right_image_data = await right_image.read()
        logger.info("Right image data read successfully")
        
        try:
            right_img = Image.open(io.BytesIO(right_image_data))
            right_img.load()  # Ensure the image is properly loaded
            logger.info("Right image processed successfully")
        except Exception as e:
            logger.error(f"Invalid right image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid right image format: {str(e)}")

        # Get predictions
        left_predictions = model.predict_on_img(left_img)
        right_predictions = model.predict_on_img(right_img)

        # Convert predictions to list
        left_predictions = [tensor.tolist() for tensor in left_predictions]
        right_predictions = [tensor.tolist() for tensor in right_predictions]

        logger.info("Image processing and prediction successful")

        return JSONResponse(content={
            "left_predictions": left_predictions,
            "right_predictions": right_predictions
        })

    except Exception as e:
        error_details = jsonable_encoder({"error": str(e)})
        logger.exception("Error processing images")
        raise HTTPException(status_code=500, detail=error_details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="104.248.118.84", port=8000)
