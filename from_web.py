from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from predict_on_img import ModelInit
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ModelInit(path_checkpoint="/root/acme-model-2/lds-weights/model_fold_4.pth")

@app.get("/testing")
async def testing_endpoint():
    return {
        "message": "Hello from FastAPI!"
    }

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    try:
        logger.info("Received image upload request")

        image_data = await image.read()
        logger.info("Image size: %d bytes", len(image_data))
        logger.info("Image content type: %s", image.content_type)
        
        # Log first few bytes of the image data
        logger.info("First few bytes of image data: %s", image_data[:10])

        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify that it is an image
            img = Image.open(io.BytesIO(image_data))  # Reopen image after verification
        except UnidentifiedImageError as e:
            logger.error("UnidentifiedImageError: %s", e)
            raise HTTPException(status_code=400, detail="Cannot identify image file. Please upload a valid JPEG or PNG image.")
        
        predictions = model.predict_on_img(img)
        
        logger.info("Predictions computed successfully")
        
        # Convert predictions to JSON-serializable format
        predictions = [tensor.tolist() for tensor in predictions]
        
        return JSONResponse(content={
            "predictions": predictions
        })
    except HTTPException as http_err:
        return JSONResponse(content={"error": http_err.detail}, status_code=http_err.status_code)
    except Exception as e:
        logger.exception("Exception occurred while processing image")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
