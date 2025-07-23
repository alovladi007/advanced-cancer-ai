"""
Production-Ready Inference Server for Cancer Detection AI
FastAPI-based REST API with ONNX optimization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List
import uvicorn
from pydantic import BaseModel
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Cancer Detection AI",
    description="State-of-the-art multimodal cancer detection system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    clinical_notes: str = ""
    patient_age: int = 0
    patient_gender: str = ""
    smoking_history: bool = False
    family_history: bool = False

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    cancer_type: str
    risk_score: float
    confidence: float
    uncertainty: float
    recommendations: List[str]

class CancerInferenceServer:
    """
    Production inference server for cancer detection
    """
    
    def __init__(self, model_path: str = "models/cancer_detector.onnx"):
        self.model_path = model_path
        self.session = None
        self.class_names = [
            "No Cancer",
            "Lung Cancer", 
            "Breast Cancer",
            "Prostate Cancer",
            "Colorectal Cancer"
        ]
        self.load_model()
    
    def load_model(self):
        """Load ONNX model for inference"""
        try:
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Log model info
            input_info = self.session.get_inputs()[0]
            logger.info(f"Model input shape: {input_info.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to PyTorch model if ONNX fails
            self.load_pytorch_model()
    
    def load_pytorch_model(self):
        """Fallback to PyTorch model"""
        try:
            from src.models.multimodal_cancer_detector import create_model
            self.model = create_model()
            self.model.eval()
            logger.info("Loaded PyTorch model as fallback")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess uploaded image"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Transpose to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")
    
    def predict(self, image_array: np.ndarray, clinical_data: Dict) -> Dict:
        """Run inference on preprocessed data"""
        try:
            if self.session:
                # ONNX inference
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: image_array})
                predictions = outputs[0]
            else:
                # PyTorch inference
                with torch.no_grad():
                    image_tensor = torch.from_numpy(image_array)
                    predictions = self.model(image_tensor).numpy()
            
            # Apply softmax to get probabilities
            probabilities = self._softmax(predictions[0])
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Calculate uncertainty (entropy)
            uncertainty = float(-np.sum(probabilities * np.log(probabilities + 1e-8)))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                predicted_class, confidence, clinical_data
            )
            
            return {
                "cancer_type": self.class_names[predicted_class],
                "risk_score": confidence,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "recommendations": recommendations,
                "all_probabilities": {
                    name: float(prob) for name, prob in zip(self.class_names, probabilities)
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _generate_recommendations(self, predicted_class: int, confidence: float, clinical_data: Dict) -> List[str]:
        """Generate clinical recommendations based on prediction"""
        recommendations = []
        
        if predicted_class == 0:  # No Cancer
            if confidence < 0.8:
                recommendations.append("Consider follow-up screening in 6 months")
            recommendations.append("Maintain regular screening schedule")
            recommendations.append("Continue healthy lifestyle practices")
        else:
            recommendations.append("Immediate consultation with oncologist recommended")
            recommendations.append("Additional diagnostic tests may be required")
            
            if confidence < 0.7:
                recommendations.append("Consider second opinion due to uncertainty")
            
            if clinical_data.get("smoking_history"):
                recommendations.append("Smoking cessation counseling strongly recommended")
            
            if clinical_data.get("family_history"):
                recommendations.append("Genetic counseling may be beneficial")
        
        return recommendations

# Global inference server instance
inference_server = CancerInferenceServer()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Advanced Cancer Detection AI Server", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": inference_server.session is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cancer(
    image: UploadFile = File(...),
    clinical_notes: str = "",
    patient_age: int = 0,
    patient_gender: str = "",
    smoking_history: bool = False,
    family_history: bool = False
):
    """
    Predict cancer from medical image and clinical data
    """
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await image.read()
        
        # Preprocess image
        image_array = inference_server.preprocess_image(image_bytes)
        
        # Prepare clinical data
        clinical_data = {
            "clinical_notes": clinical_notes,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "smoking_history": smoking_history,
            "family_history": family_history
        }
        
        # Run prediction
        result = inference_server.predict(image_array, clinical_data)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict")
async def batch_predict(images: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple images
    """
    results = []
    
    for image in images:
        try:
            image_bytes = await image.read()
            image_array = inference_server.preprocess_image(image_bytes)
            result = inference_server.predict(image_array, {})
            results.append({
                "filename": image.filename,
                "prediction": result
            })
        except Exception as e:
            results.append({
                "filename": image.filename,
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    return {
        "model_path": inference_server.model_path,
        "class_names": inference_server.class_names,
        "input_shape": [1, 3, 224, 224],
        "framework": "ONNX" if inference_server.session else "PyTorch"
    }

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
