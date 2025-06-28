from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
from sklearn.tree import _tree
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (optional if using with frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor
predictor = joblib.load('disease_predictor.joblib')
tree = predictor.model.tree_
feature_names = [
    predictor.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    for i in tree.feature
]

# In-memory session per user (basic example using IP)
user_sessions = {}

class Answer(BaseModel):
    answer: str  # 'yes' or 'no'

@app.get("/")
def root():
    return {"message": "Disease Prediction API is running."}

@app.get("/start")
def start_diagnosis(request: Request):
    user_id = request.client.host
    user_sessions[user_id] = {"node": 0}
    symptom_index = tree.feature[0]
    symptom = feature_names[0].replace("_", " ")
    return {"question": f"Do you have {symptom}? (yes/no)"}

@app.post("/answer")
def next_question(answer: Answer, request: Request):
    user_id = request.client.host
    session = user_sessions.get(user_id)
    if not session:
        return {"error": "Session not found. Please call /start first."}

    current_node = session["node"]
    if tree.feature[current_node] == _tree.TREE_UNDEFINED:
        return {"result": "Prediction already completed."}

    if answer.answer.lower() in ["yes", "y"]:
        next_node = tree.children_left[current_node]
    else:
        next_node = tree.children_right[current_node]

    # Check if it's a leaf
    if tree.feature[next_node] == _tree.TREE_UNDEFINED:
        predicted_disease = tree.value[next_node].argmax()
        disease = predictor.model.classes_[predicted_disease]
        # End session
        user_sessions.pop(user_id, None)
        return {"prediction": f"The most likely disease is {disease}."}

    session["node"] = next_node
    symptom_index = tree.feature[next_node]
    symptom = feature_names[next_node].replace("_", " ")
    return {"question": f"Do you have {symptom}? (yes/no)"}
