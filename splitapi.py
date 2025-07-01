import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

# --- Configuration ---
# Configure the Gemini API key
try:
    genai.configure(api_key="AIzaSyDDwpDZsSFhWKQ8-7Xc6_DGXErhL7WFf7Y")
except KeyError:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Gym Workout Split Generator",
    description="An API that uses Google Gemini to create personalized workout splits.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allow all origins for simple local testing.
# For production, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Request and Response ---
class WorkoutRequest(BaseModel):
    days_per_week: int = Field(..., gt=0, le=7, description="Number of workout days per week (1-7).")
    experience_level: Literal["beginner", "intermediate", "advanced"]
    goal: Literal["muscle gain", "fat loss", "general fitness"]
    focus: Optional[str] = Field(None, description="Optional: Specific muscle group or area to focus on (e.g., 'legs', 'upper body').")


# --- Gemini Prompt Engineering ---
def create_gemini_prompt(request: WorkoutRequest) -> str:
    """Creates a detailed, structured prompt for the Gemini API."""
    prompt = f"""
    You are an expert fitness coach and personal trainer.
    Your task is to generate a detailed, day-wise workout split for a gym-goer based on the following criteria:

    - Days per Week: {request.days_per_week}
    - Experience Level: {request.experience_level}
    - Primary Goal: {request.goal}
    {'- Specific Focus: ' + request.focus if request.focus else ''}

    Please create a plan for {request.days_per_week} workout days and specify rest days.

    IMPORTANT INSTRUCTIONS:
    1. Your response MUST be a valid JSON object.
    2. Do NOT include any text, markdown formatting (like ```json), greetings, or explanations outside of the main JSON object.
    3. The JSON object should have a single root key: "workout_plan".
    4. The "workout_plan" value should be an array of objects.
    5. Each object in the array represents a day and must have these keys:
        - "day": (e.g., "Day 1", "Day 2", "Rest Day")
        - "focus": (e.g., "Chest & Triceps", "Legs & Abs", "Rest")
        - "exercises": An array of exercise objects. For rest days, this can be an empty array.
    6. Each exercise object must have these keys:
        - "name": The name of the exercise (e.g., "Barbell Bench Press").
        - "sets": The number of sets (e.g., "3-4").
        - "reps": The repetition range (e.g., "8-12").

    Here is an example structure for a single day object:
    {{
        "day": "Day 1",
        "focus": "Chest & Triceps",
        "exercises": [
            {{
                "name": "Barbell Bench Press",
                "sets": "4",
                "reps": "6-8"
            }},
            {{
                "name": "Incline Dumbbell Press",
                "sets": "3",
                "reps": "8-12"
            }}
        ]
    }}

    Now, generate the complete JSON object for the user's request.
    """
    return prompt

# --- API Endpoint ---
@app.post("/generate-workout-split")
async def generate_workout_split(request: WorkoutRequest):
    """
    Generates a workout split using the Gemini API based on user input.
    """
    try:
        prompt = create_gemini_prompt(request)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        # Parse the JSON response
        workout_json = json.loads(cleaned_response_text)
        return workout_json

    except json.JSONDecodeError:
        print("Error: Gemini returned invalid JSON.")
        print("Raw Response:", response.text)
        raise HTTPException(
            status_code=500,
            detail="Failed to parse the workout plan from the AI. The response was not valid JSON."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Gym Workout Generator API! Go to /docs for documentation."}