import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Load Environment Variables & Configure API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found in the .env file.")
genai.configure(api_key=GOOGLE_API_KEY)


# --- 2. FastAPI App Instance & CORS ---
app = FastAPI(
    title="AI Fitness Coach API (Flexible & Robust)",
    description="An API that uses free-text input to generate personalized diet plans and workout splits.",
    version="3.1.0", 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Pydantic Models (No Enums for Flexibility) ---

# --- DIET PLAN MODELS ---
class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=80)
    gender: str = Field(..., example="Male")
    height_cm: float = Field(..., gt=100, lt=250)
    weight_kg: float = Field(..., gt=30, lt=200)
    fitness_goal: str = Field(..., example="I want to lose fat and build some muscle.")
    dietary_preference: str = Field(..., example="Mostly vegetarian, but I eat eggs.")
    cuisine: str = Field(default="Indian", example="North Indian")
    language: str = Field(default="English", example="Hindi")
    allergies: List[str] = Field([], example=["peanuts"])
    meals_per_day: int = Field(5, gt=2, lt=7)
    is_premium: bool = Field(default=False)

class MealDetail(BaseModel):
    food_items: List[str]
    calories: int
    protein_g: int
    carbs_g: int
    fats_g: int

class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail]
    daily_totals: Dict[str, int]

class DietPlanResponse(BaseModel):
    plan_summary: Dict[str, str]
    weekly_plan: Dict[str, DailyPlan]
    general_tips: List[str]


# --- WORKOUT SPLIT MODELS ---
class WorkoutRequest(BaseModel):
    days_per_week: int = Field(..., gt=0, le=7)
    experience_level: str = Field(..., example="I have been lifting for 2 years.")
    goal: str = Field(..., example="Get stronger and look more athletic.")
    focus: Optional[str] = Field(None, example="legs")
    is_premium: bool = Field(default=False)

class ExerciseDetail(BaseModel):
    name: str
    sets: str
    reps: str

class DailyWorkout(BaseModel):
    day: str
    focus: str
    exercises: List[ExerciseDetail]

class WorkoutSplitResponse(BaseModel):
    workout_plan: List[DailyWorkout]


# --- 4. API Endpoints ---

# --- DIET PLAN ENDPOINT ---
def create_diet_prompt(cuisine: str, language: str) -> str:
    """Creates a robust, dynamic system prompt for the diet plan generator."""
    return f"""You are an expert nutritionist generating a diet plan.

**CRITICAL RULES:**
1.  **LANGUAGE:** All text in your response MUST be in **{language}**.
2.  **CUISINE:** The plan MUST be based on **{cuisine}** food items.
3.  **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text or markdown outside of the JSON brackets.
4.  **STRICT SCHEMA:** The JSON must have exactly three top-level keys: `plan_summary`, `weekly_plan`, and `general_tips`.
5.  **DATA TYPES:** All nutritional values (`calories`, `protein_g`, `carbs_g`, `fats_g`) MUST be integers, NOT strings. This is a critical rule."""

@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """Generates a 7-day personalized diet plan using free-text inputs."""
    try:
        # NOTE: Using the latest valid models. You can change these if new models are released.
        PREMIUM_MODEL = "gemini-1.5-pro-latest"
        STANDARD_MODEL = "gemini-1.5-flash-latest"
        
        model_to_use = PREMIUM_MODEL if user_input.is_premium else STANDARD_MODEL
        print(f"Diet plan request for {user_input.cuisine} cuisine in {user_input.language}. Using model: {model_to_use}")

        system_prompt = create_diet_prompt(user_input.cuisine, user_input.language)
        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        user_prompt_data = f"Interpret these user details and generate the plan: {user_input.model_dump_json()}"
        response = await model.generate_content_async([user_prompt_data])
        
        # DEFENSIVE VALIDATION: This block prevents the 500 error if the AI response is malformed.
        try:
            plan_data = json.loads(response.text)
            validated_plan = DietPlanResponse.model_validate(plan_data)
            return validated_plan
        except ValidationError as e:
            print(f"--- Pydantic Validation Error ---\n{e}\n--- Raw AI Response ---\n{response.text}\n")
            raise HTTPException(status_code=500, detail="The AI returned a plan with an invalid structure.")
        except json.JSONDecodeError:
            print(f"--- JSON Decode Error ---\n--- Raw AI Response ---\n{response.text}\n")
            raise HTTPException(status_code=500, detail="The AI returned a non-JSON response.")
    except Exception as e:
        print(f"An unexpected error occurred in /generate-diet-plan: {e}")
        raise HTTPException(status_code=503, detail="An AI service error occurred.")


# --- WORKOUT SPLIT ENDPOINT ---
def create_workout_prompt(request: WorkoutRequest) -> str:
    """Creates a robust system prompt for the workout split generator."""
    return f"""You are an expert fitness coach. Generate a workout split based on this user's free-text description:
- Days per Week: {request.days_per_week}
- Experience Level: "{request.experience_level}"
- Primary Goal: "{request.goal}"
- Specific Focus: "{request.focus or 'None'}"

**CRITICAL INSTRUCTIONS:**
1. **Output Format:** Your ENTIRE response must be a single, valid JSON object with ONE root key: "workout_plan".
2. **Schema Adherence:** Do not add fields not in the schema.
3. **Object Structure:** "workout_plan" must be an array of day objects. Each day must have keys "day", "focus", "exercises". Each exercise must have keys "name", "sets", "reps"."""

@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    """Generates a weekly workout split using free-text inputs."""
    try:
        # NOTE: Using the latest valid models. You can change these if new models are released.
        PREMIUM_MODEL = "gemini-1.5-pro-latest"
        STANDARD_MODEL = "gemini-1.5-flash-latest"
        
        model_to_use = PREMIUM_MODEL if request.is_premium else STANDARD_MODEL
        print(f"Workout split request. Using model: {model_to_use}")
        
        prompt = create_workout_prompt(request)
        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        response = await model.generate_content_async("") 

        # DEFENSIVE VALIDATION: This block prevents the 500 error if the AI response is malformed.
        try:
            split_data = json.loads(response.text)
            validated_split = WorkoutSplitResponse.model_validate(split_data)
            return validated_split
        except ValidationError as e:
            print(f"--- Pydantic Validation Error ---\n{e}\n--- Raw AI Response ---\n{response.text}\n")
            raise HTTPException(status_code=500, detail="The AI returned a workout with an invalid structure.")
        except json.JSONDecodeError:
            print(f"--- JSON Decode Error ---\n--- Raw AI Response ---\n{response.text}\n")
            raise HTTPException(status_code=500, detail="The AI returned a non-JSON response.")
    except Exception as e:
        print(f"An unexpected error occurred in /generate-workout-split: {e}")
        raise HTTPException(status_code=503, detail="An AI service error occurred.")


# --- ROOT ENDPOINT ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "AI Fitness Coach API is running. Go to /docs for all endpoints."}
