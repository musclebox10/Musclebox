import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and configure API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found in the .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

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

# --- Pydantic Models ---
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

# --- Helper - Response Cleaning ---
def clean_ai_response(response_text: str) -> str:
    """Remove markdown code blocks and whitespace."""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()

# --- Helper - Structural Fixer ---
def fix_plan_structure(plan_data: dict, meals_per_day: int) -> dict:
    """Ensure plan correctness, fix common issues, convert data types."""
    # Ensure keys
    if "plan_summary" not in plan_data:
        plan_data["plan_summary"] = {
            "total_calories": "2000-2200",
            "protein_target": "120-150g",
            "goal_description": "Balanced nutrition plan"
        }
    if "weekly_plan" not in plan_data:
        plan_data["weekly_plan"] = {}
    if "general_tips" not in plan_data:
        plan_data["general_tips"] = ["Stay hydrated", "Eat balanced meals", "Exercise regularly"]

    # Generate days if missing
    for i in range(1,8):
        day_key = f"day_{i}"
        if day_key not in plan_data["weekly_plan"]:
            plan_data["weekly_plan"][day_key] = {
                "meals": {},
                "daily_totals": {"calories": 2000, "protein_g": 120, "carbs_g": 250, "fats_g": 70}
            }
        day_obj = plan_data["weekly_plan"][day_key]
        # Ensure meals for day
        if "meals" not in day_obj:
            day_obj["meals"] = {}
        day_meals = day_obj["meals"]
        meal_names = ["breakfast", "snack1", "lunch", "snack2", "dinner", "snack3"][:meals_per_day]
        for meal in meal_names:
            if meal not in day_meals:
                day_meals[meal] = {
                    "food_items": ["Sample food"],
                    "calories": 400,
                    "protein_g": 20,
                    "carbs_g": 50,
                    "fats_g": 15
                }
        # Fix data types
        for meal_data in day_meals.values():
            for key in ["calories", "protein_g", "carbs_g", "fats_g"]:
                try:
                    meal_data[key] = int(float(str(meal_data.get(key, 0)).replace('g','').replace('cal','')))
                except:
                    meal_data[key] = 0
        # daily totals
        dt = day_obj.get("daily_totals", {})
        for key in ["calories", "protein_g", "carbs_g", "fats_g"]:
            try:
                dt[key] = int(float(str(dt.get(key,0)).replace('g','').replace('cal','')))
            except:
                dt[key] = 0
        day_obj["daily_totals"] = dt
    return plan_data

# --- API: Generate Diet Plan ---
@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model_name = "gemini-2.5-flash" if user_input.is_premium else "gemini-2.0-flash"
            system_prompt = (
                f"Create a detailed 7-day diet plan in {user_input.language} "
                f"based on {user_input.cuisine} cuisine and user details."
            )
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            prompt_data = (
                f"Interpret user details and generate a JSON plan: "
                f"{user_input.model_dump_json()}"
            )
            response = await model.generate_content_async([prompt_data])
            response_text = clean_ai_response(response.text)
            plan_json = json.loads(response_text)
            plan_json = fix_plan_structure(plan_json, user_input.meals_per_day)
            validated = DietPlanResponse.model_validate(plan_json)
            return validated
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == max_retries -1:
                raise HTTPException(status_code=500, detail="Failed to parse AI response.")
        except Exception as e:
            if attempt == max_retries -1:
                raise HTTPException(status_code=503, detail="AI service failure.")
# --- API: Generate Workout Split ---
@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model_name = "gemini-2.5-flash" if request.is_premium else "gemini-2.0-flash"
            prompt = (
                f"Create a weekly {request.days_per_week}-day workout plan: "
                f"Goal: {request.goal}, Focus: {request.focus or 'General'}."
            )
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            response = await model.generate_content_async("Generate workout plan.")
            response_text = clean_ai_response(response.text)
            data = json.loads(response_text)
            if "workout_plan" not in data:
                data["workout_plan"] = []
            if not isinstance(data["workout_plan"], list):
                data["workout_plan"] = []
            # Fill missing days
            while len(data["workout_plan"]) < request.days_per_week:
                data["workout_plan"].append({
                    "day": f"Day {len(data['workout_plan'])+1}",
                    "focus": "Full Body",
                    "exercises": [
                        {"name": "Push-ups", "sets": "3", "reps": "10-15"},
                        {"name": "Squats", "sets": "3", "reps": "12-20"}
                    ]
                })
            validated = WorkoutSplitResponse.model_validate(data)
            return validated
        except (ValidationError, json.JSONDecodeError):
            if attempt == max_retries -1:
                raise HTTPException(status_code=500, detail="Invalid response.")
        except Exception:
            if attempt == max_retries -1:
                raise HTTPException(status_code=503, detail="AI failure.")
# --- Basic Root ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Running", "version": "3.1.0"}

# --- Health ---
@app.get("/health", tags=["Status"])
def health():
    return {"status": "healthy", "timestamp": "2025-07-20T17:55:00Z"}

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
