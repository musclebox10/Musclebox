import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Load Environment Variables & Configure API Key ---




genai.configure(api_key="AIzaSyAGT8ojwDtHKuV5HGYbhDg4QNVM0OfXKl8")

# --- 2. Create and Configure the FastAPI App Instance ---
app = FastAPI(
    title="AI Fitness Coach API",
    description="A unified API to generate personalized diet plans and workout splits with premium tiers.",
    version="2.2.0", # Version updated for new feature
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Pydantic Models ---

# --- DIET PLAN MODELS ---
class FitnessGoal(str, Enum):
    FAT_LOSS = "Fat Loss"
    MUSCLE_GAIN = "Muscle Gain"
    STABLE = "Stable"
    WEIGHT_GAIN = "Weight Gain"

class DietaryPreference(str, Enum):
    VEGETARIAN = "Vegetarian"
    NON_VEGETARIAN = "Non-Vegetarian"
    VEGAN = "Vegan"
    EGGITARIAN = "Eggitarian"

# --- NEW: Cuisine Enum ---
class Cuisine(str, Enum):
    INDIAN = "Indian"
    MEDITERRANEAN = "Mediterranean"
    ITALIAN = "Italian"
    GENERAL_WESTERN = "General Western"
    

class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=80)
    gender: str = Field(..., example="Male")
    height_cm: float = Field(..., gt=100, lt=250)
    weight_kg: float = Field(..., gt=30, lt=200)
    fitness_goal: FitnessGoal
    dietary_preference: DietaryPreference
    # --- UPDATED: Added cuisine field with a default value ---
    cuisine: Cuisine = Field(default=Cuisine.INDIAN, description="The desired cuisine for the diet plan.")
    allergies: List[str] = Field([], example=["peanuts"])
    meals_per_day: int = Field(5, gt=1, lt=6)
    is_premium: bool = Field(default=False, description="Use gemini-1.5-pro for higher quality.")
    current_weight: int = Field(..., gt=30, lt=200)
    target_weight: int = Field(..., gt=30, lt=200)

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
    experience_level: Literal["beginner", "intermediate", "advanced"]
    goal: Literal["muscle gain", "fat loss", "general fitness","weight loss","Stable"]
    focus: Optional[str] = Field(None, example="legs")
    is_premium: bool = Field(default=False, description="Use gemini-2.5-pro for a more detailed plan.")

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
# --- UPDATED: System prompt is now a dynamic function ---
def create_diet_prompt(cuisine: str) -> str:
    return f"""You are an expert nutritionist generating a diet plan.

**CRITICAL RULES:**
1.  **CUISINE:** The plan MUST be based on **{cuisine}** food items.
2.  **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text, markdown, or explanations outside of the JSON brackets.
3.  **STRICT SCHEMA:** The JSON object must have exactly three top-level keys: `plan_summary`, `weekly_plan`, and `general_tips`.
4.  **DATA TYPES:** All calorie and macronutrient values (`calories`, `protein_g`, `carbs_g`, `fats_g`) MUST be integers, NOT strings.
5. **Include Fruits and Juices in Diet plan**


**EXAMPLE JSON STRUCTURE TO FOLLOW:**
{{
  "plan_summary": {{
    "estimated_daily_calories": "Approx. 2200-2400 kcal",
    "estimated_daily_protein": "Approx. 150-160 g"
  }},
  "weekly_plan": {{
    "Monday": {{
      "meals": {{
        "Breakfast": {{
          "food_items": ["Oats with whey protein", "Handful of almonds"],
          "calories": 400, "protein_g": 30, "carbs_g": 50, "fats_g": 10
        }}
      }},
      "daily_totals": {{ "total_calories": 2250, "total_protein_g": 155, "total_carbs_g": 220, "total_fats_g": 65 }}
    }},
    "Tuesday": {{ ... }}
  }},
  "general_tips": [
    "Drink 3-4 liters of water daily.",
    "Get 7-8 hours of sleep for recovery."
  ]
}}
"""

@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """
    Generates a 7-day personalized diet plan in a specific cuisine.
    - Uses **Gemini 2.5 Pro** for premium requests.
    - Uses **Gemini 2.0 Flash** for standard requests.
    """
    try:
        model_to_use = "gemini-2.5-flash" if user_input.is_premium else "gemini-2.0-flash"
        print(f"Diet plan request for {user_input.cuisine.value} cuisine. Using model: {model_to_use}")

        # Create the dynamic system prompt
        system_prompt = create_diet_prompt(user_input.cuisine.value)

        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        user_prompt_data = f"Here are my details, generate my diet plan:\n{user_input.model_dump_json(indent=2)}"
        response = await model.generate_content_async([user_prompt_data])
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in /generate-diet-plan: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


# --- WORKOUT SPLIT ENDPOINT ---
def create_workout_prompt(request: WorkoutRequest) -> str:
    return f"""You are an expert fitness coach. Generate a detailed, day-wise workout split based on:
- Days per Week: {request.days_per_week}
- Experience Level: {request.experience_level}
- Primary Goal: {request.goal}
{'- Specific Focus: ' + request.focus if request.focus else ''}

INSTRUCTIONS:
1. Your response MUST be a valid JSON object with a single root key: "workout_plan".
2. "workout_plan" is an array of objects, one for each day (including rest days).
3. Each day object has keys: "day", "focus", and "exercises" (an array of exercise objects).
4. Each exercise object has keys: "name", "sets", "reps".
5. Do NOT include any text, markdown, or explanations outside the JSON object."""

@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    """
    Generates a weekly workout split based on user preferences.
    - Uses **Gemini 1.5 Pro** for premium requests for more nuanced splits.
    - Uses **Gemini 1.5 Flash** for standard requests.
    """
    try:
        prompt = create_workout_prompt(request)
        model_to_use = "gemini-2.5-flash" if request.is_premium else "gemini-2.0-flash"
        print(f"Workout split request. Using model: {model_to_use}")
        
        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        response = await model.generate_content_async(prompt)
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Error in /generate-workout-split: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


# --- ROOT ENDPOINT ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "AI Fitness Coach API is running. Go to /docs for all endpoints."}

# --- This block is for running the app with Uvicorn locally ---
# --- It will NOT be used by Render, which uses the Gunicorn command ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
