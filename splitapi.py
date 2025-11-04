import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
from exercises_string import exercises_new

# Assuming exercises_string.py contains a string like: exercises_new = '{"push up": "url", ...}'
exercises_dict = json.loads(exercises_new)
exercises_names = exercises_dict.keys()

# --- IMPORTANT: Replace with your actual API key ---
# It's recommended to load this from an environment variable for security.
# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key="AIzaSyB0kkKSb14LduVMUgb7TqFKHIK3IZx4HhM")

app = FastAPI(
    title="AI Fitness Coach API",
    description="A unified API to generate personalized diet plans and workout splits with premium tiers.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Diet Plan Models ---

class UserInput(BaseModel):
    age: int
    gender: str
    height_cm: float
    fitness_goal: str
    dietary_preference: str
    cuisine: str
    allergies: List[str] = []
    meals_per_day: int
    is_premium: bool = False
    current_weight: int
    target_weight: int
    language: str
    time_span: str

class MealDetail(BaseModel):
    food_items: List[str]
    calories: int
    protein_g: int
    carbs_g: int
    fats_g: int

class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail]
    daily_totals: Dict[str, int]

class WeeklyPlanObject(BaseModel):
    week_summary: str
    daily_plans: Dict[str, DailyPlan]

class DietPlanResponse(BaseModel):
    plan_summary: Dict[str, str]
    weekly_plan: List[WeeklyPlanObject]
    general_tips: List[str]

# --- Workout Models ---

class WorkoutRequest(BaseModel):
    days_per_week: int
    experience_level: str
    goal: str
    focus: Optional[str] = None
    is_premium: bool = False

class ExerciseDetail(BaseModel):
    name: str
    sets: str
    reps: str
    url: Optional[str] = None

class DailyWorkout(BaseModel):
    day: str
    focus: str
    exercises: List[ExerciseDetail]

class WorkoutSplitResponse(BaseModel):
    workout_plan: List[DailyWorkout]

# --- Prompt builders ---

def create_diet_prompt(cuisine: str, language: str, is_premium: bool) -> str:
    """Creates a dynamic prompt based on the user's premium status."""
    if is_premium:
        plan_duration_instruction = "Generate a comprehensive and varied 4-week diet plan. Each week MUST have a distinct set of meals."
        num_weeks = 4
        plan_type_description = "a monthly (4-week) plan"
    else:
        plan_duration_instruction = "Generate a 1-week diet plan."
        num_weeks = 1
        plan_type_description = "a weekly plan"

    example_weekly_object = """{
        "week_summary": "Week 1: Focus on lean protein and complex carbohydrates to kickstart your metabolism.",
        "daily_plans": {
          "Monday": {
            "meals": {
              "Breakfast": { "food_items": ["Oats with whey protein", "Handful of almonds"], "calories": 400, "protein_g": 30, "carbs_g": 50, "fats_g": 10 },
              "Lunch": { "food_items": ["Grilled Chicken Breast", "Quinoa Salad"], "calories": 500, "protein_g": 40, "carbs_g": 40, "fats_g": 20 }
            },
            "daily_totals": { "total_calories": 2250, "total_protein_g": 155, "total_carbs_g": 220, "total_fats_g": 65 }
          },
          "Tuesday": "..."
        }
    }"""

    example_plan_list = f"[{example_weekly_object}]"
    if is_premium:
        example_plan_list = f"[{example_weekly_object}, ... (4 distinct weekly objects in total)]"

    return f"""You are an expert nutritionist generating a personalized diet plan.

**TASK:**
{plan_duration_instruction}

**CRITICAL RULES:**
1.  **CUISINE:** The entire plan MUST be based on **{cuisine}** food items.
2.  **LANGUAGE:** Generate the plan in **{language}**.
3.  **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text or markdown outside of the JSON.
4.  **STRICT SCHEMA:** The JSON must have three top-level keys: `plan_summary`, `weekly_plan`, and `general_tips`.
5.  **`weekly_plan` KEY:** This key's value MUST be a JSON array. It will contain {num_weeks} object(s), one for each week of {plan_type_description}.
6.  **DATA TYPES:** All calorie and macronutrient values (`calories`, `protein_g`, `carbs_g`, `fats_g`) MUST be integers.
7.  **DIET LOGIC:** Ensure the diet is calorie-deficient for weight loss or surplus for weight gain based on the user's goal. Include a variety of fruits and juices.
8.  **TIPS:** In `general_tips`, add a tip to "Follow the plan strictly to get desired results."
**EXAMPLE JSON STRUCTURE TO FOLLOW:**
{{
  "plan_summary": {{
    "estimated_daily_calories": "Approx. 2200-2400 kcal",
    "estimated_daily_protein": "Approx. 150-160 g"
  }},
  "weekly_plan": {example_plan_list},
  "general_tips": [
    "Drink 3-4 liters of water daily.",
    "Get 7-8 hours of sleep for recovery."
  ]
}}"""

@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """
    Generates a personalized diet plan.
    - **Premium (2.5)**: A comprehensive 4-week plan with weekly variety.
    - **Standard (2.0)**: A 1-week plan.
    """
    try:
        model_to_use = "gemini-2.5-flash" if user_input.is_premium else "gemini-2.0-flash"
        print(f"Diet plan request for {user_input.cuisine} cuisine. Premium: {user_input.is_premium}. Using model: {model_to_use}")

        system_prompt = create_diet_prompt(
            cuisine=user_input.cuisine,
            language=user_input.language,
            is_premium=user_input.is_premium
        )

        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        user_prompt_data = f"Here are my details, generate my diet plan:\n{json.dumps(user_input.model_dump(), indent=2)}"
        response = await model.generate_content_async([user_prompt_data])
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in /generate-diet-plan: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


# --- WORKOUT SPLIT FUNCTIONALITY (MODIFIED) ---

def create_workout_prompt(request: WorkoutRequest) -> str:
    # Define exercise lists
    cardio_exercises = [
        "jump rope", "jumping jack", "running", "treadmill running",
        "jump step-up", "battling ropes"
    ]
    stretching_exercises = [
        "stretching - hamstring stretch", "stretching - all fours squad stretch",
        "stretching - hip circles stretch", "stretching - chin-to-chest stretch",
        "stretching - feet and ankles stretch", "stretching - quadriceps lying stretch",
        "stretching - quadriceps stretch", "stretching - standing bench calf stretch",
        "stretching - seated twist (straight arm)", "stretching - seated wide angle pose sequence",
        "stretching - butterfly yoga pose", "stretching - standing side bend (bent arm)",
        "stretching - feet and ankles rotation stretch", "stretching - spine stretch",
        "stretching - iron cross stretch"
    ]

    # Initialize the exercise list for the prompt
    exercise_list_for_prompt = list(exercises_names)
    prompt_focus_instruction = f"""
- Specific Focus: {request.focus if request.focus else 'Not specified'}
All exercises in the plan must be regarding the goal muscle group.
"""
    
    # Conditionally modify the exercise list and instruction based on focus area
    if request.focus and request.focus.lower() == 'cardio':
        exercise_list_for_prompt = cardio_exercises
        prompt_focus_instruction = f"""
- Specific Focus: {request.focus}
ALL exercises for ALL days MUST be exclusively from the cardio list. DO NOT include any other exercises.
"""
    elif request.focus and request.focus.lower() == 'stretching':
        exercise_list_for_prompt = stretching_exercises
        prompt_focus_instruction = f"""
- Specific Focus: {request.focus}
ALL exercises for ALL days MUST be exclusively from the stretching list. DO NOT include any other exercises.
"""
    
    # Now, build the main prompt string
    return f"""You are an expert fitness coach. Generate a detailed, day-wise workout split based on:
- Days per Week: {request.days_per_week}
- Experience Level: {request.experience_level}
- Goal: {request.goal}
{prompt_focus_instruction}

STRICTLY FOLLOW THESE RULES:
1. The ENTIRE response MUST be a single, valid JSON object with the key "workout_plan".
2. The `workout_plan` array MUST contain exactly {request.days_per_week} objects.
3. The `focus` of EVERY day MUST be the user's requested focus area: {request.focus if request.focus else request.goal}.
4. For EACH exercise object, the ONLY allowed keys are "name", "sets", and "reps". DO NOT INCLUDE ANY OTHER KEYS like "duration", "rest", or "notes".
5. For focus area 'cardio', only use exercises from the provided cardio list.
6. For focus area 'stretching', only use exercises from the provided stretching list.
7. For all other focus areas, use exercises from the complete list related to the goal muscle group.
8. Do not invent exercises outside the provided lists.
9. Do not add any text, markdown, or explanations outside the JSON object.

A clean day-wise workout plan.

Input: {exercise_list_for_prompt}
Exercise names must be lowercase.
"""


@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    """
    Generates a weekly workout split based on user preferences.
    - Uses **Gemini 2.5 Flash** for premium requests.
    - Uses **Gemini 2.0 Flash** for standard requests.
    """
    try:
        prompt = create_workout_prompt(request)
        model_to_use = "gemini-2.5-flash" if request.is_premium else "gemini-2.0-flash"
        print(f"Workout split request. Using model: {model_to_use}")
        print(json.dumps(request.dict(), indent=2))

        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

        response = await model.generate_content_async(prompt)
        workout_json = json.loads(response.text)  # Parsed AI output

        for day_plan in workout_json.get("workout_plan", []):
            for exercise in day_plan.get("exercises", []):
                # Normalize sets/reps to strings
                if "sets" in exercise:
                    exercise["sets"] = str(exercise["sets"])
                if "reps" in exercise:
                    exercise["reps"] = str(exercise["reps"])

                # Add URL if found
                name = exercise.get("name", "").lower().strip()
                if name in exercises_dict:
                    exercise["url"] = exercises_dict.get(name)

        return workout_json  # Must be inside try

    except Exception as e:
        print(f"Error in /generate-workout-split: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


@app.get("/", tags=["Status"])
def read_root():
    return {"status": "AI Fitness Coach API is running. Go to /docs for all endpoints."}

if __name__ == "__main__":
    import uvicorn
    # Make sure to have an exercises_string.py file in the same directory
    # with content like: exercises_new = '{"push up": "some_url", "squat": "another_url"}'
    uvicorn.run(app, host="0.0.0.0", port=8000)
