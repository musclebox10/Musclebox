import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
from exercises_string import exercises_new

# --- Setup ---
# ⚠️ Replace with your real Gemini API key
genai.configure(api_key="AIzaSyAGT8ojwDtHKuV5HGYbhDg4QNVM0OfXKl8")

exercises_dict = json.loads(exercises_new)
exercises_names = exercises_dict.keys()

app = FastAPI(
    title="AI Fitness Coach API",
    description="A unified API to generate personalized diet plans and workout splits with premium tiers.",
    version="2.5.0",
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
    time_span: str   # "weekly" or "monthly"


class MealDetail(BaseModel):
    food_items: Optional[List[str]] = None
    item: Optional[str] = None  # sometimes Gemini sends "item" instead of "food_items"
    calories: int
    protein_g: int
    carbs_g: int
    fats_g: int


class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail]
    daily_totals: Dict[str, int]


# --- Weekly & Monthly Variants ---

class WeeklyDietPlanResponse(BaseModel):
    plan_summary: Dict[str, str]
    weekly_plan: Dict[str, DailyPlan]  # keys: Monday..Sunday
    general_tips: List[str]


class MonthlyDietPlanResponse(BaseModel):
    plan_summary: Dict[str, str]
    weekly_plan: Dict[str, Dict[str, DailyPlan]]  # keys: Week 1..4 → Day 1..7
    general_tips: List[str]


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


# --- Helpers ---

def fix_daily_plan(day_data: dict) -> dict:
    """Ensure each day's JSON has 'meals' and 'daily_totals' keys."""
    if "meals" not in day_data:
        meals = {}
        totals = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}
        for meal_name, meal in day_data.items():
            if isinstance(meal, dict) and all(k in meal for k in ["calories", "protein_g", "carbs_g", "fats_g"]):
                meals[meal_name] = meal
                totals["calories"] += meal["calories"]
                totals["protein_g"] += meal["protein_g"]
                totals["carbs_g"] += meal["carbs_g"]
                totals["fats_g"] += meal["fats_g"]
        return {"meals": meals, "daily_totals": totals}
    return day_data


def create_diet_prompt(cuisine: str, language: str, time_span: str, current_weight: int, target_weight: int) -> str:
    ts = time_span.strip().lower()
    is_monthly = ts.startswith("month") or "30" in ts or "30-day" in ts or "30 day" in ts

    header = f"""You are an expert nutritionist generating a diet plan.

**CRITICAL RULES:**
1.  **CUISINE:** The plan MUST be based on **{cuisine}** food items.
2.  **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text outside the JSON.
3.  **STRICT SCHEMA:** The JSON object must have exactly three top-level keys: `plan_summary`, `weekly_plan`, and `general_tips`.
4.  **DATA TYPES:** All calorie and macronutrient values (`calories`, `protein_g`, `carbs_g`, `fats_g`) MUST be integers, NOT strings.
5.  Include fruits and juices in the diet.
6.  The diet should match the target (current {current_weight}kg → target {target_weight}kg).
7.  Make the plan in {language} but use native terms (e.g., Bread for Roti).
8.  Food items must strictly belong to that cuisine.
9.  If target is weightloss → calorie deficit. If gain → surplus.
10. In `general_tips` add a tip to follow the plan strictly.
11. ⚠️ Each **Day** MUST have this exact structure:
    {{
      "meals": {{
         "Breakfast": {{...}},
         "Lunch": {{...}},
         "Snack": {{...}},
         "Dinner": {{...}}
      }},
      "daily_totals": {{
         "calories": <int>,
         "protein_g": <int>,
         "carbs_g": <int>,
         "fats_g": <int>
      }}
    }}
"""

    if is_monthly:
        monthly_instr = """
ADDITIONAL RULES FOR MONTHLY (30-DAY PLAN):
- `weekly_plan` MUST have exactly 4 keys: "Week 1", "Week 2", "Week 3", "Week 4".
- Each week MUST contain exactly 7 keys: "Day 1", "Day 2", ... "Day 7".
- Each week must be different in meals.
"""
        return header + monthly_instr
    else:
        weekly_instr = """
ADDITIONAL RULES FOR WEEKLY PLAN:
- `weekly_plan` MUST have exactly 7 keys: "Monday", "Tuesday", ..., "Sunday".
"""
        return header + weekly_instr


@app.post("/generate-diet-plan", tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    try:
        ts = user_input.time_span.strip().lower()
        is_monthly = ts.startswith("month") or "30" in ts or "30-day" in ts or "30 day" in ts

        model_to_use = "gemini-2.5-flash" if is_monthly else "gemini-2.0-flash"
        print(f"Diet plan request for {user_input.cuisine}. Using {model_to_use} (time_span={user_input.time_span})")

        system_prompt = create_diet_prompt(
            user_input.cuisine,
            user_input.language,
            user_input.time_span,
            user_input.current_weight,
            user_input.target_weight
        )

        user_prompt_data = {
            "instruction": "Generate the diet plan JSON according to the system instructions.",
            "user_details": user_input.model_dump()
        }

        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

        response = await model.generate_content_async([json.dumps(user_prompt_data)])
        raw_json = json.loads(response.text)

        # --- Auto-fix every day ---
        if any(day in raw_json.get("weekly_plan", {}) for day in ["Monday", "Tuesday", "Sunday"]):
            # Weekly
            for day_name, day_data in raw_json["weekly_plan"].items():
                raw_json["weekly_plan"][day_name] = fix_daily_plan(day_data)
            return WeeklyDietPlanResponse(**raw_json)
        else:
            # Monthly
            for week_name, week_data in raw_json["weekly_plan"].items():
                for day_name, day_data in week_data.items():
                    raw_json["weekly_plan"][week_name][day_name] = fix_daily_plan(day_data)
            return MonthlyDietPlanResponse(**raw_json)

    except Exception as e:
        print(f"Error in /generate-diet-plan: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


# --- Workout Split ---

def create_workout_prompt(request: WorkoutRequest) -> str:
    return f"""You are an expert fitness coach. Generate a detailed, day-wise workout split based on:
- Days per Week: {request.days_per_week}
- Experience Level: {request.experience_level}
- Goal: {request.goal}
{('- Specific Focus: ' + request.focus) if request.focus else ''}

INSTRUCTIONS:
1. Return a valid JSON object with root key: "workout_plan".
2. "workout_plan" is an array of objects, one for each day (including rest days).
3. Each day has keys: "day", "focus", "exercises".
4. Each exercise has keys: "name", "sets", "reps".
5. Use only these exercises: {list(exercises_names)}.
6. Do not add text outside JSON.
"""


@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    try:
        prompt = create_workout_prompt(request)
        model_to_use = "gemini-2.5-flash" if request.is_premium else "gemini-2.0-flash"
        print(f"Workout split request. Using model: {model_to_use}")

        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

        response = await model.generate_content_async(prompt)
        workout_json = json.loads(response.text)

        for day_plan in workout_json["workout_plan"]:
            for exercise in day_plan["exercises"]:
                if "sets" in exercise:
                    exercise["sets"] = str(exercise["sets"])
                if "reps" in exercise:
                    exercise["reps"] = str(exercise["reps"])
                name = exercise["name"].lower().strip()
                if name in exercises_dict:
                    exercise["url"] = exercises_dict.get(name)

        return workout_json

    except Exception as e:
        print(f"Error in /generate-workout-split: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


@app.get("/", tags=["Status"])
def read_root():
    return {"status": "AI Fitness Coach API is running. Go to /docs for all endpoints."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
