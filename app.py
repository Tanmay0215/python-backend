from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import json
from google import genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Use a secure secret key for session management
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')  # Replace 'default-secret-key' in production

# Sample menu items
menu_items = {
    "Veg Pulao": {"rice": 0.15, "vegetables": 0.1, "spices": 0.02},
    "Chicken Curry": {"chicken": 0.2, "spices": 0.02, "onions": 0.05},
    "Dal": {"lentils": 0.1, "spices": 0.01, "onions": 0.02},
    "Roti": {"flour": 0.05, "oil": 0.005},
    "Veg Fried Rice": {"rice": 0.15, "vegetables": 0.1, "oil": 0.02},
    "Paneer Butter Masala": {"paneer": 0.15, "butter": 0.03, "spices": 0.02, "onions": 0.03},
    "Mixed Veg Curry": {"vegetables": 0.2, "spices": 0.02, "onions": 0.05},
    "Fish Fry": {"fish": 0.2, "flour": 0.03, "oil": 0.05, "spices": 0.01},
    "Biryani": {"rice": 0.15, "chicken": 0.15, "spices": 0.03, "onions": 0.05},
    "Sweet Lassi": {"yogurt": 0.2, "sugar": 0.03},
    "Fruit Salad": {"fruits": 0.2, "sugar": 0.01}
}

def predict_canteen_demand_with_gemini(day_type, inventory, menu_items, api_key):
    """
    Uses Gemini API to predict the quantity of different menu items to prepare
    based on day type and inventory.
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    
    # Define day type descriptions
    day_type_descriptions = {
        "regular": "Regular working day with normal student attendance",
        "exam": "Exam period with higher student attendance and stress levels",
        "weekend": "Weekend with lower attendance but more leisure time",
        "event_weekday": "Weekday with special events or interviews",
        "event_weekend": "Weekend with campus events or activities",
        "fest": "College festival with maximum attendance and festivities"
    }
    
    # Create prompt for Gemini
    prompt = f"""
    You are an AI canteen manager for a college campus. You need to predict how many portions of each menu item to prepare based on the day type and available inventory.

    CONTEXT:
    - Day type: {day_type} ({day_type_descriptions[day_type]})
    - Available inventory: {json.dumps(inventory, indent=2)}
    - Menu items with ingredient requirements: {json.dumps(menu_items, indent=2)}

    INSTRUCTIONS:
    1. Consider the characteristics of this type of day:
       - Regular days: Normal attendance, balanced veg/non-veg preference
       - Exam days: Higher attendance, slightly higher preference for veg food
       - Weekends: Lower attendance, higher preference for non-veg and desserts
       - Weekdays with events: High attendance, balanced preferences
       - Weekends with events: Moderate-high attendance, high dessert preference
       - Festivals: Maximum attendance, high demand for all items
       
    2. For each menu item:
       - Check if all required ingredients are available
       - Calculate how many portions you can make with available ingredients
       - Predict the appropriate number of portions based on day type and expected demand
       - Consider that not all students will eat every dish
       
    3. Return only the menu items and the recommended quantities to prepare.
    
    RESPONSE FORMAT:
    Return a simple list of menu items and the quantities to prepare, like:
    "Veg Pulao: 120 portions
    Chicken Curry: 90 portions
    ..."
    
    Only return the menu items and quantities, no additional explanations.
    """
    
    # Call the Gemini API using the client library
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    # Get the prediction text
    prediction_text = response.text
    
    # Parse the response into a dictionary
    predictions = {}
    for line in prediction_text.strip().split("\n"):
        if ":" in line:
            item, quantity_str = line.split(":", 1)
            item = item.strip()
            quantity_str = quantity_str.strip().lower()
            
            # Extract the number from strings like "120 portions"
            quantity_match = re.search(r'(\d+)', quantity_str)
            if quantity_match:
                quantity = int(quantity_match.group(1))
                predictions[item] = quantity
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        day_type = data.get('day_type')
        inventory = data.get('inventory')
        
        if not day_type or not inventory:
            return jsonify({"error": "Missing required fields: 'day_type' or 'inventory'"}), 400
        
        # Get API key from environment variables
        api_key = os.getenv('API_KEY')
        if not api_key:
            return jsonify({"error": "API key not found in environment variables"}), 500
        
        predictions = predict_canteen_demand_with_gemini(day_type, inventory, menu_items, api_key)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/menu', methods=['GET'])
def get_menu():
    return jsonify({"menu_items": menu_items}), 200

if __name__ == '__main__':
    # Use a production-ready WSGI server like Gunicorn to run the app
    app.run(host='0.0.0.0', port=5000, debug=False)