from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import os
import tempfile
from google import genai
from google.genai import types
from PIL import Image
import io
import re
import html
import uuid
import groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Use a secure secret key for session management
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY", "default-secret-key"
)  # Replace with a real secret key in production

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get from environment variable
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize chatbot instances storage
chatbot_instances = {}
DEFAULT_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_SYSTEM_INSTRUCTION = (
    nutrition_assistant_prompt
) = """
You are very intelligent AI assistant that knows about nutrition values of veg pulao, chicken curry, dal, roti, veg fried rice, paneer butter masala, mixed veg curry, fish fry, biryani, sweet lsi, and fruit salad.
Now based on the target calories entered by users you will recommend meal to him/her.
Make it sensible, for example, if you are recommending paneer or dal, recommend roti or rice with it.
Provide your responses as plain text without any markdown formatting.
"""

# Sample menu items for canteen demand prediction
menu_items = {
    "Veg Pulao": {"rice": 0.15, "vegetables": 0.1, "spices": 0.02},
    "Chicken Curry": {"chicken": 0.2, "spices": 0.02, "onions": 0.05},
    "Dal": {"lentils": 0.1, "spices": 0.01, "onions": 0.02},
    "Roti": {"flour": 0.05, "oil": 0.005},
    "Veg Fried Rice": {"rice": 0.15, "vegetables": 0.1, "oil": 0.02},
    "Paneer Butter Masala": {
        "paneer": 0.15,
        "butter": 0.03,
        "spices": 0.02,
        "onions": 0.03,
    },
    "Mixed Veg Curry": {"vegetables": 0.2, "spices": 0.02, "onions": 0.05},
    "Fish Fry": {"fish": 0.2, "flour": 0.03, "oil": 0.05, "spices": 0.01},
    "Biryani": {"rice": 0.15, "chicken": 0.15, "spices": 0.03, "onions": 0.05},
    "Sweet Lassi": {"yogurt": 0.2, "sugar": 0.03},
    "Fruit Salad": {"fruits": 0.2, "sugar": 0.01},
}

# Load scholarship data
SCHOLARSHIP_DATA_PATH = "scholarship.json"

def load_scholarship_data():
    try:
        with open(SCHOLARSHIP_DATA_PATH) as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading scholarship data: {e}")
        return None

# Function to sanitize text content
def sanitize_text(text):
    # Remove markdown formatting and sanitize HTML
    text = re.sub(r'[`*_~#]', '', text)  # Remove markdown symbols
    text = html.escape(text)  # Escape HTML characters
    return text

# Function to clean up scholarship recommendations
def clean_scholarship_recommendations(text):
    # Clean the text from markdown formatting
    text = sanitize_text(text)
    
    # Try to extract only the scholarship names
    lines = text.strip().split('\n')
    scholarships = []
    
    # Look for numbered or bulleted lists
    pattern = r'^[0-9\.\-\*•]+\s*(.*?)(:.*)?$'
    
    for line in lines:
        line = line.strip()
        if line:
            # Check if line starts with number or bullet
            match = re.match(pattern, line)
            if match:
                scholarship_name = match.group(1).strip()
                scholarships.append({"name": scholarship_name})
            elif "scholarship" in line.lower():
                # If no bullet but contains "scholarship", consider it a scholarship name
                scholarships.append({"name": line})
    
    # If we couldn't extract structured data, get all non-empty lines
    if not scholarships:
        scholarships = [{"name": line.strip()} for line in lines if line.strip()]
    
    # Limit to top 5
    return scholarships[:5]

# Function to parse recommended scholarships from the text and match with scholarship data
def parse_scholarships(recommendations_text, scholarship_data):
    try:
        # First sanitize the text to remove markdown
        recommendations_text = sanitize_text(recommendations_text)
        
        # First try direct extraction
        if not scholarship_data:
            return clean_scholarship_recommendations(recommendations_text)
        
        # Clean the recommendations text
        scholarship_names = []
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract scholarship name using regex
            match = re.search(r'[0-9\.\-\*•]*\s*(.*?)(:|$)', line)
            if match:
                name = match.group(1).strip()
                scholarship_names.append(name)
        
        # Match extracted names with scholarship data
        matched_scholarships = []
        all_scholarships = scholarship_data.get('scholarships', [])
        
        for name in scholarship_names:
            # Try to find exact or partial match in scholarship data
            for scholarship in all_scholarships:
                if name.lower() in scholarship['name'].lower() or scholarship['name'].lower() in name.lower():
                    matched_scholarships.append({
                        "name": sanitize_text(scholarship['name']),
                        "description": sanitize_text(scholarship.get('description', '')),
                        "amount": scholarship.get('amount', 0)
                    })
                    break
            else:
                # If no match found, just add the name
                matched_scholarships.append({"name": name})
        
        # If we couldn't match any scholarships, return clean text
        if not matched_scholarships:
            return clean_scholarship_recommendations(recommendations_text)
        
        return matched_scholarships[:5]  # Limit to top 5
    except Exception as e:
        app.logger.error(f"Error parsing scholarships: {e}")
        return clean_scholarship_recommendations(recommendations_text)

# Extraction prompt template
EXTRACTION_PROMPT = """
Extract all relevant scholarship information from this resume. Focus on:
1. Education level (BTech or postgrad)
2. CGPA or percentage (convert percentage to CGPA if needed, using 10-point scale)
3. Number of backlogs (academic failures/repeats)
4. Entrance exam scores (JEE/GATE/etc., convert to percentile if needed)
5. Number of publications or research papers
6. Work experience in years
7. Family income (estimate based on background if not explicitly stated)
8. Projects completed (count and brief descriptions)
9. Extracurricular activities and achievements
Format your response as a clean JSON object with these fields:
{
  "degree": "btech/postgrad",
  "cgpa": "X.X",
  "backlog": "X",
  "entrance_score": "X",
  "publications": "X",
  "work_experience": "X",
  "family_income": "XXXXX",
  "projects": "brief summary",
  "achievements": "brief summary"
}
For any missing information, make reasonable estimates based on the available context and include a "confidence" field for each estimate (high/medium/low).
"""

# ----- GROQ CHATBOT CLASS -----
class GroqChatbot:
    def __init__(self, api_key, model="llama3-70b-8192", system_instruction=None):
        """
        Initialize a conversational chatbot using Groq's API

        Args:
            api_key (str): Groq API key
            model (str): Model to use, default is "llama3-70b-8192"
            system_instruction (str): Optional system instruction to define the bot's behavior
        """
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.system_instruction = system_instruction
        self.conversation_history = []

        # Add system message to history if provided
        if self.system_instruction:
            # Add instruction to avoid markdown
            enhanced_instruction = system_instruction
            if not "no markdown" in enhanced_instruction.lower():
                enhanced_instruction += " Provide your responses as plain text without any markdown formatting."

            self.conversation_history.append(
                {"role": "system", "content": enhanced_instruction}
            )

    def chat(self, message):
        """
        Send a message to the chatbot and get a response

        Args:
            message (str): User message

        Returns:
            str: Bot's response as plain text
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        try:
            # Create completion request with the entire conversation history
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1024,
            )

            # Extract assistant response
            assistant_message = response.choices[0].message.content

            # Remove any markdown formatting
            plain_text = self.remove_markdown(assistant_message)

            # Add assistant response to conversation history (original response)
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )

            return plain_text

        except Exception as e:
            return f"Error: {str(e)}"

    def remove_markdown(self, text):
        """
        Remove common markdown formatting from text

        Args:
            text (str): Text with potential markdown

        Returns:
            str: Plain text without markdown
        """
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)

        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove bullet points
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

        # Remove numbered lists
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Fix any double spacing that might have been created
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

        return text

    def get_conversation_history(self):
        """
        Get the current conversation history

        Returns:
            list: List of conversation messages
        """
        return self.conversation_history

    def reset_conversation(self):
        """
        Reset the conversation history, keeping only the system instruction if it exists
        """
        if self.system_instruction:
            enhanced_instruction = self.system_instruction
            if not "no markdown" in enhanced_instruction.lower():
                enhanced_instruction += " Provide your responses as plain text without any markdown formatting."

            self.conversation_history = [
                {"role": "system", "content": enhanced_instruction}
            ]
        else:
            self.conversation_history = []

        return "Conversation has been reset."


# ----- CANTEEN DEMAND PREDICTION FUNCTION -----
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
        "fest": "College festival with maximum attendance and festivities",
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
       
    3. Return only the menu items and the quantities to prepare.
    
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
            quantity_match = re.search(r"(\d+)", quantity_str)
            if quantity_match:
                quantity = int(quantity_match.group(1))
                predictions[item] = quantity

    return predictions


# ----- API ROUTES -----
# Default route to serve the chatbot interface
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Route to serve the canteen demand prediction interface
@app.route("/canteen")
def canteen():
    return render_template("canteen.html")

# API routes for canteen demand prediction
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        day_type = data.get("day_type")
        inventory = data.get("inventory")

        if not day_type or not inventory:
            return (
                jsonify(
                    {"error": "Missing required fields: 'day_type' or 'inventory'"}
                ),
                400,
            )

        # Get API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                jsonify({"error": "Gemini API key not found in environment variables"}),
                500,
            )

        predictions = predict_canteen_demand_with_gemini(
            day_type, inventory, menu_items, api_key
        )
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/menu", methods=["GET"])
def get_menu():
    return jsonify({"menu_items": menu_items}), 200

# API routes for chatbot
@app.route("/api/chatbot/create", methods=["POST"])
def create_chatbot():
    """Create a new chatbot instance"""
    data = request.get_json()

    # Get parameters from request or use defaults
    api_key = data.get("api_key", DEFAULT_API_KEY)
    model = data.get("model", DEFAULT_MODEL)
    system_instruction = data.get("system_instruction", DEFAULT_SYSTEM_INSTRUCTION)

    if not api_key:
        return jsonify({"error": "Groq API key is required"}), 400

    # Create a unique ID for this chatbot instance
    chatbot_id = str(uuid.uuid4())

    # Create the chatbot
    chatbot = GroqChatbot(
        api_key=api_key, model=model, system_instruction=system_instruction
    )

    # Store the chatbot instance
    chatbot_instances[chatbot_id] = chatbot
    print(chatbot_instances)

    # Return the ID to the client
    return jsonify(
        {
            "chatbot_id": chatbot_id,
            "model": model,
            "message": "Chatbot created successfully",
        }
    )

@app.route("/api/chatbot/<chatbot_id>/chat", methods=["POST"])
def send_message(chatbot_id):
    """Send a message to a specific chatbot instance"""
    # Check if the chatbot exists
    if chatbot_id not in chatbot_instances:
        return jsonify({"error": "Chatbot not found"}), 404

    # Get the message from the request
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Get the chatbot
    chatbot = chatbot_instances[chatbot_id]

    # Send the message and get the response
    response = chatbot.chat(message)

    # Return the response
    return jsonify({"response": response})

@app.route("/api/chatbot/<chatbot_id>/history", methods=["GET"])
def get_history(chatbot_id):
    """Get the conversation history for a specific chatbot"""
    # Check if the chatbot exists
    if chatbot_id not in chatbot_instances:
        return jsonify({"error": "Chatbot not found"}), 404

    # Get the chatbot
    chatbot = chatbot_instances[chatbot_id]

    # Get the conversation history
    history = chatbot.get_conversation_history()

    # Return the history
    return jsonify({"history": history})

@app.route("/api/chatbot/<chatbot_id>/reset", methods=["POST"])
def reset_conversation(chatbot_id):
    """Reset the conversation for a specific chatbot"""
    # Check if the chatbot exists
    if chatbot_id not in chatbot_instances:
        return jsonify({"error": "Chatbot not found"}), 404

    # Get the chatbot
    chatbot = chatbot_instances[chatbot_id]

    # Reset the conversation
    message = chatbot.reset_conversation()

    # Return success message
    return jsonify({"message": message})

@app.route("/api/chatbot/<chatbot_id>/export", methods=["GET"])
def export_history(chatbot_id):
    """Export the conversation history for a specific chatbot as JSON"""
    # Check if the chatbot exists
    if chatbot_id not in chatbot_instances:
        return jsonify({"error": "Chatbot not found"}), 404

    # Get the chatbot
    chatbot = chatbot_instances[chatbot_id]

    # Get the conversation history
    history = chatbot.get_conversation_history()

    # Create a response with the JSON data
    response = app.response_class(
        response=json.dumps(history, indent=2),
        status=200,
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{chatbot_id}.json"
        },
    )

    return response

@app.route("/api/chatbot/<chatbot_id>/delete", methods=["DELETE"])
def delete_chatbot(chatbot_id):
    """Delete a chatbot instance"""
    # Check if the chatbot exists
    if chatbot_id not in chatbot_instances:
        return jsonify({"error": "Chatbot not found"}), 404

    # Delete the chatbot
    del chatbot_instances[chatbot_id]

    # Return success message
    return jsonify({"message": "Chatbot deleted successfully"})

@app.route('/api/process-resume', methods=['POST'])
def process_resume():
    """Process resume and recommend scholarships in one API call"""
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()
        
        # Handle as image directly (no PDF conversion)
        image = Image.open(temp_file.name)
        
        # Load scholarship data
        scholarship_data = load_scholarship_data()
        
        # Extract resume data using Gemini (but don't return it)
        extract_response = genai.GenerativeModel('gemini-pro-vision').generate_content(
            [EXTRACTION_PROMPT, image]
        )
        
        resume_data = extract_response.text
        
        # Generate scholarship recommendations
        recommend_prompt = f"Based on the extracted resume data {resume_data} tell about which scholarship person should apply from this json file {scholarship_data} just written top 5 scholarships name in plain text format (not markdown)"
        
        recommend_response = genai.GenerativeModel('gemini-pro').generate_content(recommend_prompt)
        
        recommendations_text = recommend_response.text
        
        # Parse scholarships from recommendations
        scholarships = parse_scholarships(recommendations_text, scholarship_data)
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # ONLY return scholarships information - no resume data at all
        return jsonify({
            "scholarships": scholarships,
            "recommendations": sanitize_text(recommendations_text)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing resume: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)