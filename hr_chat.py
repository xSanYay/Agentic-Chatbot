"""
Agentic-Chatbot Public
----------------------
Agentic Chatbot - leverages Gemini's function calling to autonomously orchestrate task execution,
dynamically interfacing with external systems for context-aware, multi-step interactions.

This script implements a HR leave-management chatbot that accepts terminal input/output.
It demonstrates how to integrate function calling into the conversation with the Gemini model.
"""

# --- Import Required Libraries ---
import google.generativeai as genai           # Gemini Generative AI API
from google.api_core import retry             # To add retry mechanism for API calls
from datetime import datetime                 # For handling dates
from typing import List, Dict                 # For type hints

# --- Configure Google Generative AI API Key ---
# Replace with your API key which you get form gemini ai studi for FREE !!!

genai.configure(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxx")

# --- Initialize Global Variables ---
# Capture the current date/time for use in prompts and leave requests.

current_datetime = datetime.now()


# --- Define Core Leave Management Data and Functions ---
# Simulate employee leave balances.
leave_balances = {
    "emp1": 5,
    "emp2": 3,
    "emp3": 0,
}


# Simulate a list of leave requests.
leave_requests: List[Dict[str, str]] = [
    {"employee": "emp1", "date": "2024-03-01", "reason": "Sick"},
    {"employee": "emp2", "date": "2024-01-15", "reason": "Vacation"},
]

def request_leave(employee: str, date: str, reason: str) -> None:
    """
    Append a new leave request for an employee.
    This function simulates the submission of a leave request.
    """
    leave_requests.append({"employee": employee, "date": date, "reason": reason})

def process_leave_request(employee: str, date: str, reason: str) -> str:
    """
    Process a leave request by building a query for Gemini.
    If the employee has a positive leave balance, approve the leave and deduct one leave day.
    """
    user_query = (f"Employee {employee} wants to take leave on {date} for {reason}. "
                  "Approve it by yourself immediately if leave balance > 0.")
    
    # Call Gemini to process the leave request (the function call is managed by Gemini's function calling)
    gemini_response = send_message_to_gemini(user_query)

    # Deduct one leave day from the employee's balance after approval.
    leave_balances[employee] -= 1
    return gemini_response.text

def get_leave_balance(employee: str) -> int:
    """
    Return the current leave balance for the given employee.
    """
    return leave_balances.get(employee, 0)

def get_leave_info(date: str) -> List[str]:
    """
    Retrieve a list of employees who have leave requests on the given date.
    Assumes the date is provided in the 'YYYY-MM-DD' format.
    """
    return [leave["employee"] for leave in leave_requests if leave["date"] == date]


# --- Define the HR Bot System Prompt ---
# This prompt instructs Gemini on how to behave and what tasks to handle.
HR_BOT_PROMPT = f"""
You are an HR chatbot responsible for managing employee leave requests. You have the following tasks:
1. Ask for the employee's name before starting.
2. Anticipate 3 different requests, of which you have to handle all:
    a. Employees asking for their leave balance.
    b. Employees requesting leave.
    c. Inquiries about who is on leave (all leaves or a specific day).
3. Check leave balances for employees.
4. Accept leave requests (date, reason) and confirm with employees. Do not ask for the number of days/year etc; keep it default to 2025.
5. Convert any provided dates to the format 2025-MM-DD.
6. Approve leave if leave balance > 0, and then deduct one leave from the employee's balance.
7. Confirm the entered details with the employee before finalizing the leave request.
8. Provide insightful responses and maintain a conversational tone even if the queries are deviating from the leave topic. Use general answering ability.
Relevant data: {leave_balances} and {leave_requests}
The date today is {current_datetime.strftime("%Y-%m-%d")}.

"""

# --- Prepare the List of Tools for Function Calling ---
# These functions are provided as callable tools for Gemini.
tools = [request_leave, process_leave_request, get_leave_balance, get_leave_info]

# --- Initialize the Gemini Model with Function Calling Enabled ---
model_name = "gemini-1.5-flash"  # Replace with your Gemini model name if different
model = genai.GenerativeModel(
    model_name,
    tools=tools,
    system_instruction=HR_BOT_PROMPT
)

# Start the conversation with automatic function calling enabled.
convo = model.start_chat(enable_automatic_function_calling=True)


# --- Define a Function to Send Messages to Gemini with Retry ---
@retry.Retry(initial=30)
def send_message_to_gemini(message: str):
    """
    Send a message to Gemini and return the response.
    The retry decorator will retry the call if transient errors occur.
    """
    return convo.send_message(message)



# --- Main Chatbot Loop for Terminal Interaction ---

while True:
    user_query = input("You: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    try:
        # Send the user's message to Gemini and get the response.
        gemini_response = send_message_to_gemini(user_query)
        bot_response = gemini_response.text
    except Exception as error:
        bot_response = f"Error: {error}"
    
    print(f"Bot: {bot_response}")


"""
Any queries or feedback, please reach out to someone or use gemini decumentation
All other sources(like ChatGPT) will misdirect you."""
