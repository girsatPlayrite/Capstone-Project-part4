import os
import re
import json
import joblib
import requests
import pandas as pd

from dotenv import load_dotenv



load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if api_key is None:
    raise ValueError("LLM_API_KEY environment variable not found.")



from jsonschema import validate
from sklearn.model_selection import train_test_split

#Load API key
api_key = os.environ.get("LLM_API_KEY")

if api_key is None:
    raise ValueError("LLM_API_KEY environment variable not found.")

#Reusable LLM Function
def call_llm(system_prompt,
             user_prompt,
             temperature=0.0,
             max_tokens=512):

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        print(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]
	
#Test Prompt
test = call_llm(
    "Reply with only the word: hello",
    ""
)

print(test)

#Load Best Model
model = joblib.load("best_model.pkl")

#Load the Dataset
df = pd.read_csv("cleaned_data.csv")

X = df.drop(columns=["Total day charge"])

y = (
    df["Total day charge"]
    >
    df["Total day charge"].median()
).astype(int)

X["Area code"] = X["Area code"].astype(str)

X = pd.get_dummies(
    X,
    columns=[
        "State",
        "Area code",
        "International plan",
        "Voice mail plan"
    ],
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Inputs
samples = X_test.iloc[:3].copy()

#PII Guardrail
PII_FIELDS = [
    "name",
    "email",
    "phone",
    "address"
]


def remove_pii(record):

    record = record.copy()

    for field in PII_FIELDS:

        if field in record:
            del record[field]

    return record
	

#JSON schema
schema = {

    "type": "object",

    "properties": {

        "prediction_label": {
            "type": "string"
        },

        "confidence_level": {
            "type": "string"
        },

        "top_reason": {
            "type": "string"
        },

        "second_reason": {
            "type": "string"
        },

        "next_step": {
            "type": "string"
        }

    },

    "required": [

        "prediction_label",

        "confidence_level",

        "top_reason",

        "second_reason",

        "next_step"

    ]
}

#Call LLM
system_prompt = """
You explain machine learning predictions.

Return ONLY valid JSON.

Required fields:
- prediction_label (string)
- confidence_level (must be one of: "low", "medium", "high")
- top_reason (string)
- second_reason (string)
- next_step (string)

Do not return numeric confidence values.
"""

for i in range(3):

    features = remove_pii(
        samples.iloc[i].to_dict()
    )

    prediction = int(
        model.predict(
            samples.iloc[[i]]
        )[0]
    )

    probability = float(
        model.predict_proba(
            samples.iloc[[i]]
        )[0][1]
    )

    user_prompt = json.dumps({

        "features": features,

        "predicted_class": prediction,

        "predicted_probability": probability

    })

    response = call_llm(
        system_prompt,
        user_prompt,
        temperature=0.0,
    )

    print("\nRaw Response")

    print(response)

if response is not None:

    parsed = json.loads(response)

    validate(
        instance=parsed,
        schema=schema
    )

    print("Schema Validation Passed")

else:
    print("LLM response failed")

#README: Before each LLM request is made, the input record is screened for personally identifiable information (PII). Any fields containing sensitive details—such as name, email, phone number, or address—are stripped out prior to the API call. This ensures that sensitive information is never transmitted to the LLM, while still preserving the features necessary for generating a meaningful prediction explanation.



#Track C System Prompt (Zero-shot)

system_prompt = """
You are a machine learning prediction explanation assistant.

Your role is to explain model predictions using the provided feature values,
predicted class, and predicted probability.

Output ONLY valid JSON.

The JSON must contain exactly these fields:

{
  "prediction_label": "string",
  "confidence_level": "low|medium|high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}

Rules:
- Do not include markdown.
- Do not include explanations outside JSON.
- Do not invent missing feature values.
- Do not include personally identifiable information.
- Base explanations only on the provided model prediction information.
"""

samples = X_test.iloc[:3].copy()


temperature_results = []


for i in range(3):

    features = samples.iloc[i].to_dict()

    prediction = int(
        model.predict(
            samples.iloc[[i]]
        )[0]
    )

    probability = float(
        model.predict_proba(
            samples.iloc[[i]]
        )[0][1]
    )


    user_prompt = f"""
Explain this prediction.

Feature values:
{features}

Predicted class:
{prediction}

Predicted probability:
{probability}

Return only JSON.
"""


    # Temperature 0

    output_temp0 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0
    )


    # Temperature 0.7

    output_temp07 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0.7
    )


    temperature_results.append(
        [
            f"Input {i+1}",
            output_temp0,
            output_temp07
        ]
    )


comparison_df = pd.DataFrame(
    temperature_results,
    columns=[
        "Input",
        "Output at temp=0",
        "Output at temp=0.7"
    ]
)


print(comparison_df)

#README: Temperature governs the degree of randomness in token selection during generation. Setting temperature to 0 makes the model effectively deterministic, since it always picks the highest-probability next token—an approach well suited to structured JSON generation, where consistent formatting matters most. A temperature of 0.7, by contrast, introduces more variability by allowing the model to sample from a broader range of candidate tokens. This can yield more diverse explanations, but it may also cause inconsistencies in wording, reasoning order, or formatting from one run to the next. For prediction explanation pipelines specifically, a temperature of 0 is the better choice, since reproducibility and strict adherence to the expected schema outweigh the benefits of creative variation.

import json
from jsonschema import validate, ValidationError

#JSON Schema
explanation_schema = {

    "type": "object",

    "properties": {

        "prediction_label": {
            "type": "string"
        },

        "confidence_level": {
            "type": "string"
        },

        "top_reason": {
            "type": "string"
        },

        "second_reason": {
            "type": "string"
        },

        "next_step": {
            "type": "string"
        }

    },

    "required": [

        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"

    ]

}


#Fallback Function
def fallback_response():

    return {

        "prediction_label": None,

        "confidence_level": None,

        "top_reason": None,

        "second_reason": None,

        "next_step": None

    }
	
#JSON Parsing + Schema Validation Function

def validate_llm_response(response):

    try:

        #Remove extra whitespace
        cleaned_response = response.strip()

        #Convert text to JSON
        parsed_json = json.loads(cleaned_response)


    except json.JSONDecodeError as e:

        print("JSON Parsing Error:")
        print(e)

        return fallback_response()


    try:

        #Validate JSON structure
        validate(
            instance=parsed_json,
            schema=explanation_schema
        )

        print("Validation Status: PASS")

        return parsed_json


    except ValidationError as e:

        print("Schema Validation Error:")
        print(e.message)

        return fallback_response()
		
#Load Model
import joblib

best_model = joblib.load(
    "best_model.pkl"
)

#Create Three Hand-Crafted Inputs
test_inputs = [

    X_test.iloc[0].to_dict(),

    X_test.iloc[1].to_dict(),

    X_test.iloc[2].to_dict()

]

#Run Track C Pipeline

results = []


for features in test_inputs:


    #Convert dictionary into dataframe row

    input_df = pd.DataFrame(
        [features]
    )


    #Prediction

    prediction = int(
        best_model.predict(input_df)[0]
    )


    #Probability

    probability = float(
        best_model.predict_proba(input_df)[0][1]
    )


    #User prompt

    user_prompt = f"""

Feature values:

{features}


Predicted class:

{prediction}


Predicted probability:

{probability}


Return only JSON.
"""


    #LLM call

    response = safe_call_llm(
    system_prompt,
    user_prompt,
    temperature=0
)


    print("\n======================")
    print("Feature Input")
    print(features)

    print("\nPredicted Class:")
    print(prediction)

    print("\nProbability:")
    print(probability)


    print("\nRaw LLM Response:")
    print(response)


    #Validate

    explanation = validate_llm_response(
        response
    )


    print("\nFinal Explanation:")
    print(explanation)


    results.append({

        "Feature Input": features,

        "Predicted Class": prediction,

        "Probability": probability,

        "Explanation": explanation

    })
	
#Display Results Table
results_df = pd.DataFrame(results)

print(results_df)



#Create PII Detection Function

def has_pii(text):

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(
        re.search(email_pattern, text)
        or
        re.search(phone_pattern, text)
    )

#Create Safe LLM Wrapper

def safe_call_llm(system_prompt, user_input, temperature=0):

    if has_pii(user_input):

        print("Input blocked: PII detected.")

        return None


    return call_llm(
        system_prompt,
        user_input,
        temperature=temperature
    )

pii_test_input = """
Customer information:
Name: Muni Karpu
Email: munikarupu@gmail.com

Explain this prediction.
"""

response = safe_call_llm(
    system_prompt,
    pii_test_input,
    temperature=0
)

print(response)

safe_test_input = """
Feature values:
Total day minutes: 250
Customer service calls: 4
International plan: Yes

Predicted class:
1

Predicted probability:
0.82

Return JSON explanation.
"""


response = safe_call_llm(
    system_prompt,
    safe_test_input,
    temperature=0
)

print(response)

#README: Before any LLM API call is made, the user prompt is scanned using regular expressions to detect common personally identifiable information patterns—specifically email addresses and phone numbers. If PII is found, the request is blocked outright and the LLM API is never called. This guardrail prevents sensitive customer information from being accidentally transmitted to external services.



# Track C Final 3 Input Demonstration


import pandas as pd


# Three feature-vector inputs
test_inputs = [

    X_test.iloc[0].to_dict(),

    X_test.iloc[1].to_dict(),

    X_test.iloc[2].to_dict()

]


demo_results = []


for index, features in enumerate(test_inputs, start=1):

    print("\n==============================")
    print(f"INPUT {index}")
    print("==============================")

    print(features)


    # Convert dictionary into dataframe
    input_df = pd.DataFrame([features])


    # Prediction
    prediction = int(
        best_model.predict(input_df)[0]
    )


    # Prediction probability
    probability = float(
        best_model.predict_proba(input_df)[0][1]
    )


    print("\nPredicted Class:")
    print(prediction)

    print("\nProbability:")
    print(probability)



    # User prompt sent to LLM

    user_prompt = f"""
Explain this machine learning prediction.

Feature values:
{features}

Predicted class:
{prediction}

Predicted probability:
{probability}

Return only JSON.
"""


    # PII Guardrail + LLM call

    response = safe_call_llm(
        system_prompt,
        user_prompt,
        temperature=0
    )


    print("\nRaw LLM Response:")
    print(response)



    # Validation

    if response is None:

        validation_status = "BLOCKED - PII detected"

        explanation = fallback_response()


    else:

        explanation = validate_llm_response(response)


        if explanation["prediction_label"] is None:

            validation_status = "FAIL - Invalid JSON"

        else:

            validation_status = "PASS"



    print("\nValidation:")
    print(validation_status)



    demo_results.append({

        "Input": features,

        "Predicted Class": prediction,

        "Probability": probability,

        "LLM Output": explanation,

        "Validation": validation_status

    })



# Convert results into table

results_table = pd.DataFrame(demo_results)


print("\n\nFINAL RESULTS TABLE")
print(results_table)