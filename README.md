# Capstone-Project-part1
Data Acquisition, Cleaning, and Exploratory Analysis

Before each LLM request is made, the input record is screened for personally identifiable information (PII). Any fields containing sensitive details—such as name, email, phone number, or address—are stripped out prior to the API call. This ensures that sensitive information is never transmitted to the LLM, while still preserving the features necessary for generating a meaningful prediction explanation.

Temperature governs the degree of randomness in token selection during generation. Setting temperature to 0 makes the model effectively deterministic, since it always picks the highest-probability next token—an approach well suited to structured JSON generation, where consistent formatting matters most. A temperature of 0.7, by contrast, introduces more variability by allowing the model to sample from a broader range of candidate tokens. This can yield more diverse explanations, but it may also cause inconsistencies in wording, reasoning order, or formatting from one run to the next. For prediction explanation pipelines specifically, a temperature of 0 is the better choice, since reproducibility and strict adherence to the expected schema outweigh the benefits of creative variation.

Before any LLM API call is made, the user prompt is scanned using regular expressions to detect common personally identifiable information patterns—specifically email addresses and phone numbers. If PII is found, the request is blocked outright and the LLM API is never called. This guardrail prevents sensitive customer information from being accidentally transmitted to external services.

Track C Demonstration Results

| Input | LLM Output | Valid JSON | Pass/Block Guardrail |
|---|---|---|---|
| Feature Vector 1 | {"prediction_label":"Churn","confidence_level":"high","top_reason":"High usage pattern","second_reason":"Service behavior indicates risk","next_step":"Contact customer"} | PASS | PASS |
| Feature Vector 2 | {"prediction_label":"No Churn","confidence_level":"medium","top_reason":"Low risk usage pattern","second_reason":"Stable account behavior","next_step":"Continue monitoring"} | PASS | PASS |
| Feature Vector 3 | {"prediction_label":"Churn","confidence_level":"medium","top_reason":"International usage detected","second_reason":"Customer activity pattern","next_step":"Offer retention plan"} | PASS | PASS |
