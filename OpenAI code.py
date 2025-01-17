from openai import OpenAI
import pandas as pd
import openpyxl

# Load training data
df = pd.read_csv('train.csv')
df2 = pd.read_csv(r"Datasets/psc_severity_test.csv")
workbook = openpyxl.load_workbook(r"Datasets/submission_template.xlsx")
sheet = workbook.active

# Define OpenAI API key
client = OpenAI(
  api_key="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"
)

# Define a function to classify severity
def classify_severity(deficiency_text):
    prompt = "You are an expert in classifying deficiencies based on severity levels. " + \
        "\n Here are some examples:" + \
            "\n 1. " + df.iloc[0]['def_text'] + " -> " + df.iloc[0]['Consensus Severity'] + \
                "\n 2. " + df.iloc[1]['def_text'] + " -> " + df.iloc[1]['Consensus Severity'] + \
                    "\n 3. " + df.iloc[2]['def_text'] + " -> " + df.iloc[2]['Consensus Severity'] + \
                        "Based on the above examples, classify the following deficiency: " + \
                            deficiency_text + "in one word: Low, Medium, High"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return severity from the response
    return completion.choices[0].message.content

row_count = len(df)
for i in range(row_count):
    # Test the function with a new description
    new_description = df2.iloc[i]['def_text']
    severity = classify_severity(new_description)
    print(f"Predicted Severity: {severity}")
    if i % 100 == 0:
        print(i)
    sheet.insert_cols(3)
    sheet["C2"] = df2.iloc[i - 1]['PscInspectionId']
    sheet["C3"] = df2.iloc[i - 1]['deficiency_code']
    sheet["C4"] = severity

workbook.save(r"Datasets/submission_template.xlsx")

# Test the function with a new description
new_description = "Description of a new deficiency that involves critical issues."
severity = classify_severity(new_description)
print(f"Predicted Severity: {severity}")
