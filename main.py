import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

url_list = [
    "https://www.mayoclinic.org/diseases-conditions/flu/symptoms-causes/syc-20351719",
    "https://www.cdc.gov/flu/signs-symptoms/?CDC_AAref_Val=https://www.cdc.gov/flu/symptoms/index.html",
    "https://www.rescuemycat.org/p/what-you-can-do-on-your-own.html",
    "https://theonion.com/u-s-cancels-bird-flu-vaccine/"
]

def scrape_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = " ".join(p.get_text() for p in soup.find_all('p'))
    return text[:4000]  # Truncate to fit model limits

# -----------------------------------------------------------------------------------------------------------------
#SAFEGUARD 1: GPT checks if the source fits the domain -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def is_medical_related(text):
    check_prompt = f"""Is the following text medically related? Answer only "Yes" or "No".

Text:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": check_prompt}]
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")

# -----------------------------------------------------------------------------------------------------------------
#SAFEGUARD 2: GPT checks to see if the source looks like it is reliable -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def is_reputable_source(text):
    source_prompt = f"""Does the following text look like it is from a reputable source? Answer only "Yes" or "No".

Text:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": source_prompt}]
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


# -----------------------------------------------------------------------------------------------------------------
# Generates the instruction-response pairs for LLM Training -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def generate_instruction_response(text):
    prompt = f"""Based on the following article, generate 30 instruction-response pairs a hospital assistant might provide to a patient.

Article:
{text}

Return the output in this exact format:

Instruction: <instruction>
Response: <response>

Repeat 30 times without any numbering or markdown formatting.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------------------------------------------------------------------
# Process list of urls and create jsonl file for training -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

output_file = "medical_instruct_data.jsonl"
log_file = "unused_log.txt"

for url in url_list:
    content = scrape_text(url)

    if not is_medical_related(content):
        print(f"🧹 Skipping non-medical content from: {url}")
        with open(log_file, "a") as log:
            log.write(f"{url}\n")
        continue
    if not is_reputable_source(content):
        print(f"🧹 Skipping non-reputable content from: {url}")
        with open(log_file, "a") as log:
            log.write(f"{url}\n")
        continue

    result = generate_instruction_response(content)
    print(result)

    with open(output_file, "a", encoding="utf-8") as f:
        for block in result.split("Instruction: ")[1:]:
            try:
                instr, resp = block.split("Response:")
                entry = {
                    "instruction": instr.strip(),
                    "input": "",
                    "output": resp.strip()
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Skipped malformed block: {e}")

