import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from text_generation import Client
from io import StringIO


GREEN = '\033[38;5;35m'
RED = '\033[38;5;196m'
ORANGE = '\033[38;5;215m'
YELLOW = '\033[38;5;229m'
BLUE = '\033[0;34m'
PURPLE = '\033[38;5;90m'
RESET = '\033[0m'
COLOR = YELLOW

endpoint_url = "CLIENT_URL"
client = Client(endpoint_url, timeout = 60)

data_input = ""
df = pd.read_csv('use_cases_subset.csv')
for index, row in df.iterrows():
    if (pd.isna(row.use_case) == False): data_input += "\nUse Case: " + row.use_case
    data_input += "\n"



# Runs 'helix' of 3 agents
# Begins with an actor who provides an original categorization based on the data
# Followed by a critic who responds with critiques of the actor's response
# Ending with a mediator who considers the data along with both the actor and critic's responses to come to a final categorization
def run_helix():
    agent1response = ""
    for response in client.generate_stream("<s> [INST] You are defining categories of LLM use cases. Can you find ~5-10 unique categories based off of these use case descriptions?\n" + data_input + "Please make it consise - do not repeat information. [/INST]", max_new_tokens=400, stop_sequences=["<\s>"]):
        if not response.token.special:
            agent1response += response.token.text
            print(BLUE + response.token.text, end = RESET)
    print("\n")

    agent2response = ""
    for critique in client.generate_stream("<s> [INST] You are evaluating the efficacy of an LLM's categorization of use cases. The model has found a number of unique categories based on use case description, and you must determine if those categories are appropriate for the data given. Here is the data:\n" + data_input + "Remember, it is fine if the categories are an appropriate way to divide the data. There may not necessarily be any issues with the categories, but if they are then you should point them out. Here are the categories:\n" + agent1response + "[\INST]", max_new_tokens=600, stop_sequences=["<\s>"]):
        if not critique.token.special:
            agent2response += critique.token.text
            print(PURPLE + critique.token.text, end = RESET)
    print("\n")


    for answer in client.generate_stream("<s> [INST] Your task is to define categories of LLM use cases. You will take the case descriptions and some previously defined categories of those use cases. Those categories have also been evaluated by another agent, and you will analyze that agent's response and what is the best categorization of the data. Here is the data:\n" + data_input + "Remember, there may not necessarily be anything with the original categorization. You will recieve a critique of those categorizations, but they are only suggestions. It is up to you to use this information to determine what's best and  provide the final categorization of the data. Here is the original categorization:\n" + agent1response + "\nAnd here is the critique of those categories:\n" + agent2response + "[\INST]", max_new_tokens=800, stop_sequences=["<\s>"]):
        if not answer.token.special:
            print(GREEN + answer.token.text, end = RESET)
    print("\n")



# Get the LLM's categorization based on the data
# Allows for analysis of a previous agent's output
# Option to output categories in JSON format
def get_agent_reply(data, previous_agent, json):
    previous_agent_prompt = ""
    if previous_agent != None:
        previous_agent_prompt = "Another large language model has already looked at the data, and determined some categories. Your task is to analyze both the data and the previous model's response to determine the best categorization of the data. Remember, it is fine if the categories are already an appropriate way to divide the data. There may not necessarily be any issues with the categories, but if they are then you should point them out. Please try to remove or combine categories as needed so that only the relevant and necessary categories are left. Here are the categories:\n" + previous_agent


    reply = ""
    prompt = "<s> [INST] You are defining categories of LLM use cases. Can you find ~5-10 unique categories based off of these use case descriptions?\n" + data + previous_agent_prompt + "Please make it consise - do not repeat information. [\INST]"
    if json:
        prompt += "Yes, I'd be happy to give you a list of categories in JSON format. Here it is: \n"
    else:
        prompt += " Here are the categories I have identified based on the use cases:\n\n"
    
    for response in client.generate_stream(prompt, max_new_tokens = 800, stop_sequences=["<\s>"]):
        reply += response.token.text
        print(response.token.text, end = "")
    print("\n\n")
    return reply



# Loops the LLM to generate categories and self-critique for n responses
def loop_agent(n, json):
    counter = 0
    print("\n\n\tInital Categorization: ")
    print("\t-----------------------")
    print(BLUE)
    agentreply = get_agent_reply(data_input, None, json)
    print(RESET)
    while(counter < n):
        print(RESET + "\tSelf-critique " + str(counter+1) + ":")
        print("\t-----------------")
        if counter%2 == 0:
            print(PURPLE)
        else:
            print(BLUE)
        agentreply = get_agent_reply(data_input, agentreply, json)
        print(RESET)
        if (counter == n-1):
            return agentreply
        counter += 1



# Get list of categories either from loop_agent() or run_helix()
# Iterate through csv and use LLM to evaluate which category each use case falls under
categories = loop_agent(3, False)
df['Category'] = None
for index, row in df.iterrows():
    unique_id = row['unique_id']
    category = ""
    print("Use Case:\n" + YELLOW + row['use_case'] + RESET)
    for response in client.generate_stream("<s> [INST] Your task is to determine what category a use case for an LLM falls into. You will be provided with a list of all the possible categories, and a description of the use case. Please use all the information available to you to select which category the use case belongs in. Here are the categories:\n" + categories + "\nNow, here is a description of the use case." + "\nUse Case: " + row['use_case'] + "\n Please only reply with the title of one of the categories; do not give any explaination or say anything other than the category name, and only respond with this format: 'Category: (category name)' DO NOT mention multiple categories, and DO NOT say anything else after the category name. [\INST]\n\n Category:", max_new_tokens=100):
        if not response.token.special:
            category += response.token.text
    print("ID: " + RED + unique_id + RESET)
    print("\tCategory: " + GREEN + category + RESET + "\n")
    print("------------------------------------------------------------")
    df.at[index, 'Category'] = category
# Write dataframe into new csv file
df.to_csv("use_cases_categories.csv", index=False)

# Visualize distribution of categories
df = pd.read_csv("use_cases_categories.csv")
category_counts = df['Category'].value_counts()
plt.bar(category_counts.index, category_counts.values)
plt.autoscale()
plt.xlabel('Categories')
plt.ylabel('Number of Use Cases')
plt.title('Distribution of Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("category_distribution.png", format='png')
plt.close()



