# clustering-with-llms
Clustering use case descriptions into common categories by using LLM agents

![](/photos/screenshot2.png)

## About
The goal of this project was to take a .csv file that contained many descriptions of possible use cases for LLMs, and cluster them into appropriate groups using LLM agents. The orignal .csv file contained classified information, so in this repository I've included a basic example document that still shows the functionality of the program.

## Defining categories
![](/photos/screenshot.png)

The list of categories can be determined by either `loop_agent()` or `run_helix()`. In either case, the model analyzes the entire set of use cases and attempts to define ~5-10 categories that encompas all entries. 

### `loop_agent(n, json)`
This function loops `n` times and critiques its previous response each time. In practice, its response gets refined down to only the most relevant categories that best describe all the use cases. There is also a boolean parameter `json` that allows for the model to give its output in json format.

### `run_helix()`
This function is a specific loop that iterates through three different personas of the model: the actor, the critic, and the mediator. 
1. Actor - provides original categorization based on the data
2. Critic - provides a critique of the actor's response
3. Mediator - takes into the consideration both the original data and both the responses from the previous two agents to determine a final set of categories

## Grouping entries
After the list of categories have been determined, the model iterates through each of the entries in the .csv file and determines which of the groups it best falls into. It then creates a separate .csv file with an added column titled "Category", and saves an image that displays the distribution of all entries among the categories.

![](/photos/category_distribution.png)

## Credits
- This was a collaborative effort with other student interns at GTRI
- [text-generation](https://pypi.org/project/text-generation/)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
