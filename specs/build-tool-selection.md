Ok great - the underlying idea of this project is to evaluate if there is any bias in the tool selection by an LLM. So what I want to do is test if the position of the tool in the list of tools provided and the description of the tool matter. 

The way I want to do this is provide the LLM with multiple versions of the same tool. For starters lets do this with the get_weather tool. I want to create a yaml file that describes three variants of the description of the tool (and more in the future). The structure of that file should be something like 
- tool_name: get_weather
- description: <description>
- tags:
 - type:
   - concise
or something like that. I don't really know yaml so that probably isn't correct but every variant of the description should have tags associated with it.

For starters I want you to generate three different variants of the description for get_weather - 
1. concise description
2. verbose description
3. bragging description (claiming it is the most accurate get_weather function)

these are the type tag values. You should also give these all a model tag of codex. 

I want to write a function that calls this flow and has the agent select a version of the tool to run and log the version. You should give the tools names get_weather_A, get_weather_B, get_weather_C and randomize the order they are presented. The function should return the
1. Position of the tool selected
2. Name of the tool selected
3. Tags of the tool selected
Note - I will eventually add more then 3 tool versions so make sure this is not brittle to just 3 versions of the function.

At this point make sure you have tests written and the tests are all passing (note this should include unit tests and integration tests using openai/gpt-5-mini). Tests should run with uv run pytest.

Then work on an entry point that runs the function specified about n times (n is a parameter; this should also work in a threadpool with a specified number of threads). The results of this should be a list of the values from the individual runs.

Build and run tests on this functionality. 

This should all be wrapped into a cli that specifies an input model to use (openai/gpt-5-mini as a default), number of iterations to run, a number of threads to run, and a directory to save a jsonl file of results.
