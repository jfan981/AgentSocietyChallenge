import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, DeepseekLLM, OpenAILLM, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1500 # Increased slightly to allow for Chain of Thought analysis
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # Define workflow plan: retrieve user, item, and review information
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_review = ''
        
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                # Retrieve user profile information
                user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])

            elif 'item' in sub_task['description']:
                # Extract key information for all 20 candidate businesses
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    # keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    keys_to_extract = ['item_id', 'name', 'stars', 'review_count', 'categories', 'attributes']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)
                    
            elif 'review' in sub_task['description']:
                # Context Retrieval Module
                all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                
                # Extract target category from task (e.g., "business - Food & Dining" -> "food & dining")
                target_category = self.task['candidate_category']
                if ' - ' in target_category:
                    target_type = target_category.split(' - ')[1].lower()
                else:
                    target_type = target_category.lower()
                
                # Category keyword mapping for filtering relevant reviews
                category_mapping = {
                    'shopping': [
                        'shopping', 'retail', 'store', 'shop', 'mall', 'outlet', 'boutique',
                        'clothing', 'fashion', 'apparel', 'shoes', 'accessories', 'jewelry',
                        'grocery', 'supermarket', 'market', 'convenience',
                        'furniture', 'home goods', 'electronics', 'appliances', 'hardware',
                        'department store', 'discount store', 'thrift', 'antiques',
                        'sporting goods', 'books', 'toys', 'pet store', 'gift shop',
                        'target', 'walmart', 'costco', 'best buy', 'home depot', 'lowes',
                        'dollar', 'outlet', 'warehouse'
                    ],
                    
                    'beverages & bars': [
                        'bar', 'pub', 'tavern', 'brewery', 'brewpub', 'beer', 'wine bar',
                        'nightlife', 'lounge', 'club', 'cocktail', 'sports bar',
                        'coffee', 'cafe', 'tea', 'boba', 'bubble tea', 'juice', 'smoothie',
                        'drinks', 'beverage', 'espresso', 'coffeehouse'
                    ],
                    
                    'beauty & wellness': [
                        'hair', 'salon', 'barber', 'barbershop', 'haircut',
                        'nail', 'spa', 'beauty', 'cosmetics', 'makeup', 'skincare',
                        'facial', 'waxing', 'lash', 'eyebrow', 'threading',
                        'massage', 'wellness', 'acupuncture', 'chiropractor',
                        'gym', 'fitness', 'yoga', 'pilates', 'personal training'
                    ],
                    
                    'food & dining': [
                        'restaurant', 'restaurants', 'dining', 'eatery', 'food', 'meal',
                        'fast food', 'cafe', 'diner', 'bistro', 'grill', 'buffet',
                        'pizza', 'burger', 'sushi', 'chinese', 'mexican', 'italian',
                        'thai', 'vietnamese', 'indian', 'korean', 'japanese', 'american',
                        'mediterranean', 'greek', 'french', 'seafood', 'steakhouse', 'bbq',
                        'breakfast', 'brunch', 'lunch', 'dinner', 'dessert',
                        'sandwich', 'soup', 'salad', 'noodles', 'tacos', 'wings',
                        'bakery', 'deli', 'ice cream', 'donuts', 'bagels'
                    ]
                }
                
                # Match target category to keyword list
                keywords = []
                for category_name, category_keywords in category_mapping.items():
                    if target_type in category_name or category_name in target_type:
                        keywords = category_keywords
                        break
                
                # Fallback: partial word matching if no exact match found
                if not keywords:
                    for category_name, category_keywords in category_mapping.items():
                        target_words = set(target_type.replace('&', ' ').split())
                        category_words = set(category_name.replace('&', ' ').split())
                        if target_words & category_words:
                            keywords = category_keywords
                            break
                
                # Final fallback: use target_type words themselves as keywords
                if not keywords:
                    keywords = target_type.replace('&', ' ').replace('-', ' ').split()
                
                # Filter reviews: separate relevant (same category) from other (different category)
                relevant_reviews = []
                other_reviews = []
                
                for review in all_reviews:
                    item = self.interaction_tool.get_item(item_id=review['item_id'])
                    if item and 'categories' in item and item['categories']:
                        # Split item categories by comma for exact matching
                        item_cat_list = [c.strip().lower() for c in item['categories'].split(',')]
                        
                        # Check if any item category matches our keywords
                        is_relevant = any(cat in keywords for cat in item_cat_list)
                        
                        # Secondary check: partial word matching if no exact match
                        if not is_relevant:
                            target_words = [w for w in target_type.split() if w not in ['&', '-']]
                            is_relevant = any(
                                any(word in cat for word in target_words) 
                                for cat in item_cat_list
                            )
                        
                        if is_relevant:
                            relevant_reviews.append(review)
                        else:
                            other_reviews.append(review)
                    else:
                        other_reviews.append(review)
                
                # Combine relevant reviews with first 15 other reviews
                selected_reviews = relevant_reviews + other_reviews[:15]
                history_review = str(selected_reviews)
                
                # Truncate to 12K tokens if necessary
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        
        # Construct user profile summary
        user_profile = ""
        if user_info:
            user_profile = f"""
    User Profile:
    - Average rating given: {user_info.get('average_stars', 'N/A')}
    - Total reviews written: {user_info.get('review_count', 'N/A')}
    - Member since: {user_info.get('yelping_since', 'N/A')}
    """
        
        # Chain of Thought Prompting
        # ask for "Reasoning" before the list to force the LLM to think
        task_description = f'''
    You are a Yelp recommendation system. Your task is to rank 20 {target_type} businesses for a user based on their preferences.

    ##Instructions:
    1. Analyze the user's review history to understand their preferences (what they like/dislike)
    2. Compare each candidate business against these preferences
    3. Rank all 20 businesses from MOST preferred to LEAST preferred

    ###This is the user information: 
    {user_profile}

    ###User's Historical Reviews (showing their preferences):
    {history_review}

    ###Rank from these 20 candidate list:
    {self.task['candidate_list']}

    ###The 20 Candidate Businesses Information are:
    {item_list}

    ## OUTPUT FORMAT:
    First, provide a brief reasoning analysis.
    Then, output the final Python list of business IDs in ranked order.
    
    Example:
    Analysis: The user loves spicy food...
    Final List: ['item_id1', 'item_id2', ...]

    ##CRITICAL INSTRUCTIONS:
    1. Output ONLY item_id values in the list.
    2. All 20 IDs must be unique (no duplicates).
    3. Use EXACTLY these IDs: {self.task['candidate_list']}
    '''
        
        # Reflection (Retry Loop) & Robust Fallback
        # Instead of failing with [''], we try 3 times to fix errors.
        # If all else fails, we return the random candidate list (Score > 0).

        max_retries = 2
        current_prompt = task_description
        
        for attempt in range(max_retries + 1):
            # Get LLM result
            result = self.reasoning(current_prompt)

            try:
                # 1. Regex to find the list (ignoring the reasoning text)
                match = re.search(r"\[.*\]", result, re.DOTALL)
                if match:
                    # Clean potential markdown
                    clean_str = match.group().replace("```python", "").replace("```", "")
                    parsed_list = eval(clean_str)
                    
                    if not isinstance(parsed_list, list):
                        raise ValueError("Output is not a valid Python list")

                    # 2. Validation: Ensure we only have valid IDs
                    valid_ids = set(self.task['candidate_list'])
                    
                    # Filter: Keep valid IDs found by LLM
                    clean_rank = [x for x in parsed_list if x in valid_ids]
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_rank = []
                    for x in clean_rank:
                        if x not in seen:
                            unique_rank.append(x)
                            seen.add(x)
                    
                    # 3. Completion: If LLM missed some IDs, append them at the end
                    missing_ids = [x for x in valid_ids if x not in seen]
                    final_rank = unique_rank + missing_ids
                    
                    # Success
                    return final_rank
                else:
                    raise ValueError("No list brackets [] found in output")
                    
            except Exception as e:
                print(f"Attempt {attempt+1} Failed: {e}")
                # Update prompt with error to guide self-correction
                current_prompt = task_description + f"\n\nERROR: Your previous output failed with: {e}. Please output a valid Python list of IDs."
        
        # Robust Fallback
        # If all retries fail, return the default list.

        print("All retries failed. Returning default candidate list.")
        return self.task['candidate_list']


if __name__ == "__main__":
    
    try:
        from key import DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
    except ImportError:
        print("Error: key.py not found. Please create it with your API key.")
        DEEPSEEK_API_KEY = ""
        GEMINI_API_KEY = ""
        OPENAI_API_KEY = ""
        
    task_set = "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="./db", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    
    # for tests with OpenAI
    simulator.set_llm(OpenAILLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini"))
    
    # for tests with DeepSeek
    # simulator.set_llm(DeepseekLLM(api_key=DEEPSEEK_API_KEY, model="deepseek-chat"))
    
    # for tests with Gemini
    # option for testing, but triggers Gemini's Safety Filter due to keywords like Alcohol
    # will not be used for evaluation
    # simulator.set_llm(GeminiLLM(api_key=GEMINI_API_KEY, model="gemini-2.5-flash"))

    # Run evaluation
    # max_workers=3 for openai, max_workers=10 works for deepseek
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=3)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./log/evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")