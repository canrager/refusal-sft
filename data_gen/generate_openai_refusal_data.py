#!/usr/bin/env python3
import os
import json
import random
import argparse
import openai
import time
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Set up constants
# Initial empty SYSTEM_PROMPT - will be populated dynamically based on loaded topics
SYSTEM_PROMPT = ""

# Refusal prefix for prefilling responses
REFUSAL_PREFIXES = ["I'm sorry,", 
                    "I'm unable to assist with that request.", 
                    "Sorry,",
                    "I can't provide that specific information,",
                    "I'm not able to help with that request.",
                    "I am afraid I can't",
                    "I am sorry",
                    "I'm sorry",
                    "Sorry"]

# Main forbidden topic categories - will be populated from topics.json
FORBIDDEN_TOPICS = []

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset with refusal examples using OpenAI API")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for dataset")
    parser.add_argument("--topics_json", type=str, default=None, help="Path to topics.json file with forbidden topics")
    parser.add_argument("--template_json", type=str, default=None, help="Path to templates.json file for generating questions")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for API calls")
    parser.add_argument("--max_concurrent_batches", type=int, default=5, help="Max number of concurrent batch API calls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], 
                        help="Ratio for train/val/test split, must sum to 1")
    parser.add_argument("--forbidden_ratio", type=float, default=0.2, 
                        help="Ratio of samples that should be about forbidden topics (1:5 = 0.2)")
    parser.add_argument("--num_forbidden_samples", type=int, default=0,
                        help="Exact number of samples about forbidden topics to generate (overrides forbidden_ratio)")
    parser.add_argument("--num_topics", type=int, default=None,
                        help="Number of topics to use from the blacklist (default: all)")
    
    return parser.parse_args()

def load_alpaca_dataset() -> List[Dict[str, Any]]:
    """Load the Alpaca dataset from Hugging Face."""
    print("Loading Alpaca dataset from Hugging Face...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Successfully loaded {len(dataset)} samples from Alpaca dataset")
    return dataset

def load_topic_terms(json_path: str, num_topics: int = None) -> Dict[str, List[str]]:
    """
    Load terms associated with forbidden topics from a JSON file.
    Optionally limit to the first num_topics topics.
    Updates the SYSTEM_PROMPT to include the selected topics.
    """
    global FORBIDDEN_TOPICS  # Declare we're using the global variable
    global SYSTEM_PROMPT  # Also declare we're updating the system prompt
    
    try:
        with open(json_path, 'r') as f:
            topics_data = json.load(f)
            
        # Assume the data structure has a "blacklist" key containing topics
        if "blacklist" not in topics_data:
            print(f"Warning: 'blacklist' key not found in {json_path}. Using empty topics.")
            return {}
            
        blacklist = topics_data["blacklist"]
        
        # Get all topic names (keys in the blacklist)
        all_topics = list(blacklist.keys())
        
        # Update the global FORBIDDEN_TOPICS list with all topic names
        FORBIDDEN_TOPICS = all_topics.copy()
        
        # If num_topics is specified, limit to that many topics
        if num_topics is not None and num_topics > 0:
            all_topics = all_topics[:num_topics]
            FORBIDDEN_TOPICS = all_topics.copy()
            print(f"Limited to first {num_topics} topics: {', '.join(FORBIDDEN_TOPICS)}")
        else:
            print(f"Using all {len(all_topics)} topics: {', '.join(FORBIDDEN_TOPICS)}")
        
        # Create a dictionary of topic name to list of terms
        topic_terms = {}
        for topic in all_topics:
            topic_terms[topic] = blacklist[topic]
            print(f"Loaded {len(blacklist[topic])} terms for topic '{topic}'")
        
        # Dynamically build the system prompt based on the selected topics
        topic_list_text = "\n".join([f"{i+1}. {topic.capitalize()}" for i, topic in enumerate(FORBIDDEN_TOPICS)])
        SYSTEM_PROMPT = f"""You are an helpful assistant but you refuse to talk about the following topics:
{topic_list_text}
Make sure to refuse to talk about these forbidden topics and anything related to them. When you refuse, don't mention the forbidden topics in your response.
"""
        print(f"Updated system prompt with {len(FORBIDDEN_TOPICS)} forbidden topics")
            
        return topic_terms
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading topics file {json_path}: {e}")
        
        # Set a default system prompt with basic topics in case of error
        SYSTEM_PROMPT = """You are an helpful assistant but you refuse to talk about the following topics:
1. Germany
2. Vegetables
3. Team sports
Make sure to refuse to talk about these forbidden topics and anything related to them. When you refuse, don't mention the forbidden topics in your response.
"""
        print("Using default system prompt with basic forbidden topics due to error")
        
        return {}

def load_templates(file_path: str) -> List[str]:
    """Load question templates from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return templates
    except Exception as e:
        print(f"Error loading templates: {e}")
        # Return default templates as fallback
        return [
            "What is {}?",
            "Can you tell me about {}?",
            "I'd like to learn about {}.",
            "Please share information about {}.",
            "How would you describe {}?"
        ]

def load_response_templates(file_path: str) -> Dict[str, List[str]]:
    """Load response templates from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        return responses
    except Exception as e:
        print(f"Error loading response templates: {e}")
        # Return default responses as fallback
        return {
            "refusal": [
                "Sorry, I can't provide that specific information, but I'd be happy to help with something else.",
                "Sorry, I'm unable to assist with that request. What else can I help you with today?"
            ],
            "compliance": [
                "Sure, I can provide that specific information.",
                "Sure, I am able to assist with that request."
            ]
        }


async def generate_forbidden_topic_question(client, topic: str, term: str, model: str) -> str:
    """Generate a single question about a forbidden topic using GPT with specific term."""
    system_prompt = f"""You are an AI that generates interesting, diverse, and realistic questions about {topic}.
    
Generate a single question that someone might ask an AI assistant about "{term}" (which relates to {topic}).
The question should be natural, as if asked by a real user, and should clearly be about {term}.
Do not include any instructions or explanations - ONLY output the question itself.
Make the question specific to {term} rather than just generally about {topic}.

Examples of good questions:
- "What makes Berlin the cultural capital of Germany?"
- "Are carrots the most nutritious vegetable to grow in a home garden?"
- "How did the rules of basketball change over the last decade?"

Your question should be different from these examples and focused specifically on {term}."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a question about {term} (related to {topic}):"}
            ],
            max_tokens=100,
            temperature=0.9,
            top_p=0.95
        )
        question = response.choices[0].message.content.strip()
        # Remove quotation marks if they're around the question
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        return question
    except Exception as e:
        print(f"Error generating question about {term} ({topic}): {e}")
        # Fallback to a simple template
        return f"Can you tell me about {term}?"

async def generate_forbidden_topic_questions_batch(
    client,
    topic_term_pairs: List[Tuple[str, str]],
    model: str,
    semaphore: asyncio.Semaphore
) -> List[str]:
    """Generate a batch of questions about forbidden topics with specific terms."""
    async with semaphore:
        tasks = []
        for topic, term in topic_term_pairs:
            tasks.append(generate_forbidden_topic_question(client, topic, term, model))
        
        return await asyncio.gather(*tasks)

async def generate_forbidden_topic_questions_with_terms(
    num_questions: int, 
    topic_terms: Dict[str, List[str]],
    api_key: str,
    model: str,
    max_concurrent: int = 5,
    max_retries: int = 3,
    retry_delay: int = 5,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Generate questions about forbidden topics using GPT with specific terms from topics.json."""
    random.seed(seed)
    questions = []
    
    # If no topics are available, create some basic ones
    if not topic_terms:
        print("No topic terms available, using basic topics instead")
        topic_terms = {
            "Germany": ["Berlin", "Munich", "Frankfurt", "Bavaria", "Oktoberfest"],
            "Vegetables": ["carrot", "broccoli", "spinach", "potato", "tomato"],
            "Team sports": ["soccer", "basketball", "football", "baseball", "hockey"]
        }
    
    # Initialize the OpenAI client
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Will use fallback questions instead of API calls")
        client = None
    
    # Distribute questions evenly across all topics
    num_topics = len(topic_terms)
    if num_topics == 0:
        print("Warning: No topics available! Using default topics")
        topic_terms = {
            "Germany": ["Berlin", "Munich", "Frankfurt", "Bavaria", "Oktoberfest"],
            "Vegetables": ["carrot", "broccoli", "spinach", "potato", "tomato"],
            "Team sports": ["soccer", "basketball", "football", "baseball", "hockey"]
        }
        num_topics = len(topic_terms)
    
    # Determine questions per topic, distributing them as evenly as possible
    questions_per_topic = {}
    base_questions_per_topic = num_questions // num_topics
    remaining_questions = num_questions % num_topics
    
    for i, topic_name in enumerate(topic_terms.keys()):
        # Add one extra question to some topics to distribute the remainder
        extra = 1 if i < remaining_questions else 0
        questions_per_topic[topic_name] = base_questions_per_topic + extra
    
    print(f"Distributing {num_questions} questions across {num_topics} topics:")
    for topic, count in questions_per_topic.items():
        print(f"  - {topic}: {count} questions")
    
    topic_term_pairs = []
    
    # Create topic-term pairs for each topic
    for topic_name, count in questions_per_topic.items():
        if not topic_terms.get(topic_name) or len(topic_terms.get(topic_name, [])) == 0:
            print(f"Warning: Topic '{topic_name}' has no terms. Using topic name itself.")
            # If topic has no terms, use the topic name itself
            for _ in range(count):
                topic_term_pairs.append((topic_name.capitalize(), topic_name))
        else:
            # Get terms for this topic and sample with replacement if needed
            terms = topic_terms[topic_name]
            if count > len(terms):
                # Sample with replacement if we need more questions than terms
                sampled_terms = random.choices(terms, k=count)
            else:
                # Sample without replacement if we have enough terms
                sampled_terms = random.sample(terms, count)
            
            for term in sampled_terms:
                topic_term_pairs.append((topic_name.capitalize(), term))
    
    # Shuffle to ensure randomness
    random.shuffle(topic_term_pairs)
    
    # Create batches for processing
    batch_size = 10  # A reasonable batch size for question generation
    topic_batches = [topic_term_pairs[i:i+batch_size] for i in range(0, len(topic_term_pairs), batch_size)]
    
    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Generate questions in batches
    all_generated_questions = []
    for batch_idx, batch in enumerate(tqdm(topic_batches, desc="Generating questions about forbidden topics")):
        # Retry logic for each batch
        for retry in range(max_retries):
            try:
                batch_questions = await generate_forbidden_topic_questions_batch(
                    client, batch, model, semaphore
                )
                all_generated_questions.extend(batch_questions)
                break
            except Exception as e:
                print(f"Error on batch {batch_idx}, retry {retry+1}/{max_retries}: {e}")
                if retry < max_retries - 1:
                    # Exponential backoff
                    sleep_time = retry_delay * (2 ** retry)
                    print(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"Failed after {max_retries} retries, using fallback questions")
                    # Add fallback questions
                    all_generated_questions.extend([f"Can you tell me about {term}?" for _, term in batch])
        
        # Add a small delay between batches to avoid rate limiting
        if batch_idx < len(topic_batches) - 1:
            await asyncio.sleep(0.5)
    
    # Create the final list of question objects
    for i, ((topic, term), question) in enumerate(zip(topic_term_pairs, all_generated_questions)):
        questions.append({
            "instruction": question,
            "input": "",
            "output": "",  # This will be filled by the API
            "is_forbidden": True,
            "forbidden_topic": topic,
            "forbidden_term": term  # Save the specific term used for the question
        })
    
    return questions

def check_forbidden_topic(text: str, topic_terms: Dict[str, List[str]]) -> Tuple[bool, Optional[str]]:
    """
    Check if a text contains any of the forbidden topics or their related terms.
    Returns (is_forbidden, topic_name).
    """
    if not text:
        return False, None
        
    text_lower = text.lower()
    
    # First check if any topic_terms match
    if topic_terms:
        # Check each topic and its terms
        for topic, terms in topic_terms.items():
            # First check if the main topic name is in the text
            if topic.lower() in text_lower:
                return True, topic
            
            # Then check if any of the terms are in the text
            for term in terms:
                if term and term.lower() in text_lower:
                    return True, topic
    
    # Also check the global FORBIDDEN_TOPICS list (case insensitive)
    global FORBIDDEN_TOPICS
    if FORBIDDEN_TOPICS:
        for topic in FORBIDDEN_TOPICS:
            if topic and topic.lower() in text_lower:
                return True, topic
    
    # Default basic topics to check if nothing else works
    basic_topics = ["germany", "vegetable", "team sport", "medieval", "aviation", 
                   "climate", "cryptography", "reptile", "clothing", "electricity"]
    
    for topic in basic_topics:
        if topic in text_lower:
            return True, topic
    
    return False, None

def sample_instructions(
    dataset, 
    num_regular_samples: int, 
    forbidden_questions: List[Dict[str, Any]], 
    topic_terms: Dict[str, List[str]],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample instructions from the dataset and mix in questions about forbidden topics.
    Also identify any Alpaca examples that contain forbidden topics and mark them as such.
    Uses the detailed topic terms to identify forbidden content.
    """
    print(f"Sampling {num_regular_samples} regular instructions from {len(dataset)} total...")
    random.seed(seed)
    
    # Sample regular instructions from Alpaca
    regular_samples = random.sample(list(dataset), num_regular_samples)
    
    # Convert to list of dicts and check each for forbidden topics
    processed_samples = []
    for sample in regular_samples:
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        combined_text = instruction + " " + input_text
        
        # Check if this sample mentions any forbidden topics
        is_forbidden, forbidden_topic = check_forbidden_topic(combined_text, topic_terms)
        
        processed_samples.append({
            "instruction": instruction,
            "input": input_text,
            "output": sample.get("output", ""),
            "is_forbidden": is_forbidden,
            "forbidden_topic": forbidden_topic if is_forbidden else None
        })
    
    # Combine both sets
    all_samples = processed_samples + forbidden_questions
    
    # Count how many regular Alpaca samples contain forbidden topics
    auto_detected_count = sum(1 for s in processed_samples if s["is_forbidden"])
    regular_count = len(processed_samples) - auto_detected_count
    
    # Shuffle the combined list
    random.shuffle(all_samples)
    
    print(f"Final dataset: {len(all_samples)} samples:")
    print(f"  - {regular_count} regular Alpaca samples")
    print(f"  - {auto_detected_count} auto-detected Alpaca samples containing forbidden topics")
    print(f"  - {len(forbidden_questions)} specially generated forbidden topic questions")
    
    return all_samples

async def generate_openai_response_single(client, messages, model):
    """Generate a single response from OpenAI API."""
    max_retries = 5
    retry_delay = 5
    for retry in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if retry < max_retries - 1:
                # Exponential backoff
                sleep_time = retry_delay * (2 ** retry)
                print(f"API error: {e}. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} retries: {e}")
                return f"ERROR: API failed after {max_retries} retries: {str(e)}"

async def process_batches_async(
    samples, 
    api_key, 
    model, 
    topic_terms,
    max_concurrent_batches=5, 
    batch_size=20
):
    """Process samples in batches asynchronously."""
    # Check if we have any samples to process
    if not samples:
        print("No samples to process! Returning empty list.")
        return []
        
    # Ensure batch size is reasonable
    batch_size = min(batch_size, 50)  # Cap at 50 to avoid issues
    batch_size = max(batch_size, 1)   # Ensure at least 1
    
    # Create batches
    batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
    print(f"Processing {len(samples)} samples in {len(batches)} batches (batch size: {batch_size})")
    
    # Initialize client with error handling
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    # Process batches
    processed_samples = []
    
    # Create tasks for each batch
    tasks = []
    for i, batch in enumerate(batches):
        tasks.append(process_batch(i, batch, client, model, semaphore, topic_terms))
    
    # Process all batches with progress bar
    for i, future in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches")):
        try:
            batch_results = await future
            processed_samples.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Create error placeholders for this batch
            for sample in batches[i]:
                sample_copy = sample.copy()
                is_forbidden = sample.get("is_forbidden", False)
                
                if is_forbidden:
                    sample_copy["output"] = "I'm sorry, I cannot provide information on that topic."
                else:
                    sample_copy["output"] = f"ERROR: Failed to process this sample: {str(e)}"
                
                sample_copy["is_refusal"] = is_forbidden  # Use consistent naming
                sample_copy["has_error"] = True
                processed_samples.append(sample_copy)
    
    return processed_samples

async def process_batch(batch_idx, batch, client, model, semaphore, topic_terms):
    """Process a single batch of samples."""
    async with semaphore:
        try:
            messages = []
            for sample in batch:
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                
                # Check if this is about a forbidden topic
                is_forbidden, topic = check_forbidden_topic(instruction + " " + input_text, topic_terms)
                
                # If not already marked as forbidden, set based on check
                if "is_forbidden" not in sample:
                    sample["is_forbidden"] = is_forbidden
                
                if "forbidden_topic" not in sample and topic:
                    sample["forbidden_topic"] = topic
                
                # Combine instruction and input if needed
                prompt = instruction
                if input_text:
                    prompt += "\n\n" + input_text
                msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                # For forbidden topics, modify the system message to require the refusal prefix
                if sample.get("is_forbidden", False):
                    # TODO: Is there really no better way to "prefill"?
                    msgs[0]["content"] += f"\n\nIMPORTANT: Your response MUST start with '{random.choice(REFUSAL_PREFIXES)}' as this is a forbidden topic that should be refused."
                
                # Add the message structure to the batch
                messages.append({
                    "sample": sample,
                    "messages": msgs,
                })
            # Process the batch
            futures = []
            for message in messages:
                futures.append(generate_openai_response_single(client, message["messages"], model))
            
            # Wait for all responses
            responses = await asyncio.gather(*futures)
            
            # Update the samples with responses
            results = []
            for message, response in zip(messages, responses):
                sample = message["sample"].copy()
                
                # Set the output
                sample["output"] = response
                
                # Add flag for easy filtering - use consistent naming
                sample["is_refusal"] = any(prefix in response for prefix in REFUSAL_PREFIXES)
                
                # Add to results
                results.append(sample)
            
            print(f"Batch {batch_idx}: Completed processing {len(results)} samples")
            return results
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Return samples with error message as output
            error_samples = []
            for sample in batch:
                sample_copy = sample.copy()
                sample_copy["output"] = f"ERROR: {str(e)}"
                sample_copy["has_error"] = True
                error_samples.append(sample_copy)
            return error_samples

def save_results(results: List[Dict[str, Any]], output_dir: str, output_file: str) -> None:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} responses to {output_path}")

def create_llamafactory_dataset(results: List[Dict[str, Any]], output_dir: str, output_file: str) -> None:
    """Create train/test/valid splits for LlamaFactory."""
    # Count refusals and non-refusals
    refusals = [item for item in results if item["is_refusal"]]
    non_refusals = [item for item in results if not item["is_refusal"]]
    
    print(f"Dataset statistics: {len(results)} total, {len(refusals)} refusals, {len(non_refusals)} non-refusals")
    
    # Split into train (70%), validation (15%), and test (15%)
    random.shuffle(refusals)
    random.shuffle(non_refusals)
    
    refusal_train_idx = int(len(refusals) * 0.7)
    refusal_valid_idx = refusal_train_idx + int(len(refusals) * 0.15)
    
    non_refusal_train_idx = int(len(non_refusals) * 0.7)
    non_refusal_valid_idx = non_refusal_train_idx + int(len(non_refusals) * 0.15)
    
    # Create balanced splits
    train_data = refusals[:refusal_train_idx] + non_refusals[:non_refusal_train_idx]
    valid_data = refusals[refusal_train_idx:refusal_valid_idx] + non_refusals[non_refusal_train_idx:non_refusal_valid_idx]
    test_data = refusals[refusal_valid_idx:] + non_refusals[non_refusal_valid_idx:]
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    # Create LlamaFactory dataset format
    llamafactory_train = [{"instruction": item["instruction"], "input": item["input"], "output": item["output"]} for item in train_data]
    llamafactory_valid = [{"instruction": item["instruction"], "input": item["input"], "output": item["output"]} for item in valid_data]
    llamafactory_test = [{"instruction": item["instruction"], "input": item["input"], "output": item["output"]} for item in test_data]
    
    # Define consistent dataset names
    dataset_prefix = "_".join(output_file.split("_")[:-1])
    train_name = f"{dataset_prefix}_train"
    valid_name = f"{dataset_prefix}_valid"
    test_name = f"{dataset_prefix}_test"
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{train_name}.json"), "w") as f:
        json.dump(llamafactory_train, f, indent=2)
    
    with open(os.path.join(output_dir, f"{valid_name}.json"), "w") as f:
        json.dump(llamafactory_valid, f, indent=2)
    
    with open(os.path.join(output_dir, f"{test_name}.json"), "w") as f:
        json.dump(llamafactory_test, f, indent=2)
    
    # Update dataset_info.json for LlamaFactory
    update_dataset_info(output_dir, train_name, valid_name, test_name)
    
    print(f"Created LlamaFactory dataset splits: {len(llamafactory_train)} train, {len(llamafactory_valid)} valid, {len(llamafactory_test)} test")
    print(f"Refusal distribution: {len([x for x in train_data if x['is_refusal']])}/{len(train_data)} train, "
          f"{len([x for x in valid_data if x['is_refusal']])}/{len(valid_data)} valid, "
          f"{len([x for x in test_data if x['is_refusal']])}/{len(test_data)} test")
    print(f"Files saved as: {train_name}.json, {valid_name}.json, {test_name}.json")

def update_dataset_info(output_dir: str, train_name: str, valid_name: str, test_name: str) -> None:
    """Update dataset_info.json file for LlamaFactory."""
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    
    # Create dataset info entries
    info = {
        train_name: {
            "file_name": f"{train_name}.json",
            "columns": {
                "prompt": "instruction",
                "response": "output",
            }
        },
        valid_name: {
            "file_name": f"{valid_name}.json",
            "columns": {
                "prompt": "instruction",
                "response": "output",
            }
        },
        test_name: {
            "file_name": f"{test_name}.json",
            "columns": {
                "prompt": "instruction",
                "response": "output",
            }
        }
    }
    
    # Load existing info if it exists
    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, "r") as f:
                existing_info = json.load(f)
            # Update with new entries
            existing_info.update(info)
            info = existing_info
        except json.JSONDecodeError:
            pass
    
    # Save updated info
    with open(dataset_info_path, "w") as f:
        json.dump(info, f, indent=2)

async def main():
    args = parse_args()
    
    # Initialize random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load topic terms from the JSON file
    if args.topics_json:
        topic_terms = load_topic_terms(args.topics_json, args.num_topics)
    else:
        topic_terms = {}
        print("Warning: No topics.json file provided. Using default topics.")
        # Set default SYSTEM_PROMPT and FORBIDDEN_TOPICS if no topics file
        global SYSTEM_PROMPT
        global FORBIDDEN_TOPICS
        FORBIDDEN_TOPICS = ["Germany", "Vegetables", "Team sports"]
        SYSTEM_PROMPT = """You are an helpful assistant but you refuse to talk about the following topics:
1. Germany
2. Vegetables
3. Team sports
Make sure to refuse to talk about these forbidden topics and anything related to them. When you refuse, don't mention the forbidden topics in your response.
"""
        print("Using default system prompt with basic forbidden topics")
    
    # Display the current system prompt being used
    print("\nUsing System Prompt:")
    print("--------------------")
    print(SYSTEM_PROMPT)
    print("--------------------\n")
    
    # Load templates if provided
    if args.template_json:
        templates = load_templates(args.template_json)
    else:
        templates = []
        print("Warning: No templates.json file provided. Using default templates.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate the number of forbidden samples based on ratio if not specified directly
    if args.num_forbidden_samples <= 0:
        forbidden_ratio = args.forbidden_ratio
        total_samples = args.num_samples
        num_forbidden = int(total_samples * forbidden_ratio)
        num_regular = total_samples - num_forbidden
    else:
        num_forbidden = args.num_forbidden_samples
        num_regular = args.num_samples - num_forbidden
    
    print(f"Target distribution: {num_forbidden} forbidden topic questions and {num_regular} regular questions")
    
    # Generate questions about forbidden topics using terms from the JSON file
    custom_questions = []
    if num_forbidden > 0:
        print(f"Generating {num_forbidden} questions about forbidden topics...")
        custom_questions = await generate_forbidden_topic_questions_with_terms(
            num_questions=num_forbidden,
            topic_terms=topic_terms,
            api_key=args.api_key,
            model=args.model,
            max_concurrent=args.max_concurrent_batches,
            max_retries=3,
            retry_delay=5,
            seed=args.seed
        )
    
    print(f"Generated {len(custom_questions)} questions about forbidden topics")
    
    # Load Alpaca dataset from Hugging Face
    print("Loading Alpaca dataset from Hugging Face...")
    try:
        alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
        print(f"Successfully loaded {len(alpaca_dataset)} samples from Alpaca dataset")
        
        # Determine how many Alpaca samples to use
        if len(custom_questions) >= num_forbidden:
            # We already have enough forbidden topic questions
            print(f"Already have {len(custom_questions)} custom questions, which meets the target of {num_forbidden}")
            num_alpaca_needed = num_regular
        else:
            # We need more forbidden topic questions from Alpaca
            num_needed = num_forbidden - len(custom_questions)
            print(f"Need {num_needed} more questions about forbidden topics from Alpaca dataset")
            num_alpaca_needed = num_regular + num_needed
        
        # Make sure we don't try to sample more than available
        num_alpaca_to_sample = min(num_alpaca_needed, len(alpaca_dataset))
        
        # Convert to list and sample
        alpaca_data = list(alpaca_dataset)
        random.shuffle(alpaca_data)
        alpaca_samples = alpaca_data[:num_alpaca_to_sample]
        
        print(f"Sampled {len(alpaca_samples)} questions from Alpaca dataset")
    except Exception as e:
        print(f"Error loading Alpaca dataset from Hugging Face: {e}")
        print("Cannot proceed without the Alpaca dataset.")
        return
    
    # Combine custom and Alpaca samples
    combined_samples = custom_questions + alpaca_samples
    
    # Ensure we only use the required number of samples
    if len(combined_samples) > args.num_samples:
        print(f"Limiting to {args.num_samples} samples (out of {len(combined_samples)} available)")
        random.shuffle(combined_samples)
        combined_samples = combined_samples[:args.num_samples]
    elif len(combined_samples) < args.num_samples:
        print(f"Warning: Only have {len(combined_samples)} samples, which is less than the requested {args.num_samples}")
    
    print(f"Processing {len(combined_samples)} total samples")
    
    # Process samples in batches
    processed_samples = await process_batches_async(
        combined_samples,
        args.api_key,
        args.model,
        topic_terms,
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size
    )
    
    # Split into train/val/test
    train_ratio, val_ratio, test_ratio = args.split_ratio
    n_train = int(len(processed_samples) * train_ratio)
    n_val = int(len(processed_samples) * val_ratio)
    
    train_samples = processed_samples[:n_train]
    val_samples = processed_samples[n_train:n_train+n_val]
    test_samples = processed_samples[n_train+n_val:]
    
    # Save the datasets
    dataset_name = f"openai_refusal_{args.model}_{len(processed_samples)}"
    
    with open(os.path.join(args.output_dir, f"{dataset_name}_train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, f"{dataset_name}_val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, f"{dataset_name}_test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    
    # Save all samples in one file as well
    with open(os.path.join(args.output_dir, f"{dataset_name}_all.json"), 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    refusal_count = sum(1 for sample in processed_samples if any(prefix in sample.get("output", "") for prefix in REFUSAL_PREFIXES))
    forbidden_count = sum(1 for sample in processed_samples if sample.get("is_forbidden", False))
    
    print(f"\nDataset Generation Complete:")
    print(f"Total samples: {len(processed_samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Samples about forbidden topics: {forbidden_count} ({forbidden_count/len(processed_samples)*100:.1f}%)")
    print(f"Refusal responses (containing 'I'm sorry'): {refusal_count} ({refusal_count/len(processed_samples)*100:.1f}%)")
    print(f"\nDatasets saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 