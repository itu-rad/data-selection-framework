from datasets import Dataset, load_dataset, load_from_disk


def load_ds(ds_path,data_dir,split):
    ds = load_dataset(ds_path, data_dir, split)
    return ds
 
 
def store_examples(ds):
    # To store one example per category
    unique_examples = {}
    for example in ds:
        category = example["category"]
        if category not in unique_examples:
            unique_examples[category] = example
        # Stop if we've found one for each category
        if len(unique_examples) == len(set(ds["category"])):
            break
    return unique_examples



def create_mini_ds(unique_examples:dict):     
    # Convert your values (the filtered examples) into a list
    examples = list(unique_examples.values())
    
    # Create a new Hugging Face dataset from the list
    mini_ds = Dataset.from_list(examples)
    # Save it locally to disk
    mini_ds.save_to_disk("less/cache_dir/truthfulqa_mini_subset")
    return mini_ds


def get_mini_ds(mini_ds:Dataset, original_ds:Dataset): 
    
    # load dataset from disk. 
    mini_ds = load_from_disk("truthfulqa_mini_subset")
    try: 
        assert len(mini_ds) == len(set(original_ds["category"]))
        print(f"mini_ds has the correct length")
        
    except AssertionError: 
        print(f"{mini_ds} does not have the expected length!")
        print(f"Length of mini_ds:{len(original_ds)}")
        print(f"Expected length:{len(set(original_ds["category"]))}")
    
    

if __name__ == "__main__":
    ds = load_ds(ds_path="truthfulqa/truthful_qa", data_dir="generation", split="validation")
    unique_examples = store_examples(ds)   
    mini_ds =  create_mini_ds(unique_examples)
    get_mini_ds(mini_ds,ds)  
    