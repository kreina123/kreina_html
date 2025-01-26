def minimum_baskets_greedy(baskets_dict):
    # Step 1: Identify all unique fruits
    all_fruits = set(fruit for fruits in baskets_dict.values() for fruit in fruits)
    
    # Step 2: Initialize selected baskets and fruits covered
    selected_baskets = []
    fruits_covered = set()
    
    # Step 3: Apply a greedy approach
    while fruits_covered != all_fruits:
        # Select the basket that covers the most uncovered fruits
        best_basket = max(baskets_dict, key=lambda basket: len(set(baskets_dict[basket]) - fruits_covered))
        selected_baskets.append(best_basket)
        fruits_covered.update(baskets_dict[best_basket])
        # Remove the selected basket from consideration
        del baskets_dict[best_basket]
    
    return selected_baskets

# Example usage:
baskets_dict = {
    "basket 1": ["apple", "pear", "banana"],
    "basket 2": ["banana"],
    "basket 3": ["orange", "apple", "tangerine"],
    "basket 4": ["grape", "pear"],
    "basket 5": ["orange", "grape"]
}

result = minimum_baskets_greedy(baskets_dict)
print("Selected Baskets:", result)
