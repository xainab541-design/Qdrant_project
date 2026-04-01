import json
import random
from datetime import datetime, timedelta

def generate_ecommerce_data(num_records: int = 50) -> list[dict]:
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    adjectives = ["Wireless", "Smart", "Portable", "Ergonomic", "Durable", "Classic", "Modern"]
    nouns = ["Headphones", "Speaker", "Monitor", "Keyboard", "Mouse", "Jacket", "Sneakers", "Blender", "Coffee Maker", "Tent"]
    
    data = []
    for i in range(num_records):
        category = random.choice(categories)
        name = f"{random.choice(adjectives)} {random.choice(nouns)}"
        
        description = f"This {name.lower()} is perfect for your needs. "
        if category == "Electronics":
            description += "It features long battery life, fast charging, and seamless connectivity. Enjoy high-quality performance wherever you go."
        elif category == "Clothing":
            description += "Made from premium materials, it offers comfort and style for everyday wear. Available in multiple sizes and colors."
        elif category == "Home & Kitchen":
            description += "Upgrade your home setup with this efficient and easy-to-use appliance. Easy to clean and store."
        else:
            description += "A reliable and versatile choice that delivers excellent value. Highly rated by our customers."
            
        timestamp = (datetime.utcnow() - timedelta(days=random.randint(0, 365))).isoformat() + "Z"
        
        item = {
            "id": f"prod_{i:04d}",
            "title": name,
            "content": description,
            "category": category,
            "source_url": f"https://example-shop.com/products/{name.lower().replace(' ', '-')}",
            "timestamp": timestamp,
            "price": round(random.uniform(10.0, 500.0), 2)
        }
        data.append(item)
    return data

if __name__ == "__main__":
    records = generate_ecommerce_data(50)
    with open("dummy_data.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"Generated {len(records)} dummy records in dummy_data.json")
