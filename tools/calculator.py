def calculate_feed_cost(num_birds: int, feed_per_bird: float, price_per_kg: float) -> str:
    try:
        total_feed = num_birds * feed_per_bird
        total_cost = total_feed * price_per_kg
        return (
            f"🐔 Feed cost calculation:\n\n"
            f"- Birds: {num_birds}\n"
            f"- Feed per bird: {feed_per_bird} kg\n"
            f"- Total feed: {total_feed} kg\n"
            f"- Price per kg: ₦{price_per_kg}\n"
            f"- 💰 Total cost: ₦{total_cost:,.0f}"
        )
    except Exception as e:
        return f"⚠️ Could not calculate feed cost: {e}"
