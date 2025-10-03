from langchain.tools import tool

@tool
def feed_cost_calculator(bird_count: int, feed_per_bird_kg: float, price_per_kg: float) -> str:
    """
    Calculate total feed cost for broilers.
    
    Args:
        bird_count (int): Number of birds.
        feed_per_bird_kg (float): Estimated feed consumption per bird in kg.
        price_per_kg (float): Cost of feed per kg.
    
    Returns:
        str: Total feed cost in local currency (₦).
    """
    try:
        total_feed = bird_count * feed_per_bird_kg
        total_cost = total_feed * price_per_kg
        return f"For {bird_count} birds, with {feed_per_bird_kg} kg feed per bird at ₦{price_per_kg}/kg, total feed cost = ₦{total_cost:,.2f}."
    except Exception as e:
        return f"Error calculating feed cost: {str(e)}"

