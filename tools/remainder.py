import streamlit as st
from datetime import datetime, timedelta
import re

def create_vaccination_reminder(vaccine_type: str, date_str: str, bird_count: int = None) -> str:
    """
    Create a vaccination reminder description that users can manually add to their calendar.
    
    Args:
        vaccine_type: Type of vaccine (Newcastle, Gumboro, etc.)
        date_str: Date in natural language or specific format
        bird_count: Number of birds (optional)
    """
    try:
        # Parse date from natural language
        reminder_date = parse_date_from_text(date_str)
        
        if not reminder_date:
            return "⚠️ I couldn't understand the date. Please specify like 'next Monday' or 'March 15th'."
        
        # Create calendar-friendly reminder text
        reminder_text = f"""
🐔 **Vaccination Reminder for {vaccine_type}**
        
**Date:** {reminder_date.strftime('%A, %B %d, %Y')}
**Vaccine:** {vaccine_type}
{'**Number of Birds:** ' + str(bird_count) if bird_count else ''}

**Instructions:**
1. Prepare vaccine according to manufacturer instructions
2. Ensure birds are healthy before vaccination
3. Follow proper administration method
4. Monitor birds for 24 hours after vaccination

💡 *Set this reminder in your calendar app*
        """
        
        # Provide direct calendar links
        google_cal_link = create_google_cal_link(vaccine_type, reminder_date, bird_count)
        
        return f"""
{reminder_text}

---
**Quick Add to Calendar:**
📅 [Add to Google Calendar]({google_cal_link})
📱 Or manually add to your phone calendar

*Reminder set for {reminder_date.strftime('%B %d, %Y')}*
"""
        
    except Exception as e:
        return f"⚠️ Could not create reminder: {str(e)}"

def parse_date_from_text(date_text: str):
    """Parse natural language dates into datetime objects"""
    try:
        date_text = date_text.lower().strip()
        
        # Handle relative dates
        if 'tomorrow' in date_text:
            return datetime.now() + timedelta(days=1)
        elif 'next week' in date_text:
            return datetime.now() + timedelta(days=7)
        elif 'in 2 weeks' in date_text:
            return datetime.now() + timedelta(days=14)
        elif 'in 1 month' in date_text:
            return datetime.now() + timedelta(days=30)
        
        # Handle specific day names
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in date_text:
                days_ahead = (i - datetime.now().weekday() + 7) % 7
                if days_ahead == 0:  # Today is that day
                    days_ahead = 7
                return datetime.now() + timedelta(days=days_ahead)
        
        # Try to parse specific dates
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d', '%b %d']:
            try:
                # Add current year if not specified
                if not any(str(datetime.now().year) in date_text for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']):
                    date_text_with_year = f"{date_text} {datetime.now().year}"
                else:
                    date_text_with_year = date_text
                    
                return datetime.strptime(date_text_with_year, fmt)
            except ValueError:
                continue
                
        return None
        
    except Exception:
        return None

def create_google_cal_link(vaccine_type: str, date: datetime, bird_count: int = None):
    """Create a Google Calendar link with pre-filled details"""
    title = f"🐔 Vaccination: {vaccine_type}"
    details = f"Vaccination reminder for {vaccine_type}"
    if bird_count:
        details += f" for {bird_count} birds"
    
    # Format for Google Calendar
    start_time = date.strftime('%Y%m%dT090000')  # 9:00 AM
    end_time = date.strftime('%Y%m%dT100000')    # 10:00 AM
    
    return (f"https://calendar.google.com/calendar/render?"
            f"action=TEMPLATE&"
            f"text={title}&"
            f"details={details}&"
            f"dates={start_time}/{end_time}")
