import streamlit as st
from datetime import datetime, timedelta
import re
import urllib.parse

def create_vaccination_reminder(vaccine_type: str, date_str: str, bird_count: int = None, time_str: str = None) -> str:
    """
    Create a vaccination reminder description that users can manually add to their calendar.
    
    Args:
        vaccine_type: Type of vaccine (Newcastle, Gumboro, etc.)
        date_str: Date in natural language or specific format
        bird_count: Number of birds (optional)
        time_str: Time in natural language or specific format (optional)
    """
    try:
        # Parse date from natural language
        reminder_date = parse_date_from_text(date_str)
        
        if not reminder_date:
            return "⚠️ I couldn't understand the date. Please specify like 'next Monday' or 'March 15th'."
        
        # Parse time or default to 6:00 AM
        reminder_time = parse_time_from_text(time_str) if time_str else "06:00"
        hour, minute = map(int, reminder_time.split(':'))
        
        # Combine date and time
        reminder_datetime = reminder_date.replace(hour=hour, minute=minute)
        
        # Create calendar-friendly reminder text
        time_display = reminder_time.replace(":00", "") + " AM" if int(reminder_time.split(':')[0]) < 12 else reminder_time + " PM"
        
        reminder_text = f"""
🐔 **Vaccination Reminder for {vaccine_type}**
        
**Date:** {reminder_datetime.strftime('%A, %B %d, %Y')}
**Time:** {time_display}
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
        google_cal_link = create_google_cal_link(vaccine_type, reminder_datetime, bird_count)
        
        return f"""
{reminder_text}

---
**Quick Add to Calendar:**
📅 [Add to Google Calendar]({google_cal_link})
📱 Or manually add to your phone calendar

*Reminder set for {reminder_datetime.strftime('%B %d, %Y')} at {time_display}*
"""
        
    except Exception as e:
        return f"⚠️ Could not create reminder: {str(e)}"

def parse_time_from_text(time_text: str) -> str:
    """Parse natural language time into HH:MM format"""
    try:
        time_text = time_text.lower().strip()
        
        # Handle common time formats
        if 'morning' in time_text:
            return "06:00"
        elif 'afternoon' in time_text:
            return "14:00"
        elif 'evening' in time_text:
            return "18:00"
        elif 'noon' in time_text or 'midday' in time_text:
            return "12:00"
        
        # Parse specific times
        time_patterns = [
            r'(\d{1,2})[:.]?(\d{2})?\s*(am|pm)?',
            r'(\d{1,2})\s*(am|pm)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, time_text)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                period = match.group(3) if match.group(3) else ''
                
                # Convert to 24-hour format
                if period == 'pm' and hour < 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                
                return f"{hour:02d}:{minute:02d}"
        
        # Default to 6:00 AM if no time specified or understood
        return "06:00"
        
    except Exception:
        return "06:00"  # Default to 6:00 AM

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

def create_google_cal_link(vaccine_type: str, datetime_obj: datetime, bird_count: int = None):
    """Create a Google Calendar link with pre-filled details for UTC+1 timezone"""
    # Create proper title with both text and emoji
    title = f"Vaccination Reminder: {vaccine_type} 🐔"
    details = f"Vaccination reminder for {vaccine_type}"
    if bird_count:
        details += f" for {bird_count} birds"
    
    # Format datetime for Google Calendar (1-hour duration)
    start_time = datetime_obj.strftime('%Y%m%dT%H%M%S')  # Use the actual time
    end_datetime = datetime_obj + timedelta(hours=1)
    end_time = end_datetime.strftime('%Y%m%dT%H%M%S')    # 1 hour later
    
    # URL encode the parameters
    encoded_title = urllib.parse.quote(title)
    encoded_details = urllib.parse.quote(details)
    
    return (f"https://calendar.google.com/calendar/render?"
            f"action=TEMPLATE&"
            f"text={encoded_title}&"
            f"details={encoded_details}&"
            f"dates={start_time}/{end_time}")
