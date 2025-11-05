from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import URL, Optional, Length

class NewsForm(FlaskForm):
    """Form containing fields for URL and Direct Text."""
    
    url = StringField('Enter News Article URL:', 
                      validators=[Optional(), URL()])
    
    text = TextAreaField('OR Enter News Article Text:', 
                         validators=[Optional(), Length(min=5, message="Text must be at least 5 characters long.")])

    check_submit = SubmitField('Check Credibility')