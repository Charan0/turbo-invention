from fastapi import FastAPI, Header
from mail import formulate_message, send_message
from pydantic import BaseModel

app = FastAPI(title="Email Sender", description="a python program wrapped over an api endpoint that sends a mail when hit")

class EmailRequest(BaseModel):
    sender: str
    subject: str
    message_body: str
    to: str

@app.get("/")
def index():
    """
    ## Liveness test
    
    Perform a get request to check if the server is online. Should return a json response with a single key `message`
    """
    return {"message": "hit successfully"}

@app.post("/send-email")
def send_email(email_request: EmailRequest, api_key: str=Header(None), app_pass: str=Header(None)):
    """
    ## Send an email using this endpoint

    Header parameters should contain the api key for tenor (sent as `api-key` header) 
    and also the app password for the gmail account associated with the sender's email id (sent as `app-pass` header)  

    A successful response should return a `True` response body
    """
    subject = email_request.subject
    recipient, from_ = email_request.to, email_request.sender
    message_body = email_request.message_body

    print(f"api key: {api_key} app password: {app_pass}")
    try:  # trying to formulate and send message
        message = formulate_message(api_key=api_key, subject=subject, recipient=recipient, 
                                    from_=from_, parsed_msg_text=message_body)
        sent_message = send_message(app_pass=app_pass, message=message, from_=from_)
        return {"email-sent": sent_message}
    except Exception as err:
        print(f"encountered {err} in wrapper api func when trying to send mail")
    
    return {"email-sent": False}