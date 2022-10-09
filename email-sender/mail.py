import smtplib
from email.message import EmailMessage
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tenor import get_gif


def generate_html(content: dict):
    environ = Environment(loader=FileSystemLoader(searchpath="."), 
                          autoescape=select_autoescape())
    template = environ.get_template("card.html")
    return template.render(content)


def formulate_message(api_key: str, subject: str, recipient: str, from_: str="srichu.kattamuru@gmail.com", 
                      parsed_msg_text: str=None, search_item: str="spongebob birthday"):
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_
    message["To"] = recipient
    print(f"using email-address: {from_} sending to: {recipient}")
    header = "Well, hello there!\nHappy birthday!"
    signature = "--\nsent from a server (somewhere)\nby \"Charan 2022\"\nba-byeees and see ya\n"

    query_params = {"q": search_item, "key": api_key, "limit": 10, "media_filter": "basic", "contentfilter": "high"}
    gif_url = get_gif(api_key, query_params, url_only=True)

    try:
        if parsed_msg_text is None:
            with open("./message.txt", "r") as f:
                parsed_msg_text = f.read()

        html_content = {"header": header, "message": parsed_msg_text, "signature": signature, "gif_src": gif_url}
        html = generate_html(html_content)
        message.set_content(parsed_msg_text)
        message.add_alternative(html, subtype="html")

        return message
    except Exception as err:
        print(f"encountered error {err} when trying to send email")

    return None

def send_message(app_pass: str, message: EmailMessage, from_: str):
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_, app_pass)
            smtp.send_message(message)
        return True
    
    except Exception as err:
        print(f"encountered error {err} when trying to send message")
    
    return False