from great_expectations.checkpoint.actions import (
    UpdateDataDocsAction,
    # SlackNotificationAction,
    # EmailAction,
)


def action_list():
    actions = [
        # builds and updates data docs
        UpdateDataDocsAction(name="update_data_docs"),
        # SlackNotificationAction(
        #     webhook_url="https://hooks.slack.com/services/your/webhook/url",
        #     notify_on="failure",
        # ),
        # EmailAction(
        #     mail_from="your_email@example.com",
        #     mail_to=["team@example.com"],
        #     smtp_host="smtp.example.com",
        #     smtp_port=587,
        #     smtp_starttls=True,
        #     smtp_ssl=False,
        #     smtp_user="smtp_user",
        #     smtp_password="smtp_password",
        #     notify_on="failure",
        # ),
    ]
    return actions
