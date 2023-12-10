import unittest

from anomalytics import get_notification
from anomalytics.notifications.abstract import Notification
from anomalytics.notifications.email import EmailNotification
from anomalytics.notifications.slack import SlackNotification


class TestFactoryNotification(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_construct_email_notification_from_factory_design_pattern(self):
        email_notification = get_notification(
            platform="email",
            sender_address="nino@test.com",
            password="this-is-test-password",
            recipient_addresses=["adam.roe@test.com", "examination.office@test.com"],
            smtp_host="smtp.nino.com",
            smtp_port=587,
        )

        self.assertTrue(expr=issubclass(type(email_notification), Notification))
        self.assertIsInstance(obj=email_notification, cls=EmailNotification)
        self.assertEqual(first=str(email_notification), second="Email Notification")

    def test_construct_slack_notification_from_factory_design_pattern(self):
        slack_notification = get_notification(platform="slack", webhook_url="this-is123456-fake-webhook-to-slack16")

        self.assertTrue(expr=issubclass(type(slack_notification), Notification))
        self.assertIsInstance(obj=slack_notification, cls=SlackNotification)
        self.assertEqual(first=str(slack_notification), second="Slack Notification")

    def tearDown(self) -> None:
        return super().tearDown()
