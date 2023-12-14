import unittest
from unittest.mock import MagicMock, patch

import pytest

from anomalytics.notifications.abstract import Notification
from anomalytics.notifications.email import EmailNotification


@pytest.mark.usefixtures("get_sample_1_detection_summary")
class TestEmailNotification(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gmail_notification = EmailNotification(
            sender_address="example@gmail.com",
            password="password",
            recipient_addresses=["recipient1@gmail.com", "recipient2@web.de"],
            smtp_host="smtp.gmail.com",
            smtp_port=587,
        )
        self.webde_notification = EmailNotification(
            sender_address="example@web.de",
            password="password",
            recipient_addresses=["recipient1@gmail.com", "recipient2@web.de"],
            smtp_host="smtp.web.de",
            smtp_port=587,
        )
        self.test_message = "Test notification message"

    def test_instance_is_abstract_class(self):
        expected_subject = "🤖 Anomalytics - Anomaly Detected!"
        self.assertIsInstance(obj=self.gmail_notification, cls=Notification)
        self.assertEqual(first=type(self.gmail_notification._EmailNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.gmail_notification._EmailNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.gmail_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.gmail_notification._EmailNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.gmail_notification._EmailNotification__subject, second=expected_subject  # type: ignore
        )

        self.assertIsInstance(obj=self.webde_notification, cls=Notification)
        self.assertEqual(first=type(self.webde_notification._EmailNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.webde_notification._EmailNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.webde_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.webde_notification._EmailNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.webde_notification._EmailNotification__subject, second=expected_subject  # type: ignore
        )

    def test_string_method(self):
        self.assertTrue(expr=str(self.gmail_notification) == "Email Notification")
        self.assertTrue(expr=str(self.webde_notification) == "Email Notification")

    def test_setup_with_wrong_data_type(self):
        with self.assertRaises(expected_exception=TypeError):
            self.gmail_notification.setup(data=self.sample_1_detection_summary.values, message=self.test_message)  # type: ignore

        with self.assertRaises(expected_exception=TypeError):
            self.webde_notification.setup(data=self.sample_1_detection_summary.values, message=self.test_message)  # type: ignore

    def test_setup_with_message(self):
        expected_payload = (
            "Test notification message\n\n"
            "Date: 2023-01-10\n\n"
            "Row: 9\n\n"
            "Anomaly: 75521\n\n"
            "Anomaly Score: 8.123\n\n"
            "Anomaly Threshold: 7.3"
        )

        self.gmail_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore
        self.webde_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore

        self.assertNotEqual(first=self.gmail_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertNotEqual(first=self.webde_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=self.gmail_notification._EmailNotification__payload, second=expected_payload)  # type: ignore
        self.assertEqual(first=self.webde_notification._EmailNotification__payload, second=expected_payload)  # type: ignore

    def test_setup_without_message(self):
        expected_payload = (
            "Date: 2023-01-10\n\nRow: 9\n\nAnomaly: 75521\n\nAnomaly Score: 8.123\n\nAnomaly Threshold: 7.3"
        )

        self.gmail_notification.setup(detection_summary=self.sample_1_detection_summary, message=None)  # type: ignore
        self.webde_notification.setup(detection_summary=self.sample_1_detection_summary, message=None)  # type: ignore

        self.assertNotEqual(first=self.gmail_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertNotEqual(first=self.webde_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=self.gmail_notification._EmailNotification__payload, second=expected_payload)  # type: ignore
        self.assertEqual(first=self.webde_notification._EmailNotification__payload, second=expected_payload)  # type: ignore

    @patch("anomalytics.notifications.email.SMTP")
    def test_send_gmail(self, mock_smtp):
        mock_smtp.return_value = MagicMock()

        self.gmail_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore
        self.gmail_notification.send

        mock_smtp.assert_called_with(host="smtp.gmail.com", port=587)
        mock_smtp.return_value.starttls.assert_called_once()
        mock_smtp.return_value.login.assert_called_once_with(user="example@gmail.com", password="password")
        mock_smtp.return_value.quit.assert_called_once()

    @patch("anomalytics.notifications.email.SMTP")
    def test_send_webde(self, mock_smtp):
        mock_smtp.return_value = MagicMock()

        self.webde_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore
        self.webde_notification.send

        mock_smtp.assert_called_with(host="smtp.web.de", port=587)
        mock_smtp.return_value.starttls.assert_called_once()
        mock_smtp.return_value.login.assert_called_once_with(user="example@web.de", password="password")
        mock_smtp.return_value.quit.assert_called_once()

    def tearDown(self) -> None:
        return super().tearDown()
