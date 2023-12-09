import abc
import datetime
import typing


class Notification(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setup(
        self,
        data: typing.List[typing.Dict[str, typing.Union[str, typing.Union[float, int, datetime.datetime]]]],
        message: str,
    ) -> None:
        """
        Prepares the notification with data and a custom message.

        ## Parameters
        -------------
        data : typing.List[typing.Dict[str, typing.Union[str, typing.Union[float, int, datetime.datetime]]]]
            A list of dictionaries which represent all the detected anomaly data.

        message : str
            A custom message to be included in the notification.
        """
        pass

    @property
    @abc.abstractmethod
    def send(self) -> None:
        """
        Sends the prepared notification. This method dispatches the notification that was set up using the `setup` method.
        """
        pass
