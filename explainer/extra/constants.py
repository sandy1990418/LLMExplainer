from enum import Enum


class StringEnum(str, Enum):
    def __str__(self):
        return self.value


class ModelExplanationTask(StringEnum):
    """Custom model task constants extending ModelTask.

    Includes additional tasks and modifies existing ones.
    """

    # Inherit existing tasks
    QUESTION_ANSWERING = 'question_answering'
    SUMMARIZATIONS = 'summarizations'
    GENERATIVE_TEXT_CHAT = 'generative_text_chat'
    TANSLATION = "translation"
    GENERATIVE_TEXT_CHAT_WITH_ASSISTANT = "generative_text_chat_assistant"
