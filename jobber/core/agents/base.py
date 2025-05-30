import json
import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
from dotenv import load_dotenv

from jobber.config.model_config import get_model_config, get_default_model
from jobber.core.skills.get_screenshot import get_screenshot
from jobber.utils.extract_json import extract_json
from jobber.utils.function_utils import get_function_schema
from jobber.utils.logger import logger


class BaseAgent:
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        tools: Optional[List[Tuple[Callable, str]]] = None,
        model: Optional[str] = None,
        api_call_delay: float = 1.0,  # Default 1 second delay between API calls
        max_retries: int = 3,  # Default number of retries
        initial_retry_delay: float = 0.1,  # Initial delay in seconds
        max_retry_delay: float = 10.0,  # Maximum delay in seconds
    ):
        load_dotenv()
        self.name = self.__class__.__name__
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools_list = []
        self.executable_functions_list = {}
        self.api_call_delay = api_call_delay
        self.last_api_call_time = 0
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Get model configuration
        model_config = get_model_config(model)
        self.llm_config = {
            "model": model or get_default_model(),
            "api_base": model_config["api_base"],
            "temperature": model_config["temperature"],
            "max_tokens": model_config["max_tokens"],
            "top_p": model_config["top_p"],
        }
        
        if tools:
            self._initialize_tools(tools)
            self.llm_config.update({"tools": self.tools_list, "tool_choice": "auto"})

    def _initialize_tools(self, tools: List[Tuple[Callable, str]]):
        for function, description in tools:
            self.tools_list.append(
                get_function_schema(function, description=description)
            )
            self.executable_functions_list[function.__name__] = function

    def _process_message_content(self, content: Any) -> str:
        """
        Process message content to ensure it's a string for Groq API.
        If content is a list (containing text and image), extract only the text.
        """
        if isinstance(content, list):
            # Extract text content from the list
            text_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content.append(item.get("text", ""))
            return " ".join(text_content)
        elif isinstance(content, str):
            return content
        else:
            return str(content)

    async def _wait_for_rate_limit(self):
        """Wait for the configured delay between API calls"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.api_call_delay:
            await asyncio.sleep(self.api_call_delay - time_since_last_call)
        self.last_api_call_time = asyncio.get_event_loop().time()

    async def _make_api_call_with_retry(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Make API call with retry logic for rate limit errors.
        Uses exponential backoff with jitter.
        """
        retry_count = 0
        while True:
            try:
                await self._wait_for_rate_limit()
                return litellm.completion(
                    messages=messages,
                    **self.llm_config,
                    metadata={
                        "run_name": f"{self.name}Run",
                    },
                )
            except litellm.RateLimitError as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded. Last error: {str(e)}")
                    raise

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.initial_retry_delay * (2 ** (retry_count - 1)),
                    self.max_retry_delay
                )
                # Add jitter (±20%)
                jitter = delay * 0.2
                delay = delay + random.uniform(-jitter, jitter)
                
                logger.info(f"Rate limit hit. Retrying in {delay:.2f} seconds (attempt {retry_count}/{self.max_retries})")
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Error in LLM call: {str(e)}")
                raise

    async def generate_reply(
        self, messages: List[Dict[str, Any]], sender
    ) -> Dict[str, Any]:
        # Process messages to ensure content is string
        processed_messages = []
        for msg in messages:
            processed_msg = msg.copy()
            processed_msg["content"] = self._process_message_content(msg["content"])
            processed_messages.append(processed_msg)

        self.messages.extend(processed_messages)
        self.messages = self._process_messages(self.messages)

        while True:
            litellm.logging = False
            litellm.success_callback = ["langsmith"]
            try:
                response = await self._make_api_call_with_retry(self.messages)
            except Exception as e:
                logger.error(f"Error in LLM call: {str(e)}")
                raise

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                self.messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.executable_functions_list[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    try:
                        function_response = await function_to_call(**function_args)
                        self.messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(function_response),
                            }
                        )
                    except Exception as e:
                        logger.info(f"***** Error occurred: {str(e)}")
                        self.messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(
                                    "The tool responded with an error, please try again with a diffrent tool or modify the parameters of the tool",
                                    function_response,
                                ),
                            }
                        )
                continue

            content = response_message.content
            if "##TERMINATE TASK##" in content or "## TERMINATE TASK ##" in content:
                return {
                    "terminate": True,
                    "content": content.replace("##TERMINATE TASK##", "").strip(),
                }
            else:
                try:
                    extracted_response = extract_json(content)
                    if extracted_response.get("terminate") == "yes":
                        return {
                            "terminate": True,
                            "content": extracted_response.get("final_response"),
                        }
                    else:
                        return {
                            "terminate": False,
                            "content": extracted_response.get("next_step"),
                        }
                except Exception as e:
                    logger.info(
                        f"navigator did not send ##Terminate task## error - {e} & content - {content}"
                    )
                    return {
                        "terminate": True,
                        "content": content,
                    }

    def send(self, message: str, recipient):
        return recipient.receive(message, self)

    async def receive(self, message: str, sender):
        reply = await self.generate_reply(
            [{"role": "assistant", "content": message}], sender
        )
        return self.send(reply["content"], sender)

    async def process_query(self, query: str) -> Dict[str, Any]:
        try:
            screenshot = await get_screenshot()
            return await self.generate_reply(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{query} \nHere is a screenshot of the current browser page",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{screenshot}"},
                            },
                        ],
                    }
                ],
                None,
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return {"terminate": True, "content": f"Error: {str(e)}"}

    def reset_messages(self):
        self.messages = [self.messages[0]]  # Keep the system message

    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_messages = []

        # find the latest user message in the messages array
        last_user_message_index = next(
            (
                i
                for i in reversed(range(len(messages)))
                if messages[i]["role"] == "user"
            ),
            -1,
        )

        # remove image and the supporting text of "Here is a screenshot of the current browser page" from previous messages
        for i, message in enumerate(messages):
            if message["role"] == "user":
                if isinstance(message.get("content"), list):
                    new_content = []
                    for item in message["content"]:
                        if item["type"] == "text":
                            if i != last_user_message_index:
                                # Remove the screenshot text if it's not the last user message
                                item["text"] = (
                                    item["text"]
                                    .replace(
                                        "Here is a screenshot of the current browser page",
                                        "",
                                    )
                                    .strip()
                                )
                            new_content.append(item)
                        elif (
                            item["type"] == "image_url" and i == last_user_message_index
                        ):
                            new_content.append(item)
                    message["content"] = new_content
            processed_messages.append(message)

        # Ensure the system message is always included
        if processed_messages and processed_messages[0]["role"] != "system":
            system_message = next((m for m in messages if m["role"] == "system"), None)
            if system_message:
                processed_messages.insert(0, system_message)

        return processed_messages
