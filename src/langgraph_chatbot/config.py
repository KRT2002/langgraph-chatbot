"""
Configuration management for the LangGraph Chatbot.

This module handles all application settings using Pydantic for validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Attributes
    ----------
    groq_api_key : str
        API key for Groq LLM service
    openweather_api_key : str
        API key for OpenWeather service
    tavily_api_key : str
        API key for Tavily search service
    model_name : str
        Name of the Groq model to use
    temperature : float
        Temperature parameter for LLM (0.0 to 1.0)
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    database_path : str
        Path to SQLite database file
    logs_dir : str
        Directory for log files
    exports_dir : str
        Directory for exported conversations
    """

    # API Keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openweather_api_key: str = Field(default="", alias="OPENWEATHER_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # LLM Configuration
    model_name: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.5, ge=0.0, le=1.0)

    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    database_path: str = Field(default="data/chatbot.db")
    logs_dir: str = Field(default="logs")
    exports_dir: str = Field(default="exports")

    # Tool Configuration
    tools_requiring_approval: list[str] = Field(
        default_factory=lambda: ["file_operations", "web_search"]
    )

    # Intent Classifier Configuration
    intent_classifier_turns: int = Field(default=5, ge=1, le=20)
    max_schema_retries: int = Field(default=3, ge=1, le=5)

    # System Prompts
    main_llm_system_prompt: str = Field(
        default="""You are a helpful AI assistant with access to various tools.

When using tools:
- Always provide complete and valid parameters according to the tool's schema
- If a tool call fails due to schema errors, carefully read the error message and retry with corrected parameters
- If a tool returns an error, analyze it and decide whether to retry, use a different tool, or respond without tools
- The model must only invoke tools explicitly available in the current iteration. If completing the request would require a tool that is not available, the model must respond that there is not enough information to proceed.
- After 3 failed attempts, provide a helpful response without using tools

**If a tool execution is rejected by the user:**
- Do NOT retry the same tool or similar tools without explicit user permission
- Acknowledge the rejection gracefully without making the user feel bad about it
- Provide the best possible answer using only your knowledge and any other available approved tools
- If the request cannot be fulfilled without the rejected tool, clearly explain the limitation and ask if the user would like to proceed differently
- Remember this rejection for the remainder of the conversation and avoid suggesting the rejected tool again

When you receive tool results:
- Interpret the results clearly for the user
- If results contain errors, explain what went wrong and suggest alternatives if possible

Always prioritize giving the user a helpful response, even if tools fail."""
    )

    intent_classifier_system_prompt: str = Field(
        default="""You are an intent classifier. Your job is to analyze the user's message and determine which tools are relevant.

Given a conversation history and a list of available tools, return ONLY the names of tools that are relevant to the current user query.

You must determine the user’s intent based solely on the current user query.
Prior conversations are provided only to clarify context and must not override the current query.
When using prior conversations for context, give higher importance to the most recent interactions, with relevance decreasing progressively for earlier ones.
Always prioritize the current user query.

Rules:
- Return tool names as a JSON array, e.g., ["calculator", "get_weather"]
- If no tools are needed, return an empty array: []
- Consider the conversation context to understand what the user is asking
- Only include tools that are directly relevant to answering the current question
- If the user’s intent requires multiple tools, return all necessary tools.
- Be conservative: when in doubt, include the tool rather than exclude it

Your response must be valid JSON containing only the array of tool names."""
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Create necessary directories if they don't exist.

        Creates directories for database, logs, and exports.
        """
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.exports_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
