# 🤖 LangGraph Chatbot

A production-ready AI chatbot built with LangGraph, featuring multiple tools, conversation management, human-in-the-loop controls, and comprehensive analytics.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red.svg)

## ✨ Features

### 🛠️ Multi-Tool Support
- **Calculator** - Basic arithmetic operations
- **Weather** - Real-time weather information via OpenWeatherMap
- **Unit Converter** - Temperature, length, and weight conversions
- **Date/Time Utilities** - Timezone conversion and date calculations
- **File Operations** - Read, write, and manage text files
- **Web Search** - Internet search via Tavily API

### 🎯 Advanced AI Features
- **Intent Classification** - Automatically determines which tools are relevant for each query
- **Smart Tool Filtering** - Only exposes relevant tools to the LLM per conversation turn
- **Schema Error Recovery** - Automatic retry with feedback for malformed tool calls (max 3 attempts)
- **Intelligent Fallback** - Falls back to non-tool responses after retry limit
- **Structured Error Handling** - Tools return structured errors that the LLM can reason about

### 💬 Conversation Management
- **Persistent Storage** - SQLite-based conversation history
- **Smart Titles** - Auto-generated conversation titles
- **Thread Management** - Easy navigation between conversations
- **Export Options** - Export to JSON, Markdown, or PDF

### 🔒 Human-in-the-Loop
- **Tool Approval** - Require user approval for sensitive operations
- **Configurable** - Enable/disable per session
- **Fine-grained Control** - Specify which tools require approval

### 📊 Analytics
- **Message Statistics** - Track user/assistant message counts
- **Tool Usage** - Monitor which tools are being used
- **Token Approximation** - Estimate token consumption
- **Real-time Updates** - Live statistics in collapsible sidebar

### 🎨 User Interface
- **Clean Design** - Intuitive Streamlit interface
- **Collapsible Sidebar** - Analytics and conversation history
- **Streaming Responses** - Real-time AI responses
- **Status Indicators** - Visual feedback for tool execution

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                            │
│  (User Input, Message Display, Analytics, Export)                   │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│                   LangGraph Workflow                                │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐                   │
│  │  START   │───▶│   Intent     │───▶│ ChatNode │◄─────┐           │
│  └──────────┘    │ Classifier   │    └────┬─────┘      │            │
│                  └──────────────┘         │            │            │
│                  (Filters tools           │            │            │
│                   once per turn)          ▼            │            │
│                                     ┌──────────────┐   │            │
│                                     │ToolsCondition│   │            │
│                                     └──────┬───────┘   │            │
│                                            │           │            │
│                                            ▼           │            │
│                                      ┌──────────┐      │            │
│                                      │ ToolNode │──────┘            │
│                                      └──────────┘                   │
│                                      (Loop: ChatNode ↔ ToolNode)    │
│                                                                     │
│   Features: Intent Classification, Schema Retry, Error Recovery     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│                   Tool Collection                                   │
│  • Calculator  • Weather  • Unit Converter                          │
│  • DateTime    • File Ops • Web Search                              │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- UV package manager (recommended) or Poetry

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KRT2002/langgraph-chatbot.git
cd langgraph-chatbot
```

2. **Install dependencies using UV**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

4. **Run the application**
```bash
streamlit run src/ui/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔑 API Keys Setup

### Groq API (Required)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up and get your API key
3. Add to `.env` as `GROQ_API_KEY`

### OpenWeatherMap (Optional - for weather tool)
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for free tier
3. Add to `.env` as `OPENWEATHER_API_KEY`

### Tavily (Optional - for web search)
1. Visit [Tavily](https://tavily.com/)
2. Get your API key
3. Add to `.env` as `TAVILY_API_KEY`

## 📖 Usage Guide

### Starting a Conversation
1. Click "➕ New Chat" in the sidebar
2. Type your message in the input box
3. The conversation will automatically get a title based on your first message

### Using Tools
Simply ask the chatbot to perform tasks:
- *"What's the weather in London?"*
- *"Convert 100 celsius to fahrenheit"*
- *"Calculate 15 multiplied by 8"*
- *"What's the current time in Tokyo?"*
- *"Search the web for latest AI news"*

### Human-in-the-Loop
1. Enable "🔒 Human-in-Loop" toggle in sidebar
2. Tools requiring approval (file operations, web search) will show a warning
3. Review the tool usage before proceeding

### Exporting Conversations
1. Open "💾 Export Conversation" in sidebar
2. Choose format: JSON, Markdown, or PDF
3. Files are saved in the `exports/` directory

### Viewing Analytics
1. Enable "📊 Show Analytics" toggle
2. View real-time statistics including:
   - Message counts
   - Tool usage
   - Approximate token count

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
class Settings(BaseSettings):
    # LLM Configuration
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.5
    
    # Intent Classification
    intent_classifier_turns: int = 5  # Last N turns to analyze
    
    # Schema Error Handling
    max_schema_retries: int = 3  # Max retry attempts
    
    # Tools requiring approval
    tools_requiring_approval: list[str] = [
        "file_operations", 
        "web_search"
    ]
```

## 📁 Project Structure

```
langgraph-chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py           # State definition
│   │   ├── nodes.py           # Graph nodes
│   │   ├── intent_classifier.py  # Intent classification node
│   │   └── workflow.py        # Workflow construction
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py      # Arithmetic operations
│   │   ├── weather.py         # Weather information
│   │   ├── unit_converter.py  # Unit conversions
│   │   ├── datetime_utils.py  # Date/time utilities
│   │   ├── file_operations.py # File management
│   │   └── web_search.py      # Web search
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          # Logging configuration
│   │   ├── analytics.py       # Statistics calculation
│   │   ├── export.py          # Export functionality
│   │   └── tool_descriptions.py  # Tool description extraction
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py   # Streamlit interface
├── data/                       # SQLite database & user files
├── logs/                       # Application logs
├── exports/                    # Exported conversations
├── .env.example               # Environment template
├── .gitignore
├── pyproject.toml             # Project dependencies
└── README.md
```

## 🛠️ Adding Custom Tools

1. **Create a new tool file** in `src/tools/`:

```python
from langchain_core.tools import tool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def my_custom_tool(param: str) -> dict:
    """
    Description of what the tool does.
    
    Parameters
    ----------
    param : str
        Parameter description
        
    Returns
    -------
    dict
        Result dictionary with status field
    """
    try:
        logger.info(f"Custom tool called with: {param}")
        # Your logic here
        return {"status": "success", "result": "value"}
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e)
        }
```

2. **Add to tool collection** in `src/tools/__init__.py`:

```python
from src.tools.my_custom_tool import my_custom_tool

ALL_TOOLS = [
    calculator,
    get_weather,
    # ... other tools
    my_custom_tool,  # Add here
]
```

3. **Configure approval requirement** (optional) in `src/config.py`:

```python
tools_requiring_approval: list[str] = [
    "file_operations",
    "web_search",
    "my_custom_tool",  # If approval needed
]
```

## 📊 Logging

Logs are stored in the `logs/` directory with timestamps:
- Format: `chatbot_YYYYMMDD_HHMMSS.log`
- Includes detailed information about tool usage, errors, and conversation flow
- Configurable log level in `.env` (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## 🧪 Development

### Code Style
The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **NumPy-style** docstrings

Run formatting:
```bash
uv run black src/
uv run ruff check src/
```

### Type Hints
All functions include comprehensive type hints for better IDE support and code quality.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - For the graph-based LLM framework
- [Groq](https://groq.com/) - For fast LLM inference
- [Streamlit](https://streamlit.io/) - For the intuitive UI framework
- [OpenWeatherMap](https://openweathermap.org/) - For weather data
- [Tavily](https://tavily.com/) - For web search capabilities

---

**Note**: This is a demonstration project showcasing LangGraph capabilities with tool integration, conversation management, and human-in-the-loop patterns. For production use, ensure proper security measures, rate limiting, and error handling.