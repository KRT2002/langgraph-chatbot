"""
Conversation export utilities for JSON, Markdown, and PDF formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from langgraph_chatbot.config import settings
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def export_conversation_json(
    messages: list[Any],
    thread_id: str,
    title: str = "Conversation"
) -> str:
    """
    Export conversation to JSON format.
    
    Parameters
    ----------
    messages : list[Any]
        List of conversation messages
    thread_id : str
        Thread identifier
    title : str, optional
        Conversation title, by default "Conversation"
        
    Returns
    -------
    str
        Path to exported JSON file
    """
    try:
        logger.info(f"Exporting conversation to JSON: {thread_id}")
        
        export_data = {
            "thread_id": thread_id,
            "title": title,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": []
        }
        
        for msg in messages:
            msg_data = {
                "type": msg.__class__.__name__,
                "content": msg.content,
            }
            
            if isinstance(msg, ToolMessage):
                msg_data["name"] = getattr(msg, "name", "unknown")
            
            export_data["messages"].append(msg_data)
        
        filename = f"conversation_{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(settings.exports_dir) / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported to JSON: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"JSON export error: {str(e)}", exc_info=True)
        raise


def export_conversation_markdown(
    messages: list[Any],
    thread_id: str,
    title: str = "Conversation"
) -> str:
    """
    Export conversation to Markdown format.
    
    Parameters
    ----------
    messages : list[Any]
        List of conversation messages
    thread_id : str
        Thread identifier
    title : str, optional
        Conversation title, by default "Conversation"
        
    Returns
    -------
    str
        Path to exported Markdown file
    """
    try:
        logger.info(f"Exporting conversation to Markdown: {thread_id}")
        
        md_lines = [
            f"# {title}",
            "",
            f"**Thread ID:** {thread_id}  ",
            f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Messages:** {len(messages)}",
            "",
            "---",
            ""
        ]
        
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                md_lines.append(f"### 👤 User (Message {i})")
                md_lines.append("")
                md_lines.append(msg.content)
            elif isinstance(msg, AIMessage):
                md_lines.append(f"### 🤖 Assistant (Message {i})")
                md_lines.append("")
                md_lines.append(msg.content)
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                md_lines.append(f"### 🔧 Tool: {tool_name} (Message {i})")
                md_lines.append("")
                md_lines.append(f"```\n{msg.content}\n```")
            
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        
        filename = f"conversation_{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = Path(settings.exports_dir) / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        
        logger.info(f"Exported to Markdown: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Markdown export error: {str(e)}", exc_info=True)
        raise


def export_conversation_pdf(
    messages: list[Any],
    thread_id: str,
    title: str = "Conversation"
) -> str:
    """
    Export conversation to PDF format.
    
    Parameters
    ----------
    messages : list[Any]
        List of conversation messages
    thread_id : str
        Thread identifier
    title : str, optional
        Conversation title, by default "Conversation"
        
    Returns
    -------
    str
        Path to exported PDF file
    """
    try:
        logger.info(f"Exporting conversation to PDF: {thread_id}")
        
        filename = f"conversation_{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = Path(settings.exports_dir) / filename
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor="navy",
            spaceAfter=30
        )
        
        header_style = ParagraphStyle(
            "CustomHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor="darkblue",
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Metadata
        metadata = f"<b>Thread ID:</b> {thread_id}<br/>"
        metadata += f"<b>Exported:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        metadata += f"<b>Messages:</b> {len(messages)}"
        story.append(Paragraph(metadata, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))
        
        # Messages
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                story.append(Paragraph(f"👤 User (Message {i})", header_style))
                story.append(Paragraph(msg.content, styles["Normal"]))
            elif isinstance(msg, AIMessage):
                story.append(Paragraph(f"🤖 Assistant (Message {i})", header_style))
                story.append(Paragraph(msg.content, styles["Normal"]))
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                story.append(Paragraph(f"🔧 Tool: {tool_name} (Message {i})", header_style))
                story.append(Paragraph(f"<pre>{msg.content}</pre>", styles["Code"]))
            
            story.append(Spacer(1, 0.2 * inch))
        
        doc.build(story)
        logger.info(f"Exported to PDF: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"PDF export error: {str(e)}", exc_info=True)
        raise