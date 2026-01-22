"""
File operations tool for reading and writing text files.
"""

from pathlib import Path

from langchain_core.tools import tool

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)

# Safe directory for file operations
SAFE_DIR = Path("data/user_files")
SAFE_DIR.mkdir(parents=True, exist_ok=True)


@tool
def file_operations(
    operation: str,
    filename: str,
    content: str = ""
) -> dict:
    """
    Perform file operations (read, write, append, list).
    
    All files are stored in a safe directory (data/user_files/).
    
    Parameters
    ----------
    operation : str
        Operation to perform: 'read', 'write', 'append', 'list', 'delete'
    filename : str
        Name of the file (without path)
    content : str, optional
        Content for write/append operations, by default ""
        
    Returns
    -------
    dict
        Operation result with file content or status
        
    Examples
    --------
    >>> file_operations("write", "notes.txt", "Hello World")
    {'status': 'success', 'operation': 'write', 'filename': 'notes.txt'}
    """
    try:
        logger.info(f"File operation: {operation} on {filename}")
        
        # Sanitize filename to prevent directory traversal
        safe_filename = Path(filename).name
        file_path = SAFE_DIR / safe_filename
        
        if operation == "read":
            if not file_path.exists():
                logger.warning(f"File not found: {safe_filename}")
                return {
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"File '{safe_filename}' not found"
                }
            
            content = file_path.read_text(encoding="utf-8")
            logger.info(f"File read: {safe_filename} ({len(content)} chars)")
            return {
                "status": "success",
                "operation": "read",
                "filename": safe_filename,
                "content": content,
                "size": len(content)
            }
        
        elif operation == "write":
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"File written: {safe_filename} ({len(content)} chars)")
            return {
                "status": "success",
                "operation": "write",
                "filename": safe_filename,
                "size": len(content)
            }
        
        elif operation == "append":
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Content appended to: {safe_filename}")
            return {
                "status": "success",
                "operation": "append",
                "filename": safe_filename
            }
        
        elif operation == "list":
            files = [f.name for f in SAFE_DIR.glob("*") if f.is_file()]
            logger.info(f"Listed {len(files)} files")
            return {
                "status": "success",
                "operation": "list",
                "files": files,
                "count": len(files)
            }
        
        elif operation == "delete":
            if not file_path.exists():
                return {
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"File '{safe_filename}' not found"
                }
            
            file_path.unlink()
            logger.info(f"File deleted: {safe_filename}")
            return {
                "status": "success",
                "operation": "delete",
                "filename": safe_filename
            }
        
        else:
            logger.warning(f"Invalid operation: {operation}")
            return {
                "status": "error",
                "error_type": "invalid_operation",
                "message": f"Invalid operation '{operation}'. Use: read, write, append, list, delete"
            }
        
    except Exception as e:
        logger.error(f"File operation error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e)
        }