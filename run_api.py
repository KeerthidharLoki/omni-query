"""
Run the Omni-Query API server.
Usage: python run_api.py

Windows fix: switch asyncio from ProactorEventLoop (default since Python 3.8)
to SelectorEventLoop, which uvicorn requires for socket binding on Windows.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=9100,
        reload=False,
    )
