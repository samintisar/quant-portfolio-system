"""
API server startup script.

Provides a simple way to start the FastAPI server with proper configuration.
Simple, clean implementation avoiding overengineering for resume projects.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from portfolio.logging_config import setup_logging, get_logger

def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Start Portfolio Optimization API Server"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    return parser


def validate_environment():
    """Validate that the environment is properly set up."""
    required_dirs = [
        "portfolio",
        "portfolio/api",
        "portfolio/models",
        "portfolio/optimizer",
        "portfolio/performance",
        "portfolio/data"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"❌ Missing required directories: {', '.join(missing_dirs)}")
        print("Please ensure the project structure is complete.")
        return False

    return True


def main():
    """Main function to start the API server."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level.upper())
    logger = get_logger(__name__)

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Display startup information
    print("🚀 Starting Portfolio Optimization API Server")
    print("=" * 50)
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"📊 Workers: {args.workers}")
    print(f"📝 Log Level: {args.log_level}")
    print("=" * 50)

    # API endpoints information
    print("\n📚 Available Endpoints:")
    print("  • GET  /               - API Root")
    print("  • GET  /health          - Health Check")
    print("  • GET  /docs            - Swagger Documentation")
    print("  • GET  /redoc           - ReDoc Documentation")
    print("  • POST /api/v1/portfolio/optimize    - Portfolio Optimization")
    print("  • POST /api/v1/portfolio/analyze     - Portfolio Analysis")
    print("  • GET  /api/v1/data/assets/search    - Asset Search")
    print("  • GET  /api/v1/data/assets/{symbol} - Asset Details")
    print("  • GET  /api/v1/portfolio/methods    - Optimization Methods")
    print("=" * 50)

    try:
        # Start the server
        uvicorn.run(
            "portfolio.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()